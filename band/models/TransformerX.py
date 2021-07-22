import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for src simplicify the norm is first as oppposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply a residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feedforward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = key.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nwords = key.size(-2)  # Note: nwords is max_words in a sent
        nwords_query = query.size(-2)
        dmodel_dim_idx, attn_head_dim_idx, nword_dim_idx = -1, -2, -3

        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.view(-1, nwords)
            mask = mask.unsqueeze(-2).unsqueeze(-2)
            # mask: [-1, 1, 1, nwords]

        # 1) Do all the linear projections in batch from d_model -> h x d_k
        # nwords_query can be != nwords
        query = self.linears[0](query).view(-1, nwords_query, self.h, self.d_k).transpose(nword_dim_idx,
                                                                                          attn_head_dim_idx)
        key, value = \
            [l(x).view(-1, nwords, self.h, self.d_k).transpose(nword_dim_idx, attn_head_dim_idx)
             for l, x in zip(self.linears[1:3], (key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) Concat using a view and apply a final linear
        x = x.transpose(nword_dim_idx, attn_head_dim_idx).contiguous().view(-1, nwords_query, self.h * self.d_k)

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

        self.pe.requires_grad = False  # TODO: correct?

    def forward(self, x):
        x = x + self.pe[:, :x.size(-2)]  # dim=-2 instead of 1 for more flexibility
        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, d_model, d_out, attn):
        super(Decoder, self).__init__()

        self.linear = nn.Linear(d_model, d_out)
        self.src_attn = attn

    def forward(self, x, memory, src_mask):
        out = self.src_attn(x, memory, memory, src_mask)
        out = self.linear(out)

        return out


class EncoderCovertDecoder(nn.Module):
    def __init__(self, encoder, decoder, converter, to_decode=True):
        super(EncoderCovertDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.converter = converter
        self.to_decode = to_decode

    def forward(self, x, src_mask, query=None):
        m = self.encode(x, src_mask)
        # average to get representative
        if query is None:
            # getting a query by averaging using `converter`
            query = self.converter(m, src_mask)
            # only 1 after averaging with converter -> unsqueeze
            query = query.unsqueeze(-2)

        if not self.to_decode:
            return query.squeeze(-2)

        # only 1 after decoding -> squeeze
        out = self.decode(query, m, src_mask).squeeze(-2)

        return out

    def encode(self, x, mask):
        return self.encoder(x, mask)

    def decode(self, x, memory, mask):
        return self.decoder(x, memory, mask)


class EncoderCovertEncoderCovertDecoder(nn.Module):
    def __init__(self, d_emb, d_out1, h1, d_out2, h2, pe_dropout=0.1, **kwargs):
        super(EncoderCovertEncoderCovertDecoder, self).__init__()
        self.model1 = make_model(d_model=d_emb, h=h1, d_out=d_out1, **kwargs)
        self.model2 = make_model(d_model=d_out1, h=h2, d_out=d_out2, **kwargs)
        self.pe = PositionalEncoding(d_model=d_emb, dropout=pe_dropout)

    def forward(self, x, mask):
        word_mask = mask  # word_mask [bs, nsent, nword]
        sent_mask = (word_mask.sum(dim=2) != 0.)  # sent_mask [bs, nsent], dim=2 because want to sum over nword dim
        # get embeddings
        # xb = embeddings(xb)  # this should already happen before passing x into this model

        # add positional encoding
        # TODO: disable PE?
        # x = self.pe(x)

        # get sizes()
        bs, nsents, nwords, demb = x.size()

        # prepare sizes to feed to the first model
        x = x.view(-1, nwords, demb)
        word_mask = word_mask.view(-1, nwords)
        x = self.model1(x, word_mask)

        # prepare sizes to feed to the second model
        demb = x.size(-1)
        x = x.view(-1, nsents, demb)
        sent_mask = sent_mask.view(-1, nsents)
        x = self.model2(x, sent_mask)

        return x


class Converter(nn.Module):
    def __init__(self):
        super(Converter, self).__init__()

    def forward(self, x, mask):
        if mask is not None:
            mask = mask.unsqueeze(-1)
            fill_value = 0.  # TODO: should use eps instead?
            x = x.masked_fill(mask == 0., fill_value)
            x = x.sum(dim=-2)  # x: [-1, nwords, demb] -> dim=-2 to sum over words
            mask = mask.sum(dim=1).float()
            x = x / torch.sqrt(mask)
            x[x != x] = 0.  # filter NaN
        else:
            x = x.mean(dim=-2)  # x: [-1, nwords, demb] -> dim=-2 to sum over words

        return x


def make_model(N=2, d_model=50, d_ff=512, h=5, dropout=0.1, d_out=512, to_decode=True):
    "Helper: Construct a model from hyperparameters."

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    enc = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)

    dec_attn = MultiHeadAttention(h=1, d_model=d_model)
    dec = Decoder(d_model, d_out, dec_attn)

    converter = Converter()

    model = EncoderCovertDecoder(enc, dec, converter, to_decode)

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
