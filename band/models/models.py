from tqdm import tqdm
from pathlib import Path
from glob import glob
import math, time, os, re
import numpy as np
import pickle
import torch
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable

from band.models.resnet3d import *
from band.utilities.fwriter import FWriter
from band.models.TransformerFM import BertModel, BertLayerNorm, BertConfig


def init_bert_weights(self, module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class S3ConvXFC(nn.Module):
    def __init__(self, in_channel, num_classes, ps=None, d_model=None):
        super(S3ConvXFC, self).__init__()

        self.in_channel, self.num_classes, self.ps, self.d_model = in_channel, num_classes, ps, d_model

        # first in_channel is number of frames
        self.convs = nn.Sequential(
            nn.Conv3d(self.in_channel, 3, 1, stride=1, padding=0),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),

            nn.Conv3d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, 3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, 3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveMaxPool3d((1, 1, 1)),
        )

        self.fcs = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.num_classes)
        )

        # self.convs = nn.DataParallel(self.convs)
        # self.fcs = nn.DataParallel(self.fcs)

    def forward(self, x, return_emb=False):
        bs = x.size(0)
        emb = self.convs(x)
        emb = emb.view(bs, -1)
        out = self.fcs(emb)

        if return_emb:
            return out, emb
        else:
            return out
        # bs = x.size(0)
        # x = self.convs(x)
        # x = x.view(bs, -1)
        # x = self.fcs(x)
        # return x


class S3ConvXFCResnet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(S3ConvXFCResnet, self).__init__()

        self.in_channel, self.num_classes = in_channel, num_classes

        # self._time_conv = nn.Conv3d(self.in_channel, 3, 1, stride=1, padding=0, bias=False)
        # nn.init.kaiming_normal_(self._time_conv.weight, mode='fan_out', nonlinearity='relu')

        # first in_channel is number of frames
        self.time_conv = nn.Sequential(
            nn.Conv3d(self.in_channel, 3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True)
        )

        self.s3resnet = s3resnet18(num_classes=num_classes)

        self.convs = nn.Sequential(
            self.time_conv,
            self.s3resnet
        )

        # self.convs = nn.DataParallel(self.convs)

    def forward(self, x, return_emb=False):
        out, h = self.convs(x)
        if return_emb:
            return out, h
        else:
            return out


class S3ConvXLSTMFC(nn.Module):
    def __init__(self, in_channel, num_classes, d_conv, r_hidden, r_layers, r_bi=False, p=0., p_cls_hidden=0.,
                 batch_first=True):
        super(S3ConvXLSTMFC, self).__init__()

        self.in_channel, self.num_classes = in_channel, num_classes
        self.d_conv = d_conv

        self.r_layers = r_layers
        self.h_layers = r_layers * 2 if r_bi else r_layers
        self.r_hidden = r_hidden
        self.r_in = d_conv
        self.r_bi = r_bi
        self.p = p
        self.batch_first = batch_first
        self.p_hidden = p_cls_hidden
        self.r_out = r_hidden * 2 if r_bi else r_hidden

        self.convs = s3resnet18(num_classes=num_classes, in_channel=in_channel,
                                final_n_channel=d_conv, no_fc=True)

        self._build_model()

        self.classifier = nn.Sequential(
            nn.Dropout(p_cls_hidden),
            nn.Linear(self.r_out, num_classes)
        )

    def _build_model(self):
        # design rnn
        self.rnn = nn.LSTM(
            input_size=self.r_in,
            hidden_size=self.r_hidden,
            num_layers=self.r_layers,
            bidirectional=self.r_bi,
            dropout=self.p,
            batch_first=self.batch_first,
        )

    def init_hidden(self, bs, device):
        hidden = torch.randn(self.h_layers, bs, self.r_hidden)
        cell = torch.randn(self.h_layers, bs, self.r_hidden)

        hidden = hidden.to(device)
        cell = cell.to(device)

        return hidden, cell

    def forward(self, x):
        bs, n, w, h, d = x.size()
        x = x.view(bs * n, 1, w, h, d)
        emb = self.convs(x)
        # emb: [bs*n, self.d_conv]
        x = emb.view(bs, n, self.d_conv)
        # x: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, r_in = x.size()

        # reset the R hidden state. Must be done before you run a new batch. Otherwise the R will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden(batch_size, x.device)

        # now run through RNN model
        x, _ = self.rnn(x, self.hidden)
        # x: (bs, seq, r_hidden * 2 if bidir else r_hidden)
        x = x[:, -1, :]  # take output of the last layer only
        # x: [bs, r_out]
        x = self.classifier(x)
        # x: [bs, n_classes]
        return x


class S3ConvXLSTMFCConcat(nn.Module):
    def __init__(self, in_channel, num_classes, d_conv, r_hidden, r_layers, r_bi=False, p=0., p_cls_hidden=0.,
                 batch_first=True, val_max_frames=28):
        super(S3ConvXLSTMFCConcat, self).__init__()

        self.in_channel, self.num_classes = in_channel, num_classes
        self.d_conv = d_conv

        self.r_layers = r_layers
        self.h_layers = r_layers * 2 if r_bi else r_layers
        self.r_hidden = r_hidden
        self.r_in = d_conv
        self.r_bi = r_bi
        self.p = p
        self.batch_first = batch_first
        self.p_hidden = p_cls_hidden
        self.r_out = r_hidden * 2 if r_bi else r_hidden

        self.convs = s3resnet18(num_classes=num_classes, in_channel=in_channel,
                                final_n_channel=d_conv, no_fc=True)

        self.val_max_frames = val_max_frames

        self._build_model()

        self.classifier = nn.Sequential(
            nn.Dropout(p_cls_hidden),
            nn.Linear(self.r_out * self.val_max_frames, self.r_out * self.val_max_frames // 4),
            nn.Dropout(p_cls_hidden),
            nn.Linear(self.r_out * self.val_max_frames // 4, num_classes)
        )

    def _build_model(self):
        # design rnn
        self.rnn = nn.LSTM(
            input_size=self.r_in,
            hidden_size=self.r_hidden,
            num_layers=self.r_layers,
            bidirectional=self.r_bi,
            dropout=self.p,
            batch_first=self.batch_first,
        )

    def init_hidden(self, bs, device):
        hidden = torch.randn(self.h_layers, bs, self.r_hidden)
        cell = torch.randn(self.h_layers, bs, self.r_hidden)

        hidden = hidden.to(device)
        cell = cell.to(device)

        return hidden, cell

    def forward(self, x):
        bs, n, w, h, d = x.size()
        x = x.view(bs * n, 1, w, h, d)
        emb = self.convs(x)
        # emb: [bs*n, self.d_conv]
        x = emb.view(bs, n, self.d_conv)
        # x: (batch_size, seq_len, embedding_dim)
        batch_size, seq_len, r_in = x.size()

        # reset the R hidden state. Must be done before you run a new batch. Otherwise the R will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden(batch_size, x.device)

        # now run through RNN model
        x, _ = self.rnn(x, self.hidden)
        # x: (bs, seq, r_hidden * 2 if bidir else r_hidden)
        # x = x[:, -1, :]  # concat output at all timesteps
        x = x.contiguous().view(bs, -1)  # concat output at all timesteps
        # x: [bs, r_out]
        x = self.classifier(x)
        # x: [bs, n_classes]
        return x


class S3ConvXTransFC(nn.Module):
    def __init__(self, in_channel, num_classes, trans_config, d_conv):
        super(S3ConvXTransFC, self).__init__()

        self.in_channel, self.num_classes, self.trans_config = in_channel, num_classes, trans_config
        self.d_conv = d_conv

        self.convs = s3resnet18(num_classes=num_classes, in_channel=in_channel,
                                final_n_channel=trans_config.hidden_size, no_fc=True)

        # self.convs = nn.DataParallel(self.convs)

        self.transformer = BertModel(config=self.trans_config)

        # TODO
        self.cls = nn.Parameter(torch.randn(1, 1, trans_config.hidden_size))
        self.classifier = nn.Sequential(
            nn.Dropout(trans_config.hidden_dropout_prob),
            nn.Linear(trans_config.hidden_size, num_classes)
        )

        # self.transformer.apply(init_bert_weights)

    def forward(self, x):
        # print(f"MEAN: {x.mean()}, STD: {x.std()}")
        print(f"MEAN: {x.mean()}, STD: {x.std()}")
        print(f"SHAPE: {x.shape}")

        bs, n, w, h, d = x.size()
        x = x.view(bs * n, 1, w, h, d)
        emb = self.convs(x)
        # emb: [bs*n, self.d_conv]
        emb = emb.view(bs, n, self.d_conv)
        emb = torch.cat((self.cls.expand(bs, -1, -1), emb), dim=1)

        enc_layer, pooled_emb = self.transformer(emb, output_all_encoded_layers=False)

        logits = self.classifier(pooled_emb)

        return logits


class S3TransCNN(nn.Module):
    def __init__(self, in_channel, num_classes, trans_config, d_conv):
        super(S3TransCNN, self).__init__()

        self.in_channel, self.num_classes, self.trans_config = in_channel, num_classes, trans_config
        self.d_conv = d_conv

        self.trans_emb = nn.Linear(1, self.trans_config.hidden_size)
        self.transformer = BertModel(config=self.trans_config)
        self.trans_out = nn.Linear(self.trans_config.hidden_size, 1)
        self.trans_squeeze = nn.Linear(self.in_channel, self.in_channel)
        self.s3resnet = s3resnet18(num_classes=num_classes, in_channel=3,
                                   final_n_channel=trans_config.hidden_size, no_fc=True)

        self.convs = nn.Sequential(
            self.s3resnet
        )

        self.convs = nn.DataParallel(self.convs)

        self.classifier = nn.Sequential(
            nn.Dropout(trans_config.hidden_dropout_prob),
            nn.Linear(d_conv, num_classes)
        )

    def forward(self, x):
        bs, n, w, h, d = x.size()
        x = x.permute(0, 2, 3, 4, 1)
        # bs, w, h, d, n
        x = x.unsqueeze(-1)
        # bs, w, h, d, n, 1
        emb = self.trans_emb(x)
        # bs, w, h, d, n, 128
        emb, _ = self.transformer(emb, output_all_encoded_layers=False)
        print(emb.shape)
        # bs, w, h, d, n, 128
        emb = self.trans_out(emb)
        # bs, w, h, d, n, 1
        emb = self.trans_squeeze(emb.squeeze(-1))
        # bs, w, h, d, in_channel
        emb = emb.permute(0, 4, 1, 2, 3)
        # bs, in_channel, w, h, d
        emb = self.convs(x)
        # bs, d_conv

        logits = self.classifier(emb)

        return logits


class Trainer():
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

    def __init__(self, model, trainloader, args, criterion, optimizer, name=None, out_dir="out", valloader=None,
                 device=torch.device('cpu'), is_restart=False, resume_model=None, is_debug=True):
        self.model = model
        self.trainloader = trainloader
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.valloader = valloader
        self.device = device
        self.is_debug = is_debug

        # name mostly for saving models
        self.stime = math.floor(time.time())

        if name is None:
            self.name = "trainer.{}".format(self.stime)
        else:
            self.name = name

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        # mkdir out_dir
        self.out_dir = self.out_dir / self.name
        self.out_dir.mkdir(exist_ok=True)

        self.log_file = self.out_dir / 'logger_{}.out'.format(self.stime)

        self.fwriter = FWriter(is_debug=self.is_debug, fn=str(self.log_file))

        self.is_restart = is_restart
        self.resume_model = resume_model

    # TODO: need to fix because save_model got changed
    def restore_model(self):
        # list model files and get the largest one
        if not self.resume_model:  # find the latest model to load
            latest_pt = self.find_latest_pt()

            if latest_pt is None:
                return 0, 0  # no model to restore from

            self.resume_model = latest_pt

        # resume model using self.resume_model
        resume_path = self.out_dir / self.resume_model

        assert resume_path.exists(), "Resume model %s doesn't exist." % str(resume_path)
        print("Reloading model: %s" % str(resume_path))

        import numpy as np
        d = np.load(str(resume_path))
        state_dict = d[()]['state_dict']
        self.model.load_state_dict(state_dict)
        print("Done reloading model: %s" % str(resume_path))

        epoch, batch = self.get_epoch_and_batch_num(str(self.resume_model))
        return epoch, batch

    def get_epoch_and_batch_num(self, file_name):
        match_str = 'model.epoch-([0-9]*?).batch-([0-9]*?).pt.npy'
        r = re.search(match_str, file_name).groups()
        epoch, batch = (int(r[0]), int(r[1]))

        return (epoch, batch)

    def find_latest_pt(self):
        # sort by time way
        files = glob(str(self.out_dir / 'model.epoch*.pt.npy'))

        if len(files) == 0:
            return None

        files.sort(key=os.path.getmtime, reverse=True)
        latest_file = Path(files[0]).name  # .name to get the file name only

        # get epoch and batch numbers
        return Path(latest_file)

        # RegEx way
        # match_str = 'model.epoch-([0-9]*?).batch-([0-9]*?).pt.npy'
        # tuples = []
        # for f in files:
        #     r = re.search(match_str, f).groups()
        #     assert len(r) == 2, "Regex: {} must be of length 2, as (epoch, batch)".format(r)
        #     epoch, batch = int(r[0]), int(r[1])
        #     tuples.append((epoch, batch))

    def fit_one_batch(self, n_epoch, data):
        for epoch in range(n_epoch):
            self.model.train()

            inputs, labels = data['data'], data['label']

            # transfer to device
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # forward + backward + optimize
            out = self.model(inputs)
            loss = self.criterion(out, labels)
            loss.backward()
            self.optimizer.step()

            self.fwriter.write('\t[%d, %5d] %d datum, loss: %.3f' % (n_epoch, epoch, inputs.size(0), loss.item()))

    def train(self, n_epoch):
        F = self.fwriter

        args = self.args
        trainloader, valloader = self.trainloader, self.valloader
        model = self.model
        device = self.device
        criterion, optimizer = self.criterion, self.optimizer

        best_val_acc = 0.0

        curr_epoch, curr_batch = 0, 0

        # restore model
        if self.is_restart:
            curr_epoch, curr_batch = self.restore_model()

        n_epoch = curr_epoch + n_epoch
        pb_epoch = tqdm(range(curr_epoch, n_epoch))
        for epoch in pb_epoch:
            pb_epoch.set_description("Epoch: %d/%d" % (epoch, n_epoch))
            pb_epoch.refresh()

            F.write("Epoch: %d/%d" % (epoch + 1, n_epoch + 1))
            running_loss = 0.0
            bstime = time.time()
            blstime = bstime
            datatime = bstime
            processtime = bstime

            for i, data in enumerate(trainloader):
                # set to train mode
                model.train()

                inputs, labels = data['data'], data['label']

                # transfer to device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # forward + backward + optimize
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                preds = torch.max(out, dim=1)[1]
                accuracy = self.get_accuracy(preds, labels)
                F.write('\t[%d, %5d] %d datum, loss: %.3f, accuracy: %.3f, %.3f secs' % (
                    epoch + 1, i + 1, inputs.size(0), loss.item(), accuracy, time.time() - blstime))

                running_loss += loss.item()

                if args['print_every']:
                    interval = args['print_every']
                    if i % interval == interval - 1:
                        F.write('\t[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / interval))
                        running_loss = 0.0

                if 'val_every' in args and args['val_every']:
                    assert valloader is not None, "Need to input valloader to test on val dataset"

                    interval = args['val_every']
                    if i % interval == interval - 1:
                        F.write("\n\tValidating")
                        model.eval()

                        counter = 0
                        i_counter = 0
                        val_running_loss = 0.0
                        correct_preds = 0

                        for idx, val_data in enumerate(valloader):
                            inputs, labels = val_data['data'], val_data['label']
                            counter += inputs.size(0)  # add number of data to counter
                            # transfer to device
                            inputs, labels = inputs.to(device), labels.to(device)

                            optimizer.zero_grad()  # TODO: need?
                            out = model(inputs)
                            loss = criterion(out, labels)
                            val_running_loss += loss.item()
                            preds = torch.max(out, dim=1)[1]

                            i_counter += 1

                            correct_preds += self.get_correct_preds(preds, labels)

                        val_acc = correct_preds / counter
                        F.write('\t[%d, %5d] VAL loss: %.3f, acc: %.3f' % (
                            epoch + 1, i + 1, val_running_loss / i_counter, val_acc))

                        if val_acc > best_val_acc:
                            # save model
                            fn = "model.best_val.epoch-{}.batch-{}.pt".format(epoch, i)
                            self.save_model(fn)
                            best_val_acc = val_acc

                if args['save_every']:
                    interval = args['save_every']
                    if i % interval == interval - 1:
                        fn = "model.epoch-{}.batch-{}.pt".format(epoch, i)
                        self.save_model(fn)

                blstime = time.time()

    def save_model(self, fn):
        meta_data = {
            'args': self.args,
            'model_str': str(self.model),
            'trainer': {
                # 'epoch_count': trainer.epoch_count,
                # 'train_losses': trainer.train_losses,
                # 'val_losses': trainer.val_losses
            }

        }

        opath = self.out_dir / fn
        self.fwriter.write("\nSaving model+meta to {}\n".format(str(opath)))

        # save model
        torch.save(self.model.state_dict(), "{}.pt".format(opath))

        # save meta
        meta_path = "{}.meta".format(opath)

        with open(meta_path, 'wb') as f:
            pickle.dump(meta_data, f)

    def get_accuracy(self, preds, labels):
        # need to transfer it back to cpu
        correct_preds = self.get_correct_preds(preds, labels)

        return correct_preds / labels.size(0)

    def get_correct_preds(self, preds, labels):
        # need to transfer it back to cpu
        correct_preds = np.count_nonzero((preds == labels).cpu())
        return correct_preds


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size, torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss
