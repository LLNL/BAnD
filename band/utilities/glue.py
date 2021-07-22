import uuid
import glob
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

from band.utilities.transforms import *
from band.utilities.datasets import *
from band.models.models import *


def my_collate(batch):
    # batch is a list of dict: keys: data, length, and label, mask
    out = {}

    lengths = torch.tensor([b['length'] for b in batch])
    sorted_lengths, argsort_lengths = lengths.sort(descending=True)
    out['length'] = sorted_lengths

    # permute data, label and mask
    data = [batch[i]['data'] for i in argsort_lengths]
    label = [batch[i]['label'] for i in argsort_lengths]
    mask = [batch[i]['mask'] for i in argsort_lengths]
    out['data'] = torch.cat(data, dim=0)
    out['label'] = torch.tensor(label)
    out['mask'] = torch.stack(mask)

    return out


def my_collate_diff(batch):
    # batch is a list of dict: keys: data, length, and label, mask
    out = {}

    lengths = torch.tensor([b['length'] for b in batch])
    sorted_lengths, argsort_lengths = lengths.sort(descending=True)
    out['length'] = sorted_lengths

    # permute data, label and mask
    data = [_get_diff(batch[i]['data']) for i in argsort_lengths]
    label = [batch[i]['label'] for i in argsort_lengths]
    mask = [batch[i]['mask'] for i in argsort_lengths]
    out['data'] = torch.cat(data, dim=0)
    out['label'] = torch.tensor(label)
    out['mask'] = torch.stack(mask)

    return out


def _get_diff(data):
    for i in range(0, len(data)):
        data[i] = data[i] - data[max(0, i - 1)]

    return data


def _grad_hook(grad):
    pass


class HCPGlue(Glue):
    """
        Specify how to load data and everything for HCP project
    """

    def __init__(self, config, device):
        self.set_config(config)
        self.config.device = device

    def init(self, distributed=False, local_rank=-1, world_rank=-1, write_args=True):
        self.config.distributed, self.config.local_rank, self.config.world_rank = distributed, local_rank, world_rank

        if getattr(self.config, 'param_str', None):
            run_name = f"{self.config.param_str}|world_rank={world_rank}"
        else:
            run_name = f"{self.exp}|worldrank={world_rank}"

        out_path = Path(self.config.out_path)
        out_path.mkdir(exist_ok=True)

        run_dir = out_path / run_name
        run_dir.mkdir(exist_ok=True)

        log_file = run_dir / 'logger.out'
        args_file = run_dir / 'args.txt'

        if write_args:
            print(f"Writing args/config to {args_file}")
            with open(args_file, 'w') as f:
                f.write(str(self.config))
                f.write("\n")
                f.write(
                    f"Distributed: {self.distributed}, world rank: {self.world_rank}, local rank: {self.local_rank}")

        self.config.run_name = run_name
        self.config.run_dir = run_dir
        self.config.log_file = log_file
        self.config.args_file = args_file

        print(f"RUN DIR: {str(run_dir)}")

        if getattr(self.config, 'seed', None):
            self._set_seed(cudnn=True)
        else:
            # pass
            self._set_cudnn(cudnn=True)

    def _set_seed(self, cudnn=True):
        seed = self.config.seed
        print(f'Setting seed for torch and numpy: {seed:d}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        if cudnn:
            print('Setting cuDNN to be deterministic')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _set_cudnn(self, cudnn=True):
        if cudnn:
            print('Setting cuDNN')
            torch.backends.cudnn.benchmark = True

    def get_files(self):
        """

        :param config: project's config object
        :return: (train_files, val_files, test_files)
        """
        data_path = Path(self.data_path)
        files = glob(str(data_path / '*.npy'))
        # Sort by creation time instead, for better distribution of classes in train/val/test
        files.sort(key=os.path.getmtime)
        label_file = data_path / self.label_file
        assert data_path.exists() and label_file.exists()

        test_files, val_files, train_files = get_test_val_train_split(files, valid_size=self.val_size,
                                                                      test_size=self.test_size,
                                                                      seed=self.config.split_seed)

        self.config.data_path = data_path
        self.config.label_file = label_file
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        return train_files, val_files, test_files

    def get_transforms(self):
        """
        Transformations for data
        :param config:
        :return: (train_transforms, val_transforms)
        """
        sizes = (75, 93, 81)
        if "_test" in self.config.model:
            # test model:
            train_composed_transforms = transforms.Compose(
                [CenterCrop(sizes),
                 TimeCropAndPad(self.val_max_frames, random_crop=True, self_pad=True, n=5),
                 ToTensor(),
                 Permute(data_perm=(0, 4, 1, 2, 3))
                 ])

            val_composed_transforms = transforms.Compose(
                [CenterCrop(sizes),
                 TimeCropAndPad(self.val_max_frames, random_crop=True, self_pad=True, n=5),
                 ToTensor(),
                 Permute(data_perm=(0, 4, 1, 2, 3))
                 ])

        else:  # train
            train_composed_transforms = transforms.Compose(
                [CenterCrop(sizes),
                 TimeCropAndPad(self.val_max_frames, random_crop=True, self_pad=True,
                                random_head=self.config.train_random_head, n=1, pad_mode=self.config.pad_mode),
                 SkipFrame(n=self.config.skip_frame_n, to_skip=self.config.skip_frame_to_skip),
                 ToTensor(),
                 Permute(data_perm=(3, 0, 1, 2))
                 ])

            val_composed_transforms = transforms.Compose(
                [CenterCrop(sizes),
                 TimeCropAndPad(self.val_max_frames, random_crop=self.config.val_random_crop, self_pad=True,
                                random_head=False, n=1,
                                pad_mode=self.config.pad_mode),
                 SkipFrame(n=self.config.skip_frame_n, to_skip=self.config.skip_frame_to_skip),
                 ToTensor(),
                 Permute(data_perm=(3, 0, 1, 2))
                 ])

        return train_composed_transforms, val_composed_transforms

    def get_ds(self):
        """
        Datasets
        :param config:
        :return: train_ds, val_ds, test_ds
        """
        train_transform, val_transform = self.get_transforms()
        train_files, val_files, test_files = self.get_files()

        # Make dataset and dataloader
        train_ds = HCPCacheDataset(train_files, self.data_path, self.label_file,
                                   transform=train_transform, block_size=self.train_block,
                                   workers=self.preload_workers)
        valid_ds = HCPCacheDataset(val_files, self.data_path, self.label_file,
                                   transform=val_transform, block_size=self.val_block, workers=self.preload_workers)
        test_ds = HCPCacheDataset(test_files, self.data_path, self.label_file,
                                  transform=val_transform, block_size=self.val_block, workers=self.preload_workers)

        return train_ds, valid_ds, test_ds

    def get_dls(self, shuffle=True, pin_memory=False, **kwargs):
        train_ds, valid_ds, test_ds = self.get_ds()

        if self.distributed:
            print("WARNING: Using distributed sampler")
            train_sampler = DistributedSampler(train_ds)
            valid_sampler = DistributedSampler(valid_ds)
            test_sampler = DistributedSampler(test_ds)
            train_shuffle = None  # shuffle and sampler are mutex
        else:
            train_sampler, valid_sampler, test_sampler = None, None, None
            train_shuffle = shuffle

        return (
            BlockDataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers,
                            sampler=train_sampler, shuffle=train_shuffle, pin_memory=pin_memory, **kwargs),
            BlockDataLoader(valid_ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=valid_sampler,
                            pin_memory=pin_memory,
                            **kwargs),
            BlockDataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, sampler=test_sampler,
                            pin_memory=pin_memory,
                            **kwargs)
        )

    def get_device(self):
        if self.device:
            return self.device
        else:
            raise Exception("device should have been set in init: did you call self.init()?")

    def get_model(self):
        device = self.get_device()

        if self.config.model == "resnet":
            in_channel = self.val_max_frames
            if self.config.skip_frame_to_skip:
                in_channel = self.val_max_frames // int(self.config.skip_frame_n)
            print(in_channel)

            model = S3ConvXFCResnet(in_channel=in_channel, num_classes=self.n_classes)
            model = model.to(device)
            self._init_model(model)

        elif self.config.model == "cnn":
            in_channel = self.val_max_frames
            if self.config.skip_frame_to_skip:
                in_channel = self.val_max_frames // int(self.config.skip_frame_n)
            print(in_channel)

            model = S3ConvXFC(in_channel=in_channel, num_classes=self.n_classes)
            model = model.to(device)
            self._init_model(model)

        elif self.config.model == "pooled_transformer_full":
            in_channel, num_classes = 1, self.config.n_classes

            bert_config = BertConfig(vocab_size_or_config_json_file=-1,
                                     hidden_size=512,
                                     num_hidden_layers=1,
                                     num_attention_heads=8,
                                     intermediate_size=512,
                                     hidden_act="relu",
                                     hidden_dropout_prob=0.2,
                                     attention_probs_dropout_prob=0.2,
                                     max_position_embeddings=128,
                                     type_vocab_size=None,
                                     initializer_range=0.02,
                                     layer_norm_eps=1e-12
                                     )

            model = S3ConvXTransFC(in_channel, num_classes, bert_config, 512)

            if getattr(self.config, 'saved_model_path'):
                self._resume_model(model, self.config.saved_model_path)
            else:
                # init resnet, bert is already init inside
                self._init_model(model.convs)
                self._init_model(model.classifier)

            model = model.to(device)

        elif self.config.model == "pooled_transformer_full_finetune":
            in_channel, num_classes = 1, self.config.n_classes

            bert_config = BertConfig(vocab_size_or_config_json_file=-1,
                                     hidden_size=512,
                                     num_hidden_layers=2,
                                     num_attention_heads=8,
                                     intermediate_size=512,
                                     hidden_act="relu",
                                     hidden_dropout_prob=0.2,
                                     attention_probs_dropout_prob=0.2,
                                     max_position_embeddings=128,
                                     type_vocab_size=None,
                                     initializer_range=0.02,
                                     layer_norm_eps=1e-12
                                     )

            model = S3ConvXTransFC(in_channel, num_classes, bert_config, 512)

            if getattr(self.config, 'saved_model_path'):
                if getattr(self.config, 'is_test', None):
                    self._resume_model(model, self.config.saved_model_path)
                else:
                    self._resume_model_without_transformer(model, self.config.saved_model_path)

            else:
                # init resnet, bert is already init inside
                self._init_model(model.convs)
                self._init_model(model.classifier)

            model = model.to(device)

        elif self.config.model == "resnet_lstm":
            in_channel, num_classes = 1, self.config.n_classes

            d_conv = r_in = 512
            r_hidden = self.config.r_hidden
            r_layers = self.config.r_layers
            r_bi = bool(self.config.r_bi)
            r_p = self.config.r_p
            r_p_cls_hidden = self.config.r_p_cls_hidden

            # args: self, num_classes, r_in, r_hidden, r_layers, r_bi = False, p = 0., p_cls_hidden = 0.
            model = S3ConvXLSTMFC(in_channel, num_classes, r_in, r_hidden, r_layers, r_bi=r_bi, p=r_p,
                                    p_cls_hidden=r_p_cls_hidden)

            if getattr(self.config, 'saved_model_path'):
                self._resume_model(model, self.config.saved_model_path)
            else:
                self._init_model(model.rnn)
                self._init_model(model.convs)
                self._init_model(model.classifier)

            model = model.to(device)

        elif self.config.model == "resnet_lstm_concat":
            in_channel, num_classes = 1, self.config.n_classes

            d_conv = r_in = 512
            r_hidden = self.config.r_hidden
            r_layers = self.config.r_layers
            r_bi = bool(self.config.r_bi)
            r_p = self.config.r_p
            r_p_cls_hidden = self.config.r_p_cls_hidden

            val_max_frames = self.config.val_max_frames

            # args: self, num_classes, r_in, r_hidden, r_layers, r_bi = False, p = 0., p_cls_hidden = 0.
            model = S3ConvXLSTMFCConcat(in_channel, num_classes, r_in, r_hidden, r_layers, r_bi=r_bi, p=r_p,
                                          p_cls_hidden=r_p_cls_hidden, val_max_frames=val_max_frames)

            if getattr(self.config, 'saved_model_path'):
                self._resume_model(model, self.config.saved_model_path)
            else:
                self._init_model(model.rnn)
                self._init_model(model.convs)
                self._init_model(model.classifier)

            model = model.to(device)

        elif self.config.model == "trans_resnet":
            in_channel = self.val_max_frames
            if self.config.skip_frame_to_skip:
                in_channel = self.val_max_frames // int(self.config.skip_frame_n)
            print(in_channel)
            bert_config = BertConfig(vocab_size_or_config_json_file=-1,
                                     hidden_size=128,
                                     num_hidden_layers=2,
                                     num_attention_heads=4,
                                     intermediate_size=256,
                                     hidden_act="relu",
                                     hidden_dropout_prob=0.2,
                                     attention_probs_dropout_prob=0.2,
                                     max_position_embeddings=128,
                                     type_vocab_size=None,
                                     initializer_range=0.02,
                                     layer_norm_eps=1e-12
                                     )

            model = S3TransCNN(in_channel=in_channel, num_classes=self.n_classes, trans_config=bert_config, d_conv=512)
            model = model.to(device)

            # init resnet, bert is already init inside
            self._init_model(model.convs)
            self._init_model(model.classifier)

        else:
            raise ValueError(f"{self.config.model} not implemented.")

        return model

    def get_optimizer(self, model):
        # get optimizer
        if self.config.opt == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                  weight_decay=self.weight_decay)
        elif self.config.opt == "adam":
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"{self.config.opt} not implemented.")

        return optimizer

    def _resume_model(self, model, model_path, from_module=True):
        # from_module flag is to remove 'module.' from the state_dict's key
        p = model_path
        resume_device = torch.device('cpu')  # map state_dict to cpu at first
        print(f"Resuming model from {p}")
        state_dict = torch.load(str(p), map_location=resume_device)

        if from_module:
            print("Removing 'module.' from state_dict")
            new_state_dict = {}
            for key in state_dict:
                new_key = key
                if 'module.' in key:
                    new_key = key[7:]
                new_state_dict[new_key] = state_dict[key]
            model.load_state_dict(new_state_dict)
            del new_state_dict  # dereference seems crucial
        else:
            model.load_state_dict(state_dict)

        # optimizer.load_state_dict(checkpoint['optimizer'])  # pytorch 0.3.1 has a bug on this, it's fix in master
        del state_dict  # dereference seems crucial
        torch.cuda.empty_cache()

    def _resume_model_without_transformer(self, model, model_path, from_module=True):
        # from_module flag is to remove 'module.' from the state_dict's key
        p = model_path
        resume_device = torch.device('cpu')  # map state_dict to cpu at first
        print(f"Resuming model from {p}")
        state_dict = torch.load(str(p), map_location=resume_device)
        model_dict = model.state_dict()

        if from_module:
            print("Removing 'module.' from state_dict")
            new_state_dict = {}
            for key in state_dict:

                new_key = key
                if 'module.' in key:
                    new_key = key[7:]

                if "transformer" in new_key or "cls" in new_key or "classifier" in new_key:
                    # import ipdb; ipdb.set_trace()
                    new_state_dict[new_key] = model_dict[new_key]
                else:
                    new_state_dict[new_key] = state_dict[key]

            # another transformer layer may have been added (weights are from scratch anyway)
            for key in model_dict:
                if key not in new_state_dict:
                    new_state_dict[key] = model_dict[key]
            model.load_state_dict(new_state_dict)
            del new_state_dict  # dereference seems crucial
        else:
            raise ValueError()
            # model.load_state_dict(state_dict)

        # optimizer.load_state_dict(checkpoint['optimizer'])  # pytorch 0.3.1 has a bug on this, it's fix in master
        del state_dict  # dereference seems crucial
        torch.cuda.empty_cache()

    def _init_model(self, model):
        # Initialize parameters of the model
        init_method = getattr(self.config, 'initialization', None)
        init_funcs = {
            "xavier_norm": nn.init.xavier_normal_,
            "xavier_unif": nn.init.xavier_uniform_
        }

        if init_method in init_funcs:
            print(f"INIT model params with {init_method}.")
            for p in model.parameters():
                if p.dim() > 1:
                    init_funcs[init_method](p)
        else:
            raise Exception(f"Init method {init_method} not implemented")

    def _register_hooks(self, model):
        # backward hooks for gradients clipping
        for p in model.parameters():
            # p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
            p.register_hook(_grad_hook)