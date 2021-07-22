from band.utilities.tools import *
import ipdb

import torch


class AvgStats:
    def __init__(self, metrics, in_train, in_test=False):
        self.metrics, self.in_train, self.in_test = listify(metrics), in_train, in_test
        self.reset()

    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self):
        return [o / self.count for o in self.all_stats]

    @property
    def avg_stats_with_name(self):
        return self.metrics_names, self.avg_stats

    @property
    def metrics_names(self):
        prefix = self.mode
        return [f'{prefix}_loss'] + [f'{prefix}_{m.__name__}' for m in self.metrics]

    @property
    def mode(self):
        return 'train' if self.in_train else 'test' if self.in_test else 'valid'

    def __repr__(self):
        if not self.count: return ""
        return f"{self.mode}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i, m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats, self.test_stats = AvgStats(metrics, True), AvgStats(metrics,
                                                                                                False), AvgStats(
            metrics, False, True)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.test_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.test_stats if self.in_test else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stat_list = [self.train_stats, self.valid_stats]
        if self.in_test:
            stat_list = [self.test_stats]

        for stats in stat_list:
            self.logger.write(f"{stats.mode}: {stats}\n")
            # check for tensorboard_logger
            if self.tb_logger:
                for k, v in zip(*stats.avg_stats_with_name):
                    step = self.num_epochs
                    self.tb_logger.add_scalar(f"epoch_{k}", v, global_step=step)

        self.logger.write("\n")


class WeightGainCallback(Callback):
    _order = -1

    def __init__(self, start, end, steps, device):
        self.weights = torch.linspace(start=start, end=end, steps=steps).to(device)
        self.sum = self.weights.sum()

    def after_loss(self):
        bs, nframes = self.run.xb.shape[:2]
        self.run.loss = self.run.loss.view(bs, nframes)
        # self.run.loss = torch.matmul(self.run.loss, self.weights).div_(self.sum).mean()
        self.run.loss = torch.matmul(self.run.loss, self.weights).mean()


class AvgStatsAndSaveFeaturesCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats, self.test_stats = AvgStats(metrics, True), AvgStats(metrics,
                                                                                                False), AvgStats(
            metrics, False, True)

        self.best_metric = 0.
        # SN: keep track of current best valid_stat
        self.metric_key = "valid_accuracy" if len(metrics) > 0 else "valid_loss"
        if self.metric_key == "valid_loss":
            self.best_metric = float("inf")

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.test_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.test_stats if self.in_test else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stat_list = [self.train_stats, self.valid_stats]
        if self.in_test:
            stat_list = [self.test_stats]

        for stats in stat_list:
            self.logger.write(f"{stats.mode}: {stats}\n")
            # check for tensorboard_logger
            if self.tb_logger:
                for k, v in zip(*stats.avg_stats_with_name):
                    step = self.num_epochs
                    self.tb_logger.add_scalar(f"epoch_{k}", v, global_step=step)

        self.logger.write("\n")

        # update best_valid if it's better
        if not self.in_test:
            for k, v in zip(*self.valid_stats.avg_stats_with_name):
                if k == self.metric_key:
                    if self.metric_key == "valid_loss":
                        metric_ = float(v)
                        if metric_ < self.best_metric:
                            self.logger.write(f"{metric_} is better than {self.best_metric}")
                            self.best_metric = metric_

                            # save model instead
                            self.save_model(score=f"{k}_{metric_:.4f}")

                    if self.metric_key == "valid_accuracy":
                        metric_ = float(v.item())
                        if metric_ > self.best_metric:
                            self.logger.write(f"{metric_} is better than {self.best_metric}")
                            self.best_metric = metric_

                            # save model instead
                            self.save_model(score=f"{k}_{metric_:.4f}")

            self.logger.write(f"BEST {self.metric_key}: {self.best_metric}")
            best_ckpt = getattr(self.learn.glue.config, "best_ckpt", None)
            if best_ckpt:
                self.logger.write(f"BEST val ckpt: {self.learn.glue.config.best_ckpt}\n")

    def save_model(self, score):
        fn = f"model_best.pt"
        run_dir = self.run.learn.glue.run_dir
        p = str(run_dir / fn)
        self.logger.write(f"\nSaving model to {p}\n")
        self.learn.glue.config.best_ckpt = p
        # save model
        torch.save(self.model.state_dict(), p)


class PrepareDataCallback(Callback):
    # should have access to learn: self.learn through self.run.learn
    def __init__(self, normalizer=None):
        self.normalizer = normalizer

    def begin_batch(self):
        device = self.learn.glue.device
        if isinstance(device, list):
            device_in = device[0]  # if a list, then take the first device for input
            device_out = device[-1]  # if a list, then take the last device for output
        else:
            device_in = device
            device_out = device

        self.run.xb, self.run.yb = self.sample['data'].to(device_in), self.sample['label'].to(device_out)
        # bs, nframes, = self.run.xb.shape[:2]

        # for label for each frame (not only the whole clip)
        # self.run.yb = self.run.yb.view(-1, 1, 1).expand(bs, nframes, 1).contiguous().view(-1)

        if self.normalizer:
            self.run.xb = self.normalizer(self.run.xb)

        # LSTM
        # lengths = self.sample['length']
        # sorted_lengths, argsort_lengths = lengths.sort(descending=True)
        # import ipdb; ipdb.set_trace()
        # # permute xb and labels
        # self.run.xb = self.run.xb[argsort_lengths]
        # self.run.yb = self.run.yb[argsort_lengths]
        # padded_x = padded_x[argsort_lengths].to(device)

        self.run.inputs = (self.run.xb,)  # need a comma here for a real tuple (list)


class StatLogCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats, self.test_stats = AvgStats(metrics, True), AvgStats(metrics,
                                                                                                False), AvgStats(
            metrics, False, True)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.test_stats.reset()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.test_stats if self.in_test else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)

        self.logger.write(f"[{self.run.n_iter}|{self.run.n_epochs:.3f}] {str(stats)}")

        # check for tensorboard_logger
        if self.tb_logger:
            for k, v in zip(*stats.avg_stats_with_name):
                step = self.n_iter if self.in_train else self.val_iter
                self.tb_logger.add_scalar(k, v, global_step=step)

        stats.reset()


class ValidateCallback(Callback):
    _order = 3

    def __init__(self, validate):
        self.validate = validate

    def begin_validate(self):
        if self.validate:
            self.run.logger.write("Validating")

        return not self.validate


class TestCallback(Callback):
    _order = 2

    def __init__(self, to_test=False, skip_training=False):
        self.to_test, self.skip_training = to_test, skip_training
        if self.to_test and self.skip_training:
            print("WARNING: TESTING ONLY")
            print("WARNING: skip_training in TestCallback is set -> NOT training/validating")

    def begin_fit(self):
        return self.to_test and self.skip_training

    def begin_test(self):
        if self.to_test:
            self.model.eval()

            self.run.logger.write("\nTesting")

            config = self.learn.glue.config
            data = self.data

            self.run.in_train = False
            self.run.in_test = True

            test_dl = getattr(self.data, 'test_dl', None)
            if not test_dl:
                print("No test_dl to test. Stop.")
                return False  # don't test if no test_dl

            return True


class ABCallback(Callback):
    def after_backward(self):
        ipdb.set_trace()
        print(self)


class LoggerCallback(Callback):
    def __init__(self, tb=False):
        self.tb = tb

    def begin_fit(self):
        config = self.learn.glue.config

        logger = make_writer(config)
        self.run.logger = logger

        log_init_info(logger, config, ('data_path', 'log_file'))

        self.run.tb_logger = None
        # if self.tb -> use tensorboard
        if self.tb:
            assert config.run_dir, "must provide config.run_dir to set up tensorboard logger"
            from torch.utils.tensorboard import SummaryWriter
            print('Setting up Tensorboard logger')
            self.run.tb_logger = SummaryWriter(log_dir=str(config.run_dir), flush_secs=5)

    def after_fit(self):
        self.logger.write("END\n")
        self.logger.write(f"Run dir: {self.learn.glue.config.run_dir}\n")
        best_ckpt = getattr(self.learn.glue.config, 'best_ckpt', None)
        self.logger.write(f"Best val ckpt: {best_ckpt}\n")

    def after_epoch(self):
        save_every_epoch = getattr(self.learn.glue.config, 'save_every_epoch', -1)
        if self.run.num_epochs % save_every_epoch == save_every_epoch - 1:
            if self.learn.glue.config.distributed:
                # only save world_rank 0 model
                if self.learn.glue.config.world_rank == 0:
                    print(f"World rank: {self.learn.glue.config.world_rank} -> save model")
                    self.save_model()
            else:
                self.save_model()

    def after_batch(self):
        save_every = getattr(self.learn.glue.config, 'save_every', -1)
        if self.run.n_iter % save_every == save_every - 1:
            self.save_model()

    def save_model(self):
        fn = f"model.iter={self.run.n_iter}_epoch={self.run.num_epochs}.pt"
        run_dir = self.run.learn.glue.run_dir
        p = str(run_dir / fn)
        self.logger.write(f"\nSaving model to {p}\n")
        # save model
        torch.save(self.model.state_dict(), p)


class Recorder(Callback):
    def __init__(self, save_recorder=True):
        self.save_recorder = save_recorder

    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []
        self.val_losses = []

    def after_batch(self):
        if not self.in_train:
            self.val_losses.append(self.loss.detach().cpu())
        else:
            for pg, lr in zip(self.opt.param_groups, self.lrs): lr.append(pg['lr'])
            self.losses.append(self.loss.detach().cpu())

    # def plot_lr(self, pgid=-1):
    #     plt.plot(self.lrs[pgid])
    #
    # def plot_loss(self, skip_last=0):
    #     plt.plot(self.losses[:len(self.losses) - skip_last])

    def save(self):
        fn = f"recorder.iter-{self.run.n_iter}.epoch-{self.run.num_epochs}.pickle"
        p = str(self.run.learn.glue.run_dir / fn)
        self.logger.write(f'Saving recorder to {p}')
        with open(p, 'wb') as f:
            pickle.dump({'lrs': self.lrs, 'loss': self.losses, 'val_loss': self.val_losses}, f)

    def after_epoch(self):
        if self.save_recorder:
            self.save()


class ParamScheduler(Callback):
    _order = 1

    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs

    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list, tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs)
        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
            pg[self.pname] = f(self.n_epochs / self.epochs)
            # print(f"{self.pname}: {pg[self.pname]}")

    def begin_batch(self):
        if self.in_train: self.set_param()


class ToTrainCallback(Callback):
    _order = 999

    def __init__(self, to_train=True, to_validate_now=False):
        self.to_train = to_train
        self.to_validate_now = to_validate_now

    def begin_fit(self):
        if not self.to_train:
            print("NOT TRAINING")
            return True

    def begin_epoch(self):
        if self.to_validate_now:
            print("VALIDATING NOW INSTEAD OF TRAINING")
            return True


def pg_dicts(pgs): return [{'params': o} for o in pgs]


def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]


class CallbackScheduleCallback(Callback):
    _order = 0
    _cb_names = ['begin_fit', 'begin_epoch', 'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
                 'after_batch', 'begin_validate', 'after_epoch', 'after_fit']

    def begin_fit(self):
        print('Callback schedule:')
        names = [[] for _ in range(len(self._cb_names))]
        sorted_cbs = sorted(self.run.cbs, key=lambda x: x._order)
        for idx, cb_name in enumerate(self._cb_names):
            for cb in sorted_cbs:
                f = getattr(cb, cb_name, None)
                if f:
                    names[idx].append(cb.name)

        for i in range(len(names)):
            print(f"\t{self._cb_names[i]}:")
            print(f"\t\t{names[i]}")
        print()


class ResumeCallback(Callback):
    _order = 99

    def __init__(self, cont):
        self.cont = cont

    def begin_fit(self):
        config = self.learn.glue.config

        saved_model_path = getattr(config, 'saved_model_path', None)
        print(f"Saved model path: {saved_model_path}")
        if saved_model_path:  # restore state_dict from this model
            # check for saved_epoch and saved_iter
            saved_epoch = getattr(config, 'saved_epoch', None)
            saved_iter = getattr(config, 'saved_iter', None)

            if not (saved_epoch or saved_iter):
                raise Exception(f"Have to provide both saved_epoch and saved_iter to resume model")

            print("WARNING: Reloading model in ResumeCallback has been disabled due to OOM problems")
            print("Only restoring num_epochs and n_iter now.")
            # p = saved_model_path
            # print(f"Restoring model from: {p}")
            # self.learn.model.load_state_dict(torch.load(p))

            print(
                f"Setting n_iter, n_epochs, num_epochs and start_epoch to: {saved_iter}, {saved_epoch}, {saved_epoch}, {saved_epoch}")
            # set saved_epoch and saved_iter
            self.run.n_epochs = float(saved_epoch)
            self.run.n_iter = int(saved_iter)
            self.run.val_iter = 0  # TODO: how?, maybe get len(val_dl) * num_epoch
            self.run.num_epochs = int(saved_epoch)
            self.run.start_epoch = int(saved_epoch)

        if not self.cont:
            return True  # Stop here


class SetEpochDistCallback(Callback):
    def begin_epoch(self):
        config = self.learn.glue.config
        if config.distributed:
            # set epoch for train_dl sampler
            print(f'Setting epoch {self.run.n_epochs} for train_dl distributed sampler')
            self.learn.data.train_dl.sampler.set_epoch(self.run.n_epochs)


class Normalizer:
    def __init__(self, mean, std, device=None, inplace=True):
        self.mean, self.std, self.inplace = torch.tensor(mean), torch.tensor(std), inplace
        if device:
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)

    def __call__(self, data):
        if self.inplace:
            data.sub_(self.mean).div_(self.std)
            # print(f"Mean: {data.mean()}, std: {data.std()}")
            return data
        else:
            # print(f"Mean: {data.mean()}, std: {data.std()}")
            return (data - self.mean) / self.std
