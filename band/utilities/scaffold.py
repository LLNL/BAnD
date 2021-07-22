# Author: Jeremy Howard (FastAI)
# Modified by Sam Nguyen (samiam@llnl.gov)

from pathlib import Path
import pickle, gzip, math, torch
from torch import tensor
import operator


def test(a, b, cmp, cname=None):
    if cname is None: cname = cmp.__name__
    assert cmp(a, b), f"{cname}:\n{a}\n{b}"


def test_eq(a, b): test(a, b, operator.eq, '==')


def near(a, b): return torch.allclose(a, b, rtol=1e-3, atol=1e-5)


def test_near(a, b): test(a, b, near)

mnist_path = Path("../data/mnist.pkl.gz")
def get_data():
    path = mnist_path
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train,y_train,x_valid,y_valid))

def normalize(x, m, s): return (x-m)/s

def test_near_zero(a,tol=1e-3): assert a.abs()<tol, f"Near zero: {a}"

from torch.nn import init

def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()

from torch import nn

import torch.nn.functional as F


def accuracy(out, yb): return (torch.argmax(out, dim=1) == yb).float().mean()


from torch import optim


class Dataset():
    def __init__(self, x, y): self.x, self.y = x, y

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i], self.y[i]


from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs * 2, **kwargs))

import numpy as np
from torch.utils.data import Dataset, DataLoader
import tqdm

class FixedListIter:
    def __init__(self, li):
        self.li = li
        self.i = 0
        self.max = len(li)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.max:
            n = self.li[self.i]
            self.i += 1
            return n
        else:
            raise StopIteration

    def __len__(self):
        return len(self.li)


class BlockDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(BlockDataLoader, self).__init__(*args, **kwargs)
        # bind this loader to its dataset
        ds = getattr(self.dataset, 'loader', None)
        if ds:
            raise Exception('Dataset is already bound to another dataloader')
        else:
            setattr(self.dataset, 'loader', self)

    def __iter__(self):
        self._indices = list(self.sampler)
        # print(f"DataBlockLoader: _indices: {self._indices}")
        self.batch_sampler.sampler = FixedListIter(self._indices)
        return super(BlockDataLoader, self).__iter__()


import concurrent.futures
import time


class CacheDataset(Dataset):
    def __init__(self, transform=None, block_size=50, workers=2, keep_all=False, verbose=True):
        self.cache = {}
        self.block_size = block_size
        self.workers = workers
        self.transform = transform
        self.keep_all = keep_all
        self.verbose = verbose

    def preload_from(self, i):
        def groupify(length, chunks, arr=None):
            remainder = length % chunks
            n = max(1, length // chunks)
            groups = [list(range(n * i, n * (i + 1))) for i in range(chunks)]
            # go over each list and fill in the remainder
            max_so_far = n * chunks
            for i in range(remainder):
                groups[i].append(max_so_far + i)

            if arr:
                return [[arr[i] for i in g] for g in groups]
            else:
                return groups

        # find indices of i in loader.indices
        loader = getattr(self, 'loader', None)
        if not loader:
            raise Exception('Loader is not set. Cannot preload data.')
        if loader and not getattr(loader, '_indices', None):
            raise Exception('Loader does not have `_indices`. Please use BlockDataLoader.')

        idx = self.loader._indices.index(i)
        to_load = self.loader._indices[idx:idx + self.block_size]
        n = len(to_load)

        assert self.workers > 0, "Please specify num_workers for preloading. Current val: {}".format(self.workers)
        # make sure to use only number of workers <= n data points
        _num_workers = self.workers
        self.workers = min(n, self.workers)
        if self.verbose:
            print('Preloading {} data points with {} max workers.'.format(n, self.workers))

        groups = groupify(n, self.workers, arr=to_load)

        executor_args = []
        for i in range(len(groups)):
            show_progress = i == -1  # not showing progress anymore, bug: printing on newline -> bad
            executor_args.append((groups[i], show_progress))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            for (arg, result) in zip(groups, executor.map(self.load, executor_args)):
                for i in arg:
                    self.cache[i] = result[i]

        # reset self.workers (next batch is likely to have a bigger size if this batch n < self.workers)
        self.workers = _num_workers

    def load(self, args):
        group, show_progress = args
        cache = {}
        itera = tqdm.tqdm(group) if show_progress else group
        for i in itera:
            mini_data = self.get_item(i)
            cache[i] = mini_data
            # a hack
            # TODO: maybe this can be removed now?
            # time.sleep(0.001)

        return cache

    def fetch(self, i):
        raise NotImplementedError

    def get_item(self, i):
        item = self.fetch(i)

        if self.transform:
            item = self.transform(item)

        return item

    def release_cache(self):
        # delete self.cache to free memoory
        if self.cache:
            del self.cache
            self.cache = {}

    def __getitem__(self, i):
        if i not in self.cache:
            # if i not in cache, then all preloaded data in cache has been consumed
            self.release_cache()
            self.preload_from(i)

        return self.cache[i]


class DataBunch():
    def __init__(self, train_dl, valid_dl, test_dl=None, c=None):
        self.train_dl, self.valid_dl, self.test_dl, self.c = train_dl, valid_dl, test_dl, c

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset

    @property
    def test_ds(self): return self.test_dl.dataset


class Learner():
    def __init__(self, model, opt, loss_func, data, glue):
        self.model, self.opt, self.loss_func, self.data, self.glue = model, opt, loss_func, data, glue


import re

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


class Callback():
    # this is attacked to an instance of this class anyway, no need for self
    _order = 0

    def set_runner(self, run): self.run = run

    def __getattr__(self, k):
        return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0
        self.run.val_iter = 0
        self.run.num_epochs = 0
        self.run.start_epoch = 0  # These might be changed by a ResumeCallback
        self.run.in_test = False

    def after_batch(self):
        if not self.in_train:
            self.run.val_iter += 1
        else:
            self.run.n_epochs += 1. / self.iters
            self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
        self.run.in_test = False

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False
        self.run.in_test = False

    def after_epoch(self):
        self.run.num_epochs += 1


from typing import *


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        # turn cbs into a list
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            # self here refers to the runner, set callback name to an attr of the runner
            setattr(self, cb.name, cb)
            cbs.append(cb)
        # TrainEvalCallback always at the start
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs

    @property
    def opt(self):
        return self.learn.opt

    @property
    def model(self):
        return self.learn.model

    @property
    def loss_func(self):
        return self.learn.loss_func

    @property
    def data(self):
        return self.learn.data

    def one_batch(self, sample):
        # get data
        self.sample = sample
        # self has access to sample now
        # if a cb returns yes -> interrupt
        # if return no -> continue
        if self('begin_batch'): return
        # begin_batch cb needs to set tuple self.inputs
        self.pred = self.model(*self.inputs)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return

        # SN: important for distributed, if a module doesn't have grad (frozen, not used etc) -> error!
        # for name, param in self.model.named_parameters():
        #     print(name, param, True if param.grad is not None else False)

        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()

    def one_batch_validate(self, sample):
        # get data
        self.sample = sample
        # self has access to sample now
        # if a cb returns yes -> interrupt
        # if return no -> continue
        if self('begin_batch'): return
        # begin_batch cb needs to set tuple self.inputs
        self.pred = self.model(*self.inputs)
        if self('after_pred'): return
        self.loss = self.loss_func(self.pred, self.yb)
        if self('after_loss') or not self.in_train: return
        self.loss.backward()
        if self('after_backward'): return
        self.opt.step()
        if self('after_step'): return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for sample in dl:
            if self.stop: break
            self.one_batch(sample)
            self('after_batch')
        self.stop = False

    def fit(self, epochs, learn):
        self.epochs, self.learn, self.loss = epochs, learn, tensor(0.)

        try:
            # set runner for all cb
            for cb in self.cbs: cb.set_runner(self)
            if self('begin_fit'):
                with torch.no_grad():
                    # only begin test if there's a callback that defines begin_test
                    if self('begin_test'): self.all_batches(self.data.test_dl)
                    self('after_epoch')

                return

            for epoch in range(self.start_epoch, self.epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)

                if self('after_epoch'): break

            # If test is needed
            with torch.no_grad():
                # only begin test if there's a callback that defines begin_test
                if self('begin_test'): self.all_batches(self.data.test_dl)
                self('after_epoch')  # to record stats

        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            # True means: if a callback exists and that callback wants to stop -> return True (kinda confusing)
            if f and f(): return True
        # False means: there's no such callback (property) registered or, a callback wants to continue
        return False

from functools import partial

def create_learner(model_func, loss_func, data):
    return Learner(*model_func(data), loss_func, data)





# def get_model_func(lr=0.5): return partial(get_model, lr=lr)


def annealer(f):
    def _inner(start, end): return partial(f, start, end)

    return _inner


@annealer
def sched_lin(start, end, pos): return start + pos * (end - start)


@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


@annealer
def sched_no(start, end, pos):  return start


@annealer
def sched_exp(start, end, pos): return start * (end / start) ** pos


# This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))


def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        #         import ipdb; ipdb.set_trace()
        #         print(pos)
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos)

    return _inner



