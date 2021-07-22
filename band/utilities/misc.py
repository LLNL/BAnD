import numpy as np
import torch


def get_param_str(args, param_ls=None, kv_sep="=", param_sep="-"):
    if not param_ls:
        param_ls = list(args.__dict__.keys())

    arg_dict = args.__dict__
    params = []
    for k in param_ls:
        if k in arg_dict:
            v = arg_dict[k]
            params.append(f"{k}{kv_sep}{v}")

    param_str = param_sep.join(params)

    return param_str


# https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = m / (m + n) * self.std ** 2 + n / (m + n) * newstd ** 2 + \
                       m * n / (m + n) ** 2 * (tmp - newmean) ** 2
            self.std = np.sqrt(self.std)

            self.nobservations += n


class StatsRecorderTorch:
    def __init__(self, data=None, device=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            # data = np.atleast_2d(data)
            self.mean = data.mean()
            self.std = data.std()

            if device is not None:
                self.mean = torch.tensor(self.mean).to(device)
                self.std = torch.tensor(self.std).to(device)

            self.nobservations = data.size(0)
            self.ndimensions = data.size(1)
        else:
            self.nobservations = 0

        self.device = device

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            # data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean()
            newstd = data.std()

            m = torch.tensor(self.nobservations * 1.0).to(self.device)
            n = data.size(0)

            tmp = self.mean

            self.mean = m / (m + n) * tmp + n / (m + n) * newmean
            self.std = m / (m + n) * self.std ** 2 + n / (m + n) * newstd ** 2 + \
                       m * n / (m + n) ** 2 * (tmp - newmean) ** 2

            self.std = np.sqrt(self.std)

            self.nobservations += n
