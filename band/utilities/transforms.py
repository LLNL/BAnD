from band.utilities.tools import *


class ToNumpy(object):
    """
    Convert raw lists to ndarrays
    """

    def __call__(self, sample):
        t = time.time()
        data, label = sample['data'], sample['label']
        npdata = np.asarray(data)
        nplabel = np.asarray(label)

        sample['data'] = npdata
        sample['label'] = nplabel

        print("\t\tToNumpy: %.2f" % (time.time() - t))
        return sample


class RandomCropFrame(object):
    """Crop randomly the image in a sample.

    Args:
        nframes (int): Desired number of frames
    """

    def __init__(self, nframes, random_crop=True, padding=True, pad='zero', return_original=True):
        assert isinstance(nframes, int)
        self.nframes = nframes
        self.padding = padding
        self.random_crop = random_crop
        self.pad = pad

    def __call__(self, sample):
        data = sample['data']
        data, mask = self.crop(data)

        sample['data'] = data
        sample['mask'] = mask
        return sample

    def crop(self, d):
        """
        crop d to self.nframes length, with additional padding if required
        :param d: a numpy array of data, with 4 dim
        :return:
        """
        h, w, depth, nf = d.shape[:4]
        new_nf = self.nframes
        # find a good random start of frames
        diff = nf - new_nf

        mask = np.ones((self.nframes), dtype=int)

        if not self.padding:
            assert diff >= 0, "Diff is smaller than 0, wrong, nf: {}, new_nf: {}".format(nf, new_nf)
        else:
            # if diff >=0, this loop is not gonna run, so only for when diff < 0
            remainder = -diff
            # if remainder > 0:
            #     print("Padding: nf: {}, new_nf: {}".format(nf, new_nf))

            if (self.pad == 'zero'):
                # pad with zeros instead
                zeros = np.zeros((h, w, depth, 1), dtype=np.float32)
                for i in range(remainder):
                    idx = i % nf
                    d = np.append(d, zeros[:, :, :, 0:1], axis=3)
            elif (self.pad == 'self'):
                for i in range(remainder):
                    idx = i % nf
                    # pad d by concat itself, maybe multiple times if a long new_nf is needed
                    # appending slice at idx to d itself, that's why axis is 3
                    d = np.append(d, d[:, :, :, idx:idx + 1], axis=3)
            else:
                raise NotImplementedError

            # fill in the rest of mask to be zeros
            mask[-remainder:] = 0

        diff = d.shape[3] - new_nf
        if not self.random_crop:
            # if not random_crop -> start is 0
            start = 0
        else:
            if diff == 0:
                start = 0  # random.randint doesnt work with (0, 0)
            else:
                # d.shape[3] might have changed because of padding, so gotta call it again
                start = np.random.randint(0, diff)

        # cropped!
        d = d[:, :, :, start:start + new_nf]

        return d, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = torch.from_numpy(data)
        # label = torch.tensor(label)

        sample['data'] = data
        sample['label'] = label

        return sample


class SeriesShuffled(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data = sample['data']

        # data: [n_frames, n_hidden]
        np.random.shuffle(data)  # shuffled in place
        # data = torch.from_numpy(data)
        # label = torch.tensor(label)

        sample['data'] = data

        return sample


class GetLengthAndMask(object):
    """Get length and mask of sample['data']"""

    def __init__(self, max_frames):
        self.max_frames = max_frames

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: sample with 'length' and 'mask' key
        """
        data = sample['data']
        length = data.size(-1)  # assuming dim -1 is the length dim
        mask = torch.zeros(self.max_frames)
        mask[:length] = 1.

        sample['length'] = length
        sample['mask'] = mask

        return sample


class CenterCrop(object):
    """Get length and mask of sample['data']"""

    def __init__(self, sizes):
        self.sizes = sizes

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: center cropped sample['data'] with self.sizes
        """
        data = sample['data']
        # expected: data for tml project: [91, 109, 91, nframes]
        # center crop data

        s_ = []
        for s0, s1 in zip(self.sizes, list(data.shape)):
            # s1 > s0
            assert s1 > s0, f"CenterCrop: {s1} has to be larger than {s0}"
            s_.append((s1 - s0) // 2)

        data = data[s_[0]:s_[0] + self.sizes[0], s_[1]:s_[1] + self.sizes[1], s_[2]:s_[2] + self.sizes[2]]

        sample['data'] = data

        return sample


class TimeCrop(object):
    """Get length and mask of sample['data']"""

    def __init__(self, max_frames=None, random_crop=False, center_crop=False):
        self.max_frames, self.random_crop, self.center_crop = max_frames, random_crop, center_crop

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: center cropped sample['data']'s time dimension with self.max_frames
        """

        if not self.max_frames:
            return sample

        assert not (self.center_crop and self.random_crop), f"TimeCrop: can't have both random_crop and center_crop"

        data = sample['data']
        # expected: data for tml project: [91, 109, 91, nframes]
        n = data.shape[-1]
        start_frame = 0
        if self.center_crop:
            assert n >= self.max_frames, f"TimeCrop: max_frames {self.max_frames} should be < n {n}"
            start_frame = (n - self.max_frames) // 2
        if self.random_crop:
            diff = n - self.max_frames
            if diff == 0:
                start_frame = 0  # random.randint doesnt work with (0, 0)
            elif diff > 0:
                start_frame = np.random.randint(0, diff)

        data = data[:, :, :, start_frame:start_frame + self.max_frames]

        sample['data'] = data

        return sample


# class TimeCropAndPad(object):
#     """Get length and mask of sample['data']"""
#
#     def __init__(self, max_frames=None, random_crop=False, self_pad=False, n=1, random_head=False):
#         self.max_frames, self.random_crop, self.self_pad, self.n, self.random_head = max_frames, random_crop, self_pad, n, random_head
#
#     def __call__(self, sample):
#         """
#         :param sample: Torch tensor
#         :return: center cropped sample['data']'s time dimension with self.max_frames
#         """
#
#         if self.n == 1:
#             if not self.max_frames:
#                 return sample
#
#             data = self._crop_and_pad(sample['data'])
#             sample['data'] = data
#
#         elif self.n > 1:
#             assert (self.random_crop and self.self_pad), "Random crop and self pad must be on for n > 1"
#
#             w, h, d, nframes = sample['data'].shape
#             # 91, 109, 91, nframes
#             data = np.zeros(shape=(self.n, w, h, d, self.max_frames), dtype=np.float32)
#             for i in range(self.n):
#                 data[i] = self._crop_and_pad(sample['data'])
#
#             sample['data'] = data
#
#         return sample
#
#     def _crop_and_pad(self, target):
#         data = np.copy(target)
#         n = data.shape[-1]
#         start_frame = 0
#         if self.random_crop:
#             diff = n - self.max_frames
#             if diff == 0:
#                 start_frame = 0  # random.randint doesnt work with (0, 0)
#             elif diff > 0:
#                 start_frame = np.random.randint(0, diff)
#
#         # get a random start for data with n < max_frames also (enable with self.random_head == True)
#         if n < self.max_frames and self.self_pad and self.random_head:
#             start_frame = np.random.randint(0, n // 2)
#
#         data = data[:, :, :, start_frame:start_frame + self.max_frames]
#
#         # pad
#         n = data.shape[-1]
#         if self.self_pad:
#             for i in range(self.max_frames - n):
#                 idx = i % n
#                 data = np.append(data, data[:, :, :, idx:idx + 1], axis=3)
#
#         return data


class SkipFrame(object):
    def __init__(self, n=1, to_skip=False):
        self.n, self.to_skip = n, to_skip

    def __call__(self, sample):
        # sample: (w, h, d, nframes)
        if not self.to_skip:
            return sample

        data = sample['data']

        w, h, d, nframes = data.shape
        # skip every n frame
        new_data = []
        for i in range(0, nframes, self.n):
            new_data.append(data[:, :, :, i])

        data = np.asarray(new_data).transpose((1, 2, 3, 0))
        # print(data.shape)

        sample['data'] = data

        return sample


class TimeCropAndPadPrecomp(object):
    def __init__(self, max_frames=None, random_crop=False, self_pad=False, pad_mode="loop", min_start=999, stride=1):
        self.max_frames, self.random_crop = max_frames, random_crop
        self.self_pad, self.pad_mode = self_pad, pad_mode
        self.min_start = min_start
        self.stride = stride

    def __call__(self, sample):
        """
        :param sample: numpy array, [nframes, d]
        :return: center cropped sample['data']'s time dimension with self.max_frames
        """
        if not self.max_frames:
            return sample

        data = self._crop(sample['data'])
        sample['data'] = data

        return sample

    def _crop(self, target):
        data = np.copy(target)
        n = data.shape[0]
        start_frame = 0

        # this only works for when n > self.max_frames
        if self.random_crop:
            if self.stride > 0:
                diff = n - ((self.max_frames + 1) * self.stride)
            else:
                diff = n - self.max_frames

            if diff == 0:
                start_frame = 0  # random.randint doesnt work with (0, 0)
            elif diff > 0:
                # diff = min(diff, self.min_start)  # remove min_start trick
                start_frame = np.random.randint(0, diff)

        # get a random start for data with n < max_frames also (enable with self.random_head == True)
        # if n < self.max_frames and self.self_pad and self.random_head:
        #     start_frame = np.random.randint(0, n // 2)

        if self.stride == 0:
            data = data[start_frame:start_frame + self.max_frames, :]
        else:
            cropped_data = []

            # upper = min(((self.max_frames) * self.stride), n)
            for i in range(start_frame, start_frame + ((self.max_frames - 1) * self.stride), self.stride):
                cropped_data.append(data[i])

            data = np.stack(cropped_data)


        # pad
        # precomp data: [nframes, d]
        n = data.shape[0]
        if self.self_pad and self.max_frames > n:
            diff_1 = self.max_frames - n - 1

            if self.pad_mode == "loop":
                for i in range(self.max_frames - n):
                    idx = i % n
                    data = np.append(data, data[idx:idx + 1, :], axis=0)

        return data


class TimeCropAndPad(object):
    """Get length and mask of sample['data']"""

    def __init__(self, max_frames=None, random_crop=False, self_pad=False, n=1, random_head=False, pad_mode="loop",
                 min_start=15):
        self.max_frames, self.random_crop, self.self_pad, self.n, self.random_head, self.pad_mode = max_frames, random_crop, self_pad, n, random_head, pad_mode
        self.min_start = min_start

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: center cropped sample['data']'s time dimension with self.max_frames
        """

        if self.n == 1:
            if not self.max_frames:
                return sample

            data = self._crop_and_pad(sample['data'])
            sample['data'] = data

        elif self.n > 1:
            assert (self.random_crop and self.self_pad), "Random crop and self pad must be on for n > 1"

            w, h, d, nframes = sample['data'].shape
            # 91, 109, 91, nframes
            data = np.zeros(shape=(self.n, w, h, d, self.max_frames), dtype=np.float32)
            for i in range(self.n):
                data[i] = self._crop_and_pad(sample['data'])

            sample['data'] = data

        return sample

    def _crop_and_pad(self, target):
        data = np.copy(target)
        n = data.shape[-1]
        start_frame = 0
        if self.random_crop:
            diff = n - self.max_frames
            if diff == 0:
                start_frame = 0  # random.randint doesnt work with (0, 0)
            elif diff > 0:
                # diff = min(diff, self.min_start)
                start_frame = np.random.randint(0, diff)

        # get a random start for data with n < max_frames also (enable with self.random_head == True)
        if n < self.max_frames and self.self_pad and self.random_head:
            start_frame = np.random.randint(0, n // 2)

        data = data[:, :, :, start_frame:start_frame + self.max_frames]

        # pad
        n = data.shape[-1]
        if self.self_pad and self.max_frames > n:
            diff_1 = self.max_frames - n - 1

            if self.pad_mode == "loop":
                for i in range(self.max_frames - n):
                    idx = i % n
                    data = np.append(data, data[:, :, :, idx:idx + 1], axis=3)
            elif self.pad_mode == "duplicate":
                new_data = []
                for i in range(n):
                    this_frame = data[:, :, :, i]
                    new_data.append(this_frame)

                    # SN: count = count for all frames + count for the fact that this frame is the remainder
                    count = (diff_1 // n) + ((diff_1 % n) >= i)

                    for _ in range(count):
                        new_data.append(this_frame)

                assert len(new_data) == self.max_frames, "WRONG CALCULATION, BRO"

                data = np.asarray(new_data).transpose((1, 2, 3, 0))

            elif self.pad_mode == "weighted_duplicate":
                new_data = []
                for i in range(n):
                    this_frame = data[:, :, :, i]
                    if i == n - 1:  # last frame
                        next_frame = data[:, :, :, i]
                    else:
                        next_frame = data[:, :, :, i + 1]

                    new_data.append(this_frame)

                    count = (diff_1 // n) + ((diff_1 % n) >= i)

                    for j in range(1, count + 1):
                        ratio = j / (count + 1)
                        between_frame = (this_frame * (1 - ratio) + next_frame * ratio)
                        new_data.append(between_frame)

                assert len(new_data) == self.max_frames, "WRONG CALCULATION, BRO"

                data = np.asarray(new_data).transpose((1, 2, 3, 0))

            else:
                raise ValueError(f"Pad mode: {self.pad_mode} not implemented.")

        return data

class TimeCropAndPadStrided(object):
    """Get length and mask of sample['data']"""

    def __init__(self, max_frames=None, random_crop=False, self_pad=False, n=1, random_head=False, pad_mode="loop",
                 min_start=999, stride=1):
        self.max_frames, self.random_crop, self.self_pad, self.n, self.random_head, self.pad_mode = max_frames, random_crop, self_pad, n, random_head, pad_mode
        self.min_start = min_start
        self.stride = stride

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: center cropped sample['data']'s time dimension with self.max_frames
        """

        if self.n == 1:
            if not self.max_frames:
                return sample

            data = self._crop_and_pad(sample['data'])
            sample['data'] = data

        elif self.n > 1:
            assert (self.random_crop and self.self_pad), "Random crop and self pad must be on for n > 1"

            w, h, d, nframes = sample['data'].shape
            # 91, 109, 91, nframes
            data = np.zeros(shape=(self.n, w, h, d, self.max_frames), dtype=np.float32)
            for i in range(self.n):
                data[i] = self._crop_and_pad(sample['data'])

            sample['data'] = data

        return sample

    def _crop_and_pad(self, target):
        data = np.copy(target)
        n = data.shape[-1]
        start_frame = 0
        if self.random_crop:
            # diff = n - self.max_frames
            diff = n - ((self.max_frames + 1) * self.stride)
            if diff == 0:
                start_frame = 0  # random.randint doesnt work with (0, 0)
            elif diff > 0:
                diff = min(diff, self.min_start)
                start_frame = np.random.randint(0, diff)

        # get a random start for data with n < max_frames also (enable with self.random_head == True)
        if n < self.max_frames and self.self_pad and self.random_head:
            start_frame = np.random.randint(0, n // 2)

        # data = data[start_frame:start_frame + self.max_frames, :]
        cropped_data = []

        # upper = min(((self.max_frames) * self.stride), n)
        for i in range(start_frame, start_frame + ((self.max_frames - 1) * self.stride), self.stride):
            cropped_data.append(data[:, :, :, i])

        data = np.stack(cropped_data, axis=-1)  # stacked along last dimension

        # data = data[:, :, :, start_frame:start_frame + self.max_frames]

        # pad
        n = data.shape[-1]
        if self.self_pad and self.max_frames > n:
            diff_1 = self.max_frames - n - 1

            if self.pad_mode == "loop":
                for i in range(self.max_frames - n):
                    idx = i % n
                    data = np.append(data, data[:, :, :, idx:idx + 1], axis=3)
            elif self.pad_mode == "duplicate":
                new_data = []
                for i in range(n):
                    this_frame = data[:, :, :, i]
                    new_data.append(this_frame)

                    # SN: count = count for all frames + count for the fact that this frame is the remainder
                    count = (diff_1 // n) + ((diff_1 % n) >= i)

                    for _ in range(count):
                        new_data.append(this_frame)

                assert len(new_data) == self.max_frames, "WRONG CALCULATION, BRO"

                data = np.asarray(new_data).transpose((1, 2, 3, 0))

            elif self.pad_mode == "weighted_duplicate":
                new_data = []
                for i in range(n):
                    this_frame = data[:, :, :, i]
                    if i == n - 1:  # last frame
                        next_frame = data[:, :, :, i]
                    else:
                        next_frame = data[:, :, :, i + 1]

                    new_data.append(this_frame)

                    count = (diff_1 // n) + ((diff_1 % n) >= i)

                    for j in range(1, count + 1):
                        ratio = j / (count + 1)
                        between_frame = (this_frame * (1 - ratio) + next_frame * ratio)
                        new_data.append(between_frame)

                assert len(new_data) == self.max_frames, "WRONG CALCULATION, BRO"

                data = np.asarray(new_data).transpose((1, 2, 3, 0))

            else:
                raise ValueError(f"Pad mode: {self.pad_mode} not implemented.")

        return data


class TimePad(object):
    """Get length and mask of sample['data']"""

    def __init__(self, max_frames=None):
        self.max_frames = max_frames

    def __call__(self, sample):
        """
        :param sample: Torch tensor
        :return: center cropped sample['data']'s time dimension with self.max_frames
        """

        if not self.max_frames:
            return sample

        data = sample['data']
        # expected: data for tml project: [91, 109, 91, nframes]
        n = data.shape[-1]
        assert self.max_frames >= n, f"TimePad: max_frames {self.max_frames} should be >= n {n}"

        zeros = np.zeros_like(data, dtype=np.float32)
        for i in range(self.max_frames - n):
            idx = i % n
            data = np.append(data, data[:, :, :, idx:idx + 1], axis=3)
            # data = np.append(data, zeros[:, :, :, 0:1], axis=3)

        sample['data'] = data

        return sample


class Permute(object):
    """Permute Tensors. Expect sample to already be Torch tensors"""

    def __init__(self, data_perm=None, label_perm=None):
        self.data_perm = data_perm
        self.label_perm = label_perm

    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        # ipdb.set_trace()
        if self.data_perm:
            data = data.permute(self.data_perm)
        if self.label_perm:
            label = label.permute(self.label_perm)

        sample['data'] = data
        sample['label'] = label
        return sample
