from band.utilities.tools import *


class MNISTCacheDataset(CacheDataset):
    def __init__(self, x, y, block_size, workers):
        super(MNISTCacheDataset, self).__init__(block_size, workers)
        self.x, self.y = x, y

    def fetch(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)


class HCPCacheDataset(CacheDataset):
    def __init__(self, files, base_path, label_file, transform, block_size, workers):
        """
        init this dataset
        :param files: list of subjects (name only, not the whole path)
        :param base_path: base of npy files path
        :param label_file: path to label file (name only, whole path: base_path / label_file)
        :param transform: transform on the data
        """
        super(HCPCacheDataset, self).__init__(transform, block_size, workers)

        self.files = files
        self.base_path = Path(base_path)
        self.transform = transform

        self.label_file = label_file
        self.iid_index = 0
        self.label_index = 3

        # load labels
        self.labels = self.load_label()

    def load_label(self):
        fn = self.label_file
        p = self.base_path / fn

        assert p.exists(), "Label file doesn't exist: {}".format(str(p))

        labels = {}
        with open(str(p), 'r') as f:
            lines = f.readlines()

            for line in lines:
                fields = line.split(' ')
                iid = str(fields[self.iid_index])
                label = int(fields[self.label_index])
                labels[iid] = label

        return labels

    def fetch(self, i):
        fp = Path(self.files[i])
        iid = str(fp.stem)
        data = self.load_npy(fp)
        label = self.labels[iid]

        sample = {'data': data, 'label': label, 'id': iid}

        return sample

    def load_npy(self, fp):
        path = self.base_path / fp
        data = np.load(str(path))
        return data

    def __len__(self):
        return len(self.files)


class HCPPrecompDataset(Dataset):
    def __init__(self, precomp_file, label_file, transform):
        """
        init this dataset
        :param precomp_pkl: precomp pickle file (full path)
        :param label_file: path to label file (full path)
        :param transform: transform on the data
        """
        super(HCPPrecompDataset, self).__init__()

        self.precomp_file = Path(precomp_file)
        self.label_file = Path(label_file)

        assert precomp_file.exists() and self.label_file.exists()

        self.transform = transform

        #### Data
        self.data = self.load_precomp()

        #### Labels
        # label indices, might be different from indices in self.data (precomp pkl)
        self.iid_index = 0
        self.label_index = 3
        # load labels
        self.labels = self.load_label()

    def load_label(self):
        fp = Path(self.label_file)

        assert fp.exists(), "Label file doesn't exist: {}".format(str(fp))

        labels = {}
        with open(str(fp), 'r') as f:
            lines = f.readlines()

            for line in lines:
                fields = line.split(' ')
                iid = str(fields[self.iid_index])
                label = int(fields[self.label_index])
                labels[iid] = label

        return labels

    def __getitem__(self, i):
        item = self.data[i]
        # item, precomp from pickle, [iid, label, emb (32, 512)]
        iid = item[0]
        label_from_precomp = item[1]
        embs = item[2]

        # verify label
        label_from_label_file = self.labels[iid]
        assert int(label_from_label_file) == int(label_from_precomp)

        sample = {'data': embs, 'label': label_from_label_file, 'id': iid}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_precomp(self):
        fp = Path(self.precomp_file)

        print(f"Loading: {fp}")
        with open(fp, 'rb') as f:
            data = pickle.load(f)

        return data

    def __len__(self):
        return len(self.data)
