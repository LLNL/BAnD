from scaffold import *
from fwriter import FWriter

class Glue:
    def set_config(self, config):
        self.config = config

    def get_files(self): raise NotImplementedError

    def get_transforms(self): raise NotImplementedError

    def get_ds(self): raise NotImplementedError

    def get_dl(self): raise NotImplementedError

    def get_model(self): raise NotImplementedError

    def __getattr__(self, k):
        return getattr(self.config, k, None)


def get_test_val_train_split(data_list, valid_size=0.1, test_size=0.1, seed=None):
    N = len(data_list)
    test_size = int(np.floor(test_size * N))
    val_size = int(np.floor(valid_size * N))
    train_size = N - test_size - val_size

    if seed:
        assert isinstance(seed, int), f"{seed} must be an int"
        import random
        print(f"\nGetting test_val_train split using seed {seed}\n")
        rnd = random.Random(seed)

        perm = rnd.sample(data_list, len(data_list))

        return perm[:test_size], perm[test_size:test_size + val_size], perm[test_size + val_size:]
    else:
        return data_list[:test_size], data_list[test_size:test_size + val_size], data_list[test_size + val_size:]


def make_writer(C):
    logger = FWriter(is_debug=C.is_debug, fn=str(C.log_file))
    return logger


def log_init_info(writer, config, keys):
    config_dict = vars(config)
    for key in keys:
        writer.write(f"{key}: {config_dict[key]}")
        if not writer.is_debug:
            print(f"{key}: {config_dict[key]}")
    print()
    writer.write(f"Config: {config}")
    print()


def save_model(model, out_dir, fn, args, fwriter):
    meta_data = {
        'args': args,
        'model_str': str(model)
    }

    opath = out_dir / fn
    fwriter.write("\nSaving model+meta to {}\n".format(str(opath)))

    # save model
    torch.save(model.state_dict(), "{}.pt".format(opath))

    # save meta
    meta_path = "{}.meta".format(opath)

    with open(meta_path, 'wb') as f:
        pickle.dump(meta_data, f)


def restore_model(model, out_dir, fn):
    def get_epoch():
        # expect fn: model.epoch-1.batch-5999.pt
        epoch_str = fn.split('.')[1]
        epoch = int(epoch_str.split('-')[1])
        print("Restore from epoch: {}".format(epoch))
        return epoch

    buffer_list = ['running_mean', 'running_var', 'num_batches_tracked', 'transformer.pe.pe']
    resume_path = out_dir / fn

    print('Restoring model with: {}'.format(str(resume_path)))
    state_path = "{}.pt".format(resume_path)
    meta_path = "{}.meta".format(resume_path)

    assert Path(state_path).exists(), "Resume model: state path: %s doesn't exist." % str(resume_path)
    assert Path(meta_path).exists(), "Resume model: meta path: %s doesn't exist." % str(resume_path)

    state_dict = torch.load(state_path)

    model_params = [p[0] for p in model.named_parameters()]
    # for key in list(state_dict.keys()):
    #     # keep registered buffer in buffer_list
    #     if key not in buffer_list and key not in model_params:
    #             print("Key {} not in model_params, removing...".format(key))
    #             del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()

    # load meta
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # extract epoch from model
    epoch = get_epoch()

    meta['restore_meta'] = {
        'epoch': epoch
    }

    return model, meta
