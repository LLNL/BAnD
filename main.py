# BAnD's main entry
import argparse

from band.utilities.glue import *
from band.utilities.callbacks import *


def run(config):
    device = torch.device('cuda', 0)
    glue = HCPGlue(config, device)
    glue.init()

    data = DataBunch(*glue.get_dls(shuffle=False)[:2], glue.n_classes)
    loss_func = F.cross_entropy

    learn = Learner(*glue.get_model(), loss_func, data, glue)

    sched = combine_scheds([0.3, 0.7],
                           [sched_cos(config.lr_start, config.lr_mid), sched_cos(config.lr_mid, config.lr_final)])

    # Note: need to pre-compute mean and std of data
    normalizer = Normalizer(mean=4032.9595, std=5278.9634, device=glue.device)
    cbfs = [Recorder,
            CallbackScheduleCallback,
            partial(LoggerCallback, tb=True),
            partial(PrepareDataCallback, normalizer),
            partial(StatLogCallback, accuracy),
            partial(AvgStatsCallback, accuracy),
            partial(ParamScheduler, 'lr', sched),
            partial(ValidateCallback, validate=True),
            ]

    run = Runner(cb_funcs=cbfs)

    run.fit(glue.n_epoch, learn)


if __name__ == "__main__":
    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser(description="HCP")
    parser.add_argument("--exp", type=str, default="fmri")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-o", "--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False)

    parser.add_argument("-e", "--n_epoch", type=int, default=10)
    parser.add_argument("-l", "--lr", type=float, required=True)
    parser.add_argument("--lr_start", type=float, required=True)
    parser.add_argument("--lr_mid", type=float, required=True)
    parser.add_argument("--lr_final", type=float, required=True)
    parser.add_argument("--momentum", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--n_frames", type=int, required=True)
    parser.add_argument("--max_frames", type=int, required=True)
    parser.add_argument("--n_classes", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--preload_workers", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--random_crop_train", type=int, default=0)
    parser.add_argument("--random_crop_val", type=int, default=0)
    parser.add_argument("--train_block", type=int, default=600)
    parser.add_argument("--val_block", type=int, default=300)
    parser.add_argument("--gradient_step", type=int, default=1)
    parser.add_argument("--d_emb", type=int, default=4096)

    parser.add_argument("--trans_N", type=int, default=2)
    parser.add_argument("--trans_d_model", type=int, default=512)
    parser.add_argument("--trans_d_out", type=int, default=512)
    parser.add_argument("--trans_pe", type=int, default=1)
    parser.add_argument("--trans_pe_dropout", type=float, default=0.0)

    parser.add_argument("--pad", type=str, required=True)

    parser.add_argument("--is_restart", type=int, default=0)
    parser.add_argument("--is_debug", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=100)

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)

    parser.add_argument("--ckpt", type=str, required=False)
    parser.add_argument("--run_name", type=str, required=False)

    args = parser.parse_args()

    print("Args: {}".format(args))
    run(args)
