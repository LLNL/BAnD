# BAnD's main entry with distributed training options
import argparse

from band.utilities.misc import *
from band.utilities.callbacks import *
from band.utilities.glue import *


def main(config):
    torch.distributed.init_process_group(backend='nccl')

    device = torch.device('cuda', args.local_rank)
    glue = HCPGlue(config, device)
    glue.init(distributed=True, local_rank=args.local_rank, world_rank=args.world_rank)

    train_shuffle = True
    data = DataBunch(*glue.get_dls(shuffle=train_shuffle)[:3], c=glue.n_classes)

    loss_func = torch.nn.CrossEntropyLoss()

    model = glue.get_model()
    # distributed model
    distrib_model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[args.local_rank],
                                                              output_device=args.local_rank)
    optimizer = glue.get_optimizer(distrib_model)

    learn = Learner(distrib_model, optimizer, loss_func, data, glue)

    sched = combine_scheds([0.4, 0.6],
                           [sched_cos(config.lr_start, config.lr_mid), sched_cos(config.lr_mid, config.lr_final)])

    validate = True
    # Note: need to pre-compute mean and std of data
    normalizer = Normalizer(mean=4032.9595, std=5278.9634, device=glue.device)

    cbfs = [partial(Recorder, save_recorder=False),
            # CallbackScheduleCallback,
            partial(LoggerCallback, tb=True),
            partial(ResumeCallback, cont=True),
            partial(PrepareDataCallback, normalizer),
            partial(StatLogCallback, accuracy),
            partial(AvgStatsCallback, accuracy),
            partial(ParamScheduler, 'lr', sched),
            partial(ValidateCallback, validate),
            ]

    run = Runner(cb_funcs=cbfs)
    run.fit(glue.n_epoch, learn)


if __name__ == "__main__":
    ##############
    # Parameters #
    ##############
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--initialization', default="xavier_unif", type=str)
    parser.add_argument('--preload_workers', type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--out_path', type=Path, required=True)
    parser.add_argument('--label_file', type=Path, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--is_debug', type=int, default=0)
    parser.add_argument('--is_restart', type=int, default=0)

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--lr_start', type=float, required=True)
    parser.add_argument('--lr_mid', type=float, required=True)
    parser.add_argument('--lr_final', type=float, required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--opt', type=str, required=True)

    parser.add_argument('--split_seed', type=int, required=True)
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--n_frames', type=int, required=True)
    parser.add_argument('--max_frames', type=int, required=True)
    parser.add_argument('--val_max_frames', type=int, required=True)

    parser.add_argument('--print_every', type=int, default=-1)
    parser.add_argument('--val_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)
    parser.add_argument('--save_every_epoch', type=int, default=-1)

    parser.add_argument('--test_size', type=float, required=True)
    parser.add_argument('--val_size', type=float, required=True)
    parser.add_argument('--train_block', type=int, required=True)
    parser.add_argument('--val_block', type=int, required=True)

    parser.add_argument('--pad_mode', type=str, required=True)
    parser.add_argument('--train_random_head', type=int, default=0)
    parser.add_argument('--val_random_crop', type=int, default=0)
    parser.add_argument('--skip_frame_n', type=int, default=0)
    parser.add_argument('--skip_frame_to_skip', type=int, default=0)

    parser.add_argument('--saved_model_path', type=Path, default=None)
    parser.add_argument('--saved_epoch', type=int, default=None)
    parser.add_argument('--saved_iter', type=int, default=None)

    # Logging stuff
    parser.add_argument('--tb_logger_path', default=None, type=Path)

    # Distributed training
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()

    # Get world_rank from env
    current_env = os.environ.copy()
    RANK = current_env["RANK"]
    print(f"RANK: {RANK}")
    args.world_rank = int(RANK)

    print("###################")
    print("#######ARGS########")
    print(args)
    print("###################")
    print("###################")
    param_ls = ["name", "model", "pad_mode", "test_size", "val_size", "n_classes", "n_epoch", "opt", "lr_start",
                "lr_mid", "lr_final", "max_frames", "val_max_frames", "batch_size"]
    param_str = get_param_str(args, param_ls=param_ls)
    print("###### PARAMS ##########")
    print(param_str)

    args.param_str = param_str

    main(args)
