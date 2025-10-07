from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from data_provider.data_loader_benchmark import UTSDDatasetBenchmark, CIDatasetBenchmark
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'UTSD': UTSDDatasetBenchmark,
    'CI': CIDatasetBenchmark,
}


def data_provider(args, flag):
    Data = data_dict[args.data]    
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'long_term_forecast' or args.task_name == 'long_term_forecast_chronos':          
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            scale=True,
            timeenc=timeenc,
            freq=args.freq,
            percent=args.percent,
            task_name=args.task_name,
            is_pretraining=args.is_pretraining,
        )
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=drop_last,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError
