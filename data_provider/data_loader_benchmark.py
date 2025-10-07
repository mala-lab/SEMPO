import warnings
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from datasets import load_dataset
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe

warnings.filterwarnings('ignore')


class UTSDDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTh1.csv',
                 scale=True, timeenc=1, freq='h', percent=100, task_name='long_term_forecast',
                 is_pretraining=1, stride=1, split=0.9, horizon_lengths=[1,96,192,336,720]):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.percent = percent
        self.timeenc = timeenc
        self.scale = scale
        self.task_name = task_name
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.is_pretraining = is_pretraining
        self.horizon_lengths = horizon_lengths
       
        self.root_path = root_path
        self.dataset_name = root_path.rstrip('/').split('/')[-1]
        self.augment_resolution_dic = {
            'aus_electricity_demand': [1, 2, 48],    # meet 720 requirement
            'bitcoin': [1],                          # meet 720 requirement
            'kaggle_web_traffic': [1],               # meet 720 requirement
            'kdd_cup_2018': [1],                     # meet 720 requirement
            'london_smart_meters': [1, 2, 48],   # not all meet 720 requirement
            'LOOP_SEATTLE': [1, 2, 3, 6, 12],    # meet 720 requirement
            'LOS_LOOP': [1, 2, 3, 6, 12],        # meet 720 requirement
            'monash_weather': [1],               # meet 720 requirement
            'nn5': [1],
            'pems_bay': [1, 2, 3, 6, 12],        # meet 720 requirement
            'PEMS03': [1, 2, 3, 6, 12],          # meet 720 requirement
            'PEMS04': [1, 2, 3, 6, 12],          # meet 720 requirement
            'PEMS07': [1, 2, 3, 6, 12],          # meet 720 requirement
            'PEMS08': [1, 2, 3, 6, 12],          # meet 720 requirement
            'Q-TRAFFIC': [1, 2, 4],              # meet 720 requirement
            'saugeenday': [1],                   # meet 720 requirement
            'solar_4_seconds': [1, 150, 225, 450, 900],     # meet 720 requirement
            'solar_10_minutes': [1, 3, 6],       # meet 720 requirement
            'sunspot': [1],                      # meet 720 requirement
            'SZ_TAXI': [1, 2],                   # meet 720 requirement
            'us_births': [1],                    # meet 720 requirement
            'wind_4_seconds': [1, 150, 225, 450, 900],       # meet 720 requirement
            'wind_farms_minutely': [1, 10, 15, 30, 60],      # meet 720 requirement
        }
        self.__read_data__()
       
    def __process_each_sequence__(self, data):
        num_train = int(len(data) * self.split)
        num_test = int(len(data) * (1 - self.split) / 2)
        num_vali = len(data) - num_train - num_test
        # if num_train < self.seq_len:
            # continue
        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
       
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = data

        data = data[border1:border2]
        if self.is_pretraining:
            n_timepoint = (len(data) - self.seq_len - self.pred_len) // self.stride + 1
        else:
            n_timepoint = (len(data) - self.seq_len - self.horizon_lengths[-1]) // self.stride + 1
            # # Avoid short sequences less than 720
            # for horizon in reversed(self.horizon_lengths):
            #     n_timepoint = (len(data) - self.seq_len - horizon) // self.stride + 1
            #     if horizon == self.pred_len or n_timepoint >= 0:
            #         break
        return data, n_timepoint  
       

    def __read_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                subdataset_path = os.path.join(root, file)
                if file.endswith('.npy'):
                    data = np.load(subdataset_path)
                elif file.endswith('.npz'):
                    data = np.load(subdataset_path, allow_pickle=True)
                    data = data['data'][:, :, 0]
                elif file.endswith('.csv'):
                    df_raw = pd.read_csv(subdataset_path)
                    data = df_raw[df_raw.columns[1:]].values
                elif file.endswith('.txt'):
                    df_raw = []
                    with open(subdataset_path, "r", encoding='utf-8') as f:
                        for line in f.readlines():
                            line = line.strip('\n').split(',')
                            data_line = np.stack([float(i) for i in line])
                            df_raw.append(data_line)
                    df_raw = np.stack(df_raw, 0)
                    df_raw = pd.DataFrame(df_raw)
                    data = df_raw.values
                elif file.endswith('.parquet'):
                    ds = load_dataset('parquet', data_files=subdataset_path)
                    ds["train"].set_format("numpy")
                    # data = ds["train"]["target"]
                    columns = ds["train"].column_names
                    data = ds["train"][columns[2]]
                elif file.endswith('.arrow'):
                    ds = load_dataset('arrow', data_files=subdataset_path)
                    ds["train"].set_format("numpy")
                    data = ds["train"]["target"]
                elif file.endswith('.tsf'):
                    ds, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(subdataset_path)
                    data = [ts.astype(np.float32) for ts in ds.series_value]
                else:
                    raise ValueError('Unknown data format: {}'.format(subdataset_path))
               
                self.scaler = StandardScaler()
                dataset_list = data if file.endswith(('.parquet', '.arrow', '.tsf')) else [data]
                # downsample_dataset_list = []
               
                # process original data
                for data in dataset_list:
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    else:
                        if file.endswith('.arrow'):
                            data = data.T
                           
                    # preprocess sequence
                    data = data.copy()    # remove read-only restriction
                    # Missing Value Processing
                    data = data[~np.isnan(data).any(axis=1)]
                    # data = split_seq_by_nan_inf(data)
                    # data = split_seq_by_window_quality(data)
                    # downsample
                    # for k in self.augment_resolution_dic[subdataset_name]:
                    #     # downsample_data = np.array([np.mean(data[i * k: (i + 1) * k]) for i in range(len(data) // k)])  # downsample_by_averaging
                    #     downsample_data = data[::k]  # downsample_by_decimation
                    #     if k != 1:
                    #         downsample_dataset_list.append(downsample_data)
                   
                    # save original data
                    data, n_timepoint = self.__process_each_sequence__(data)
                   
                    if n_timepoint < 0:
                        continue
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
               
                # # process downsample data
                # for downsample_data in downsample_dataset_list:
                #     # save original data
                #     downsample_data, downsample_n_timepoint = self.__process_each_sequence__(downsample_data)
                #     if downsample_n_timepoint < 0:
                #         continue
                #     downsample_n_var = downsample_data.shape[1]
                #     self.data_list.append(downsample_data)

                #     downsample_n_window = downsample_n_timepoint * downsample_n_var
                #     self.n_window_list.append(downsample_n_window if len(
                #         self.n_window_list) == 0 else self.n_window_list[-1] + downsample_n_window)        
       
        print("Total number of windows in merged dataset: ", self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1
       
        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
           
        if self.is_pretraining:
            n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len - self.pred_len) // self.stride + 1
        else:
            n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len - self.horizon_lengths[-1]) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
       
        if self.is_pretraining:
            r_end = r_begin + self.label_len + self.pred_len
            seq_y = self.data_list[dataset_index][r_begin:r_end,
                                                c_begin:c_begin + 1]
            seq_y_mark = torch.zeros((seq_y.shape[0], 1))
        else:
            seq_y, seq_y_mark = [], []  
            for h in self.horizon_lengths:  
                r_end = r_begin + self.label_len + h  
                seq_y.append(self.data_list[dataset_index][r_begin:r_end, c_begin:c_begin + 1])  
                seq_y_mark.append(torch.zeros((r_end - r_begin, 1)))
           
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]
   
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CIDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='M', data_path='ETTh1.csv',
                 scale=True, timeenc=1, freq='h', percent=100,
                 task_name='long_term_forecast', is_pretraining=1, horizon_lengths=[1,96,192,336,720]):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.percent = percent
        self.timeenc = timeenc
        self.scale = scale
        self.task_name = task_name
        self.is_pretraining = is_pretraining
        self.root_path = root_path
        self.data_path = data_path
        self.dataset_name = self.data_path.split('.')[0]
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.dataset_name == 'ETTh' or self.dataset_name == 'ETTh1' or self.dataset_name == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.dataset_name == 'ETTm' or self.dataset_name == 'ETTm1' or self.dataset_name == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        elif self.dataset_name == 'pems03' or self.dataset_name == 'pems04' or self.dataset_name == 'pems07' \
                                            or self.dataset_name == 'pems08':
            data_len = len(df_raw)
            num_train = int(data_len * 0.6)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
       
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.timeenc == 0:
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            if isinstance(df_raw[df_raw.columns[0]][2], str):
                data_stamp = time_features(pd.to_datetime(pd.to_datetime(df_raw.date).values), freq='h')
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((len(df_raw), 4))
        else:
            raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]
       
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.set_type == 0 & self.percent != 100:
        #     self.sampled_timepoint = int(self.n_timepoint * self.percent / 100)
        #     max_start_idx = self.n_timepoint - self.seq_len - self.pred_len
        #     if max_start_idx < 0:
        #         raise ValueError("seq_len + pred_len exceeds available data range.")
        #     self.start_idxs = np.linspace(0, max_start_idx, self.sampled_timepoint, dtype=int)

    def __getitem__(self, index):
        # if self.set_type == 0 & self.percent != 100:
        #     c_begin = index // self.sampled_timepoint  # select variable
        #     s_begin = index % self.sampled_timepoint   # select start time
        #     s_begin = int(self.start_idxs[s_begin])
        # else:
        c_begin = index // self.n_timepoint  # select variable
        s_begin = index % self.n_timepoint   # select start time
           
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # assert s_end <= len(self.data_x), f"s_end out of range: {s_end} > {len(self.data_x)}"
        # assert r_end <= len(self.data_x), f"r_end out of range: {r_end} > {len(self.data_x)}"
        seq_x = self.data_x[s_begin:s_end, c_begin:c_begin + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end, c_begin:c_begin + 1]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # if self.set_type == 0 & self.percent != 100:
        #     return int(self.sampled_timepoint * self.n_var)
        # else:
        return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)