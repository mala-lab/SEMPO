from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from utils.metrics import metric
import torch
import torch.nn as nn
import os
import time
import re
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
       
        # multi-resolution scheduling
        H_hat = 0
        self.J = []
        while H_hat < args.pred_len:
            for j in reversed(args.horizon_lengths):
                if H_hat + j <= args.pred_len:
                    H_hat += j
                    self.J.append(j)
                    break
        self.idx = {i for i, h in enumerate(args.horizon_lengths) if h in self.J}

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args).to(self.device)
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # if self.args.use_weight_decay:
        #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
        #                              weight_decay=self.args.weight_decay)
        # else:
        #     model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
       
        optim_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        model_optim = torch.optim.AdamW(
            optim_groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=1e-8,
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, flag='train'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            if flag == 'train':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                   
                    if self.args.head_type == 'prediction':
                        batch_y = batch_y.float()
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    else:
                        batch_y = [y.float() for y in batch_y]
                        batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                       
                        dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                        dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]
                   
                    # encoder - decoder
                    if self.args.output_attention:
                        outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                       
                    f_dim = -1 if self.args.features == 'MS' else 0  
                    recons = recons[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]
                    recons = recons.detach().cpu()
                    batch_x = batch_x.detach().cpu()
                   
                    if self.args.head_type == 'prediction':        
                        outputs = torch.cat([outputs[j][:, -h:, f_dim:] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        pred = outputs.detach().cpu()
                        true = batch_y.detach().cpu()
                        loss = criterion(pred, true) + criterion(recons, batch_x)
                    else:
                        pre_loss = torch.tensor(0.)
                        for j, (o, y) in enumerate(zip(outputs, batch_y)):
                            o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                            y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                            o = o.detach().cpu()
                            y = y.detach().cpu()
                            pre_loss = pre_loss + criterion(o, y)
                        loss = pre_loss + criterion(recons, batch_x)
                   
                    loss = loss.detach().cpu()
                    total_loss.append(loss)
                    total_count.append(batch_x.shape[0])
        
                    torch.cuda.empty_cache()
                   
            elif flag == 'pretrain':
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                   
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
               
                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                       
                    # encoder - decoder
                    if self.args.output_attention:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0            
                    outputs = outputs[:, :, f_dim:]
                    batch_x = batch_x[:, :, f_dim:]

                    pred = outputs.detach().cpu()
                    true = batch_x.detach().cpu()
                   
                    loss = criterion(pred, true)
                    loss = loss.detach().cpu()
                    total_loss.append(loss)
                    total_count.append(batch_x.shape[0])
                   
                    torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
   
    def pretrain(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))
       
        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0
       
        for epoch in range(self.args.pretrain_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()
           
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
               
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
               
                # encoder - decoder
                if self.args.output_attention:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                   
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
               
                loss_val += loss
                count += 1
                   
                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()


                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
               
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='pretrain')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='pretrain')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def train(self, setting, train=0):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
               
        if train:
            print('loading model')
            original_setting = setting
            setting = re.sub(r'_' + self.args.data + '_', '_UTSD_', setting)
            setting = re.sub(r'_pl' + str(self.args.pred_len) + '_', '_pl96_', setting)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            setting = original_setting
           
            # for param in self.model.parameters(): param.requires_grad = False
            # trainable_modules = [
            #     self.model.module.pretrain_heads,
            # ]
            # for module in trainable_modules:
            #     for param in module.parameters(): param.requires_grad = True


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters() if param.requires_grad))
        scheduler = LargeScheduler(self.args, model_optim)
        global_step = 0

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0.).to(self.device)
            count = torch.tensor(0.).to(self.device)

            self.model.train()
            epoch_time = time.time()
           
            # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ref_x, ref_y) in enumerate(train_loader):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                             
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                if self.args.head_type == 'prediction':
                    batch_y = batch_y.float()
                    batch_y_mark = batch_y_mark.float().to(self.device) 

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                else:
                    batch_y = [y.float() for y in batch_y]
                    batch_y_mark = [y_mark.float().to(self.device) for y_mark in batch_y_mark]
                   
                    dec_inp = [torch.zeros_like(y[:, -h:, :]).float() for y, h in zip(batch_y, self.args.horizon_lengths)]
                    dec_inp = [torch.cat([y[:, :self.args.label_len, :], d], dim=1).float().to(self.device) for y, d in zip(batch_y, dec_inp)]                

                # encoder - decoder
                if self.args.output_attention:
                    outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                             
                f_dim = -1 if self.args.features == 'MS' else 0
                recons = recons[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]
                if self.args.head_type == 'prediction':
                    outputs = torch.cat([outputs[j][:, -h:, f_dim:] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + criterion(recons, batch_x)
                else:
                    pre_loss = torch.tensor(0.).to(self.device)
                    for j, (o, y) in enumerate(zip(outputs, batch_y)):
                        o = o[:, -self.args.horizon_lengths[j]:, f_dim:]
                        y = y[:, -self.args.horizon_lengths[j]:, f_dim:].to(self.device)
                        pre_loss = pre_loss + criterion(o, y)
                    loss = pre_loss + criterion(recons, batch_x)
                               
                loss_val += loss
                count += 1
                   
                if i % 100 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                scheduler.schedule_step(global_step)
                global_step += 1
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion, flag='train')
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='train')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            original_setting = setting
            if self.args.is_zeroshot:
                setting = re.sub(r'_CI_', '_UTSD_', setting)
                setting = re.sub(r'_pl' + str(self.args.pred_len) + '_', '_pl96_', setting)
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            setting = original_setting

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)       
               
                batch_y = batch_y.float()
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
               
                # encoder - decoder
                if self.args.output_attention:
                    outputs, recons, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, recons = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
               
                outputs = torch.cat([outputs[j][:, -h:, :] for j, h in enumerate(self.args.horizon_lengths) if j in self.idx], dim=1)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                   
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
       
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                   
                    if self.args.local_rank == 0:
                        if self.args.output_attention:
                            attn = attns[0].cpu().numpy()[0, 0, :, :]
                            attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
       

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        return
