from data_provider.data_factory import data_provider
from exp.exp_basic_chronos import Exp_Basic_Chronos
from utils.metrics import metric
import torch
import os
import time
import warnings
import numpy as np
from models.moirai.forecast import MoiraiForecast
from models.moirai_moe.forecast import MoiraiMoEForecast
import pandas as pd
from gluonts.dataset.common import ListDataset

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast_Chronos(Exp_Basic_Chronos):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Chronos, self).__init__(args)

    def _build_model(self):
        if self.args.model == "Moirai":
            model = MoiraiForecast(
                module=self.model_dict[self.args.model].from_pretrained(f"./models/moirai/moirai-1.0-R-large"),
                prediction_length=self.args.pred_len,
                context_length=self.args.seq_len,
                patch_size="auto",
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.args.model == "Moirai-MoE":
            model = MoiraiMoEForecast(
                module=self.model_dict[self.args.model].from_pretrained(f"./models/moirai/moirai-moe-1.0-R-small"),
                prediction_length=self.args.pred_len,
                context_length=self.args.seq_len,
                patch_size="auto",
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif self.args.model == "Timer":
            model = self.model_dict[self.args.model].from_pretrained('./models/timer-base-84m', trust_remote_code=True)
        elif self.args.model == "Chronos":
            model = self.model_dict[self.args.model].from_pretrained('./models/chronos/chronos-t5-large',  
                                                             device_map="cuda", torch_dtype=torch.bfloat16)
        elif self.args.model == "TimesFM":
            model = self.model_dict[self.args.model].TimesFm(
                hparams=self.model_dict[self.args.model].TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=self.args.batch_size,
                    horizon_len=self.args.pred_len,
                    num_layers=50,
                    context_len=self.args.seq_len,
                    model_dims=1280,
                    use_positional_embedding=False,
                ),
                checkpoint=self.model_dict[self.args.model].TimesFmCheckpoint(
                    path="./models/timesfm-1.0-200m")
            )
        else:
            print('error model!!!')
            return None
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)

        time_now = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):            
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float()
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
               
                if self.args.model == 'Moirai'or self.args.model == 'Moirai-MoE':
                    input_list = [
                        {
                            "target": batch_x[i].cpu().squeeze(-1).numpy(),  # shape: (context_length,)
                            "start": pd.Timestamp("2000-01-01")
                        }
                        for i in range(batch_x.shape[0])
                    ]
                    # GluonTS Dataset
                    gluonts_input = ListDataset(input_list, freq="1H")
                    # prediction
                    predictor = self.model.create_predictor(batch_size=32)
                    outputs = list(predictor.predict(gluonts_input))
                    outputs = [f.median for f in outputs]                # list of (prediction_length,)
                    outputs = torch.tensor(outputs).unsqueeze(-1)        # shape: (32, prediction_length, 1)
                elif self.args.model == 'Timer':
                    batch_x = batch_x.cpu().squeeze(-1)
                    outputs = self.model.generate(batch_x, max_new_tokens=self.args.pred_len)
                    outputs = outputs.unsqueeze(-1)
                elif self.args.model == 'Chronos':
                    batch_x = batch_x.cpu().squeeze(-1)
                    outputs = self.model.predict(batch_x, self.args.pred_len)
                    outputs = outputs.mean(dim=1).unsqueeze(-1)
                elif self.args.model == 'TimesFM':
                    outputs = self.model.forecast(batch_x.reshape(-1, batch_x.size(1)), freq=[0, 1, 2])
                else:
                    print("error model!")
                
                f_dim = -1 if self.args.features == 'MS' else 0      
                outputs = outputs[:, -self.args.pred_len:, :]
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

        print("Inference time: {}".format(time.time() - time_now))
       
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
