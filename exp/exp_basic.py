from models import SEMPO, SEMPO_CL
from models.moirai.module import MoiraiModule
from models.moirai_moe.module import MoiraiMoEModule
from transformers import AutoModelForCausalLM
from chronos import ChronosPipeline
import timesfm
import os
import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SEMPO': SEMPO,
            'SEMPO_CL': SEMPO_CL,
            'Moirai': MoiraiModule,
            'Moirai-MoE': MoiraiMoEModule,
            'Timer': AutoModelForCausalLM,
            'Chronos': ChronosPipeline,
            'TimesFM': timesfm,
        }
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            self.model = self._build_model()        
        else:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
   
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
