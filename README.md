<div align="center">
  <h2><b>SEMPO: Lightweight Foundation Models for Time Series Forecasting </b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/mala-lab/SEMPO?color=green)
![](https://img.shields.io/github/stars/mala-lab/SEMPO?color=yellow)
![](https://img.shields.io/github/forks/mala-lab/SEMPO?color=lightblue)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## 🎉 What's New

* Oct 2025: Release of SEMPO library, along with SEMPO preprint now on [arXiv](https://arxiv.org/pdf/2409.16040).
  
* Sep 2025: The [SEMPO Paper](https://arxiv.org/abs/2402.02592) has been accepted to NeurIPS 2025 as a Poster presentation!

## 📋 Introduction

SEMPO is a novel time series foundation model with significantly reduced model size and pre-training scale, yet demonstrating superior generalization ability on diverse downstream forecasting tasks.

<p align="center">
    <img src="figures/framework.png" alt="" align="center" width="700px" />
</p>

## 📚 Pre-training Data


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  SEMPO is implemented in Pytorch 2.1.2+cu118 with Python 3.10. All the experiments are run on 4 A6000-48G GPUs.

## Pre-training

To pretrain the model(s) in the paper, run this command:

```pre-training
bash ./scripts/time_series_forecasting/pretrain/sempo_utsd.sh
```

>📋  This command supports both single-GPU and multi-GPU execution on a single node. For convenience, we provide a single-GPU pretrained model in the folder ./checkpoints/.

## Fine-tuning

To fine-tune the model(s) in the paper, run this command:

```fine-tuning
bash ./scripts/time_series_forecasting/few_shot/sempo_ETTh1.sh
```

>📋  1. Put downstream datasets under the folder ./dataset/. 2. Put the checkpoint under the folder ./checkpoints/. 3. Fine-tune the model. We provide the fine-tuning examples under the folder ./scripts/time_series_forecasting/Long_term. We set --is_pretraining 0, --is_training 1, and --is_zeroshot 0, with two configurations of 5% and 10%.

## Evaluation

To evaluate the model(s) in the paper, run this command:

```eval
bash ./scripts/time_series_forecasting/zero_shot/sempo_weather.sh
```

>📋  Following the 1 and 2 steps as the aforementioned fine-tuning process. 3. Evaluate the model. We provide the evaluation examples under the folder ./scripts/time_series_forecasting/Long_term. We set --is_pretraining 0, --is_training 0, and --is_zeroshot 1.


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 



