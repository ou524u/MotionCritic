# Aligning Motion Generation with Human Perceptions (ICLR 2025)

This repository contains the PyTorch implementation of the paper "Aligning Motion Generation with Human Perceptions". 
<!-- submitted to NeurIPS 2024, D&B track. -->


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![arXiv](https://img.shields.io/badge/arXiv-2407.02272-b31b1b.svg)](https://arxiv.org/abs/2407.02272) <a href="https://motioncritic.github.io/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> <a href="https://youtu.be/sfFFWTpQcEQ"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"> </a>

![framework](https://github.com/ou524u/MotionCritic/assets/92263178/215232a3-6499-404a-9475-a877c63e3dd7)

## Quick Demo
MotionCritic is capable of scoring a single motion with just a few lines of code.
```bash
bash prepare/prepare_smpl.sh
```

```python
from lib.model.load_critic import load_critic
from parsedata import into_critic
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model = load_critic("critic/motioncritic_pre.pth", device)
example = torch.load("criexample.pth", map_location=device)
example_motion = example['motion'] # [bs, 25, 6, frame], rot6d with 24 SMPL joints and 1 XYZ root location
# motion pre-processing
preprocessed_motion = into_critic(example['motion']) # [bs, frame, 25, 3], axis-angle with 24 SMPL joints and 1 XYZ root location
# critic score
critic_scores = critic_model.module.batch_critic(preprocessed_motion)
print(f"critic scores are {critic_scores}") # Critic score being 4.1297 in this case
```


https://github.com/user-attachments/assets/47db045a-85d8-44ac-9aa1-82adbdeb9393



Try scoring multiple motions with some more [code](MotionCritic/visexample.py) 
```bash
bash prepare/prepare_demo.sh
```
```python
from lib.model.load_critic import load_critic
from render.render import render_multi
from parsedata import into_critic
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model = load_critic("critic/motioncritic_pre.pth", device)
example = torch.load("visexample.pth", map_location=device)
# calculate critic score
critic_scores = critic_model.module.batch_critic(into_critic(example['motion']))
print(f"critic scores are {critic_scores}")
# rendering
render_multi(example['motion'], device, example['comment'], example['path'])
```



https://github.com/user-attachments/assets/854d4f24-cc39-4e7e-95d9-336f081c64e3




## Getting Started

### Setup the Environment

```bash
conda env create -f environment.yml
conda activate mocritic
```

### Task Documentation

- [File Structure](docs/filestructure.md)
- [Annotations](docs/annotation.md)
- [Motion files](docs/motion.md)
- [Dataset](docs/dataset.md)
- [Fine-tuning](docs/finetuning.md)

### Dataset & Pretrained Model

Download the pre-processed datasets and pretrained models:

```bash
bash prepare/prepare_dataset.sh  # Download pre-processed datasets
bash prepare/prepare_pretrained.sh  # Download pretrained models
```

Alternatively, you can manually download the files from the following links:
- Pre-processed datasets: [Google Drive Link](https://drive.google.com/file/d/1H5MAPBIAygGV5HSa2yIftWDdGq4fPEXB/view?usp=drive_link)
- Pretrained MotionCritic model: [Google Drive Link](https://drive.google.com/file/d/1vifu1vktjCWDpyPpzGPugzHNalhsaMpq/view?usp=drive_link)


### Build Your Own Dataset (Optional)

To build your own dataset from the original motion files and annotation results:

```bash
bash prepare/prepare_fullannotation.sh
bash prepare/prepare_fullmotion.sh
```

Manual downloads are available here:
- Full annotation results: [Google Drive Link](https://drive.google.com/file/d/1TpZ0nVvx2c84rYGmHsdLgNbu8gBwLGkA/view?usp=sharing)
- Complete motion .npz files: [Google Drive Link](https://drive.google.com/file/d/1oM9B1InRkEpKu6-Y5sJ9Z-7DY7hemEpN/view?usp=drive_link)

After pre-processing the complete data, build your dataset with:

```bash
cd MotionCritic
python parsedata.py
```


## Evaluating the Critic Model

Reproduce the results from the paper by running:

```bash
cd MotionCritic/metric
python metrics.py
python critic_score.py
```


## Train Your Critic Model

Train your own critic model with the following command:

```bash
cd MotionCritic
python train.py --gpu_indices 0 --exp_name my_experiment --dataset mdmfull_shuffle --save_latest --lr_decay --big_model
```

## Critic Model Supervised Fine-Tuning

First, prepare the MDM baseline:

```bash
bash prepare/prepare_MDM_dataset.sh
bash prepare/prepare_MDM_pretrained.sh
```

If you encounter any issues, refer to the [MDM baseline setup](https://github.com/GuyTevet/motion-diffusion-model).

Next, start MotionCritic-supervised fine-tuning:

```bash
cd MDMCritic

python -m train.tune_mdm \
--dataset humanact12 --cond_mask_prob 0 --lambda_rcxyz 1 --lambda_vel 1 --lambda_fc 1 \
--resume_checkpoint ./save/humanact12/model000350000.pt \
--reward_model_path ./reward/motioncritic_pre.pth \
--device 0 \
--num_steps 1200 \
--save_interval 100 \
--reward_scale 1e-4 --kl_scale 5e-2 --random_reward_loss \
--ddim_sampling \
--eval_during_training \
--sample_when_eval \
--batch_size 64 --lr 1e-5 \
--denoise_lower 700 --denoise_upper 900 \
--use_kl_loss \
--save_dir save/finetuned/my_experiment \
--wandb my_experiment
```

Additional Python scripts for various fine-tuning purposes can be found in `MDMCritic/train`, detailed in the [fine-tuning documentation](docs/finetuning.md).


## Citation
If you find our work useful for your project, please consider citing the paper:
```bibtex
@inproceedings{motionpercept2025,
    title={Aligning Motion Generation with Human Perceptions},
    author={Wang, Haoru and Zhu, Wentao and Miao, Luyi and Xu, Yishu and Gao, Feng and Tian, Qi and Wang, Yizhou},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025},
    url={https://arxiv.org/pdf/2407.02272}
}
```

## Acknowledgement

If you use MotionPercept and MotionCritic in your work, please also cite the original datasets and methods on which our work is based.

MDM:

```bibtex
@inproceedings{
  tevet2023human,
  title={Human Motion Diffusion Model},
  author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023}
}
```

HumanAct12:

```bibtex
@inproceedings{guo2020action2motion,
  title={Action2motion: Conditioned generation of 3d human motions},
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Zou, Shihao and Sun, Qingyao and Deng, Annan and Gong, Minglun and Cheng, Li},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2021--2029},
  year={2020}
}
```

FLAME:

```bibtex
@inproceedings{kim2023flame,
  title={Flame: Free-form language-based motion synthesis \& editing},
  author={Kim, Jihoon and Kim, Jiseob and Choi, Sungjoon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={7},
  pages={8255--8263},
  year={2023}
}
```

UESTC:

```bibtex
@inproceedings{ji2018large,
  title={A large-scale RGB-D database for arbitrary-view human action recognition},
  author={Ji, Yanli and Xu, Feixiang and Yang, Yang and Shen, Fumin and Shen, Heng Tao and Zheng, Wei-Shi},
  booktitle={Proceedings of the 26th ACM international Conference on Multimedia},
  pages={1510--1518},
  year={2018}
}
```

DSTFormer:

```bibtex
@inproceedings{zhu2023motionbert,
  title={Motionbert: A unified perspective on learning human motion representations},
  author={Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={15085--15099},
  year={2023}
}
```

SMPL:

```bibtex
@incollection{loper2023smpl,
  title={SMPL: A skinned multi-person linear model},
  author={Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J},
  booktitle={Seminal Graphics Papers: Pushing the Boundaries, Volume 2},
  pages={851--866},
  year={2023}
}
```

We also recommend exploring other motion metrics, including [PoseNDF](https://arxiv.org/abs/2207.13807), [NPSS](https://arxiv.org/abs/1809.03036), [NDMS](http://gall.cv-uni-bonn.de/download/jgall_forecastintention_3dv21.pdf), [MoBERT](https://arxiv.org/abs/2309.10248), and [PFC](https://arxiv.org/abs/2211.10658). You can also check out a [survey](https://arxiv.org/abs/2307.10894) of different motion generation metrics, datasets, and approaches.



