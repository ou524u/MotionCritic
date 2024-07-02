# Aligning Motion Generation with Human Perceptions

This repository contains the PyTorch implementation of the paper "Aligning Motion Generation with Human Perceptions". 
<!-- submitted to NeurIPS 2024, D&B track. -->


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"> </a> [![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2210.06551) <a href="https://motioncritic.github.io/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"> </a> <a href="https://youtu.be/2K_fwIrrdII"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"> </a>

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


https://github.com/ou524u/AlignHP/assets/92263178/edd9600a-5c72-4594-80b3-356d442736c9

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

https://github.com/ou524u/AlignHP/assets/92263178/a11fa74d-43a4-4dff-a755-c0c9fe00ccfe


## Getting Started

### Setup the Environment

```bash
conda env create -f environment.yml
conda activate mocritic
```

### Task Documentation

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
@article{motionpercept2024,
    title={Aligning Motion Generation with Human Perceptions},
    author={Wang, Haoru and Zhu, Wentao and Miao, Luyi and Xu, Yishu and Gao, Feng and Tian, Qi and Wang, Yizhou},
    year={2024}
}
```