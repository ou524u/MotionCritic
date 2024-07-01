# Aligning Motion Generation with Human Perceptions

This repository contains the PyTorch implementation of the paper "Aligning Motion Generation with Human Perceptions," submitted to NeurIPS 2024, D&B track.

## Quick Demo
MotionCritic is capable of scoring a single motion with just a few lines of code.
```bash
cd MDMCritic
```

```python
from critic.load_critic import load_critic
from sample.critic_generate import into_critic
import torch
critic_model = load_critic("critic/exp8_final.pth", 'cpu')
example = torch.load("criexample.pth", map_location='cpu')
# get critic scores calculated. 
critic_scores = critic_model.module.batch_critic(into_critic(example['motion']))
print(f"critic scores are {critic_scores}") # Critic score being 4.1297 in this case
```


https://github.com/ou524u/AlignHP/assets/92263178/edd9600a-5c72-4594-80b3-356d442736c9

Try scoring multiple motions with some more [code](MDMCritic/visexample.py) 
```bash
bash prepare/prepare_demo.sh
```
```python
from critic.load_critic import load_critic
from render.render import render_multi
from sample.critic_generate import into_critic
import torch

critic_model = load_critic("critic/exp8_final.pth", 'cpu')
example = torch.load("visexample.pth", map_location='cpu')
# get critic scores calculated. 
critic_scores = critic_model.module.batch_critic(into_critic(example['motion']))
print(f"critic scores are {critic_scores}")
# get motions rendered
render_multi(example['motion'], 'cpu', example['comment'], example['path'])
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
--reward_model_path ./reward/exp8_final.pth \
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