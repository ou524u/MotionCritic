# Fine-Tuning Documentation

## Overview

This document provides an overview of the scripts available for various purposes in the `MDMCritic/train` directory and explains the tuning options available for fine-tuning the critic model.

## Available Scripts

The `MDMCritic/train` directory contains the following scripts:

- **direct.py**: Directly back-propagate onto a batch of motions to improve their critic scores.
- **eval_fid.py**: Compute FID (Frechet Inception Distance) between certain batches of motions.
- **save_gt.py**: Get ground-truth motions from the HumanAct12 test split of vanilla MDM and save the batch.
- **sample_and_save.py**: Generate samples from the HumanAct12 test split of vanilla MDM and save the batch.
- **test_mdm.py**: Generate training lines and track evaluation metrics split by pre-split critic batches, from high to low.
- **train_mdm.py**: Perform vanilla training of the model.
- **tune_mdm.py**: Perform critic-supervised tuning of the model.

## Tuning Options

Below are the available tuning options for the critic model. These options can be set using the `add_tuning_options` function.

### Tuning Option Descriptions

- **--critic_model_path**: Path to the critic model. Default is `./critic/motioncritic_pre.pth`.
- **--ddim_sampling**: Use ReFL sampling with DDIM sampling. Enabled with the `--ddim_sampling` flag.
- **--render_video**: Render video during training. Enabled with the `--render_video` flag.
- **--sample_when_eval**: Sample when evaluating. Enabled with the `--sample_when_eval` flag.
- **--render_batch_size**: Batch size for evaluation rendering. Default is `1`.
- **--random_critic_loss**: Add random loss to the critic loss. Enabled with the `--random_critic_loss` flag.
- **--no_critic_loss**: Disable critic loss. Enabled with the `--no_critic_loss` flag.
- **--use_kl_loss**: Use KL (Kullback-Leibler) loss. Enabled with the `--use_kl_loss` flag.
- **--relu_loss**: Use ReLU loss. Enabled with the `--relu_loss` flag.
- **--render_gt**: Render ground truth during training. Enabled with the `--render_gt` flag.
- **--critic_scale**: Scale factor for critic loss. Default is `0.1`.
- **--kl_scale**: Scale factor for KL loss. Default is `0`.
- **--denoise_lower**: Lower bound for denoising. Default is `700`.
- **--denoise_upper**: Upper bound for denoising. Default is `900`.
- **--wandb**: Experiment name for WandB (Weights and Biases). Default is `None`.
  
## Usage
To run the critic-supervised tuning with the specified options, you can use the following command:

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


This command sets up and runs the critic-supervised tuning with the provided parameters and logs the results to WandB under the experiment name "my_experiment".