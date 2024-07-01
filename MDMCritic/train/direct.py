# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import sys
import os
import json

import torch.nn.functional
from utils.fixseed import fixseed
from utils.parser_util import train_args, tune_args
from utils import dist_util
from train.training_loop import TrainLoop
from train.tuning_loop import TuneLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from critic.critic import MotionCritic
import torch
from render.render import render_multi
from pubcode.AlignHP.MDMCritic.sample.critic_generate import outof_mdm, into_critic, outof_critic
from argparse import ArgumentParser
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def parse():
    parser = ArgumentParser()
    parser.add_argument("--device", default=None, type=int, help="Device id to use.")
    parser.add_argument("--exp_name", default="exp1-0", type=str, help="you are dead")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--scale", default=1e0, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help='batch size')
    parser.add_argument("--critic_model_path", default=os.path.join(PROJ_DIR, "critic/exp8_final.pth"), type=str, help="path to critic model")
    return parser.parse_args()




def main():
    args = parse()

    lr = args.lr
    scale = args.scale
    exp_name = args.exp_name
    batch_size = args.batch_size
    critic_model_path = args.critic_model_path
    
    device = "cpu"
    if args.device is not None:
        device = torch.device(args.device)

    critic_model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    critic_model = torch.nn.DataParallel(critic_model)
    checkpoint = torch.load(critic_model_path)
    critic_model.load_state_dict(checkpoint['model_state_dict'])
    critic_model = critic_model.to(device)


    # Load motion and ensure it requires gradient
    data = torch.load(os.path.join(PROJ_DIR, "save/direct/batch4.pth"))

    
    motion = into_critic(data['motion'])
    comments = data['comment']
    paths = data['path']


    motion = motion[:batch_size]
    comments = comments[:batch_size]
    paths = paths[:batch_size]


    motion = motion.to(device)
    motion.requires_grad_(True)
    optimizer = torch.optim.Adam([motion], lr=lr)

    # Zero the gradients
    optimizer.zero_grad()

    num_steps = 100  # Specify the number of training steps

    for step in range(num_steps):
        if step == 0:
            continue

        if step % 10 == 0:
            # Create directory if it doesn't exist
            save_dir = os.path.join(PROJ_DIR,f"save/direct/{exp_name}/step{step}")
            new_paths = []
            new_comments = []
            new_batch_critic = critic_model.module.batch_critic(motion)
            for i in range(motion.shape[0]):
                new_paths.append(os.path.join(save_dir, os.path.basename(paths[i])))
                new_comments.append(comments[i] + f",{exp_name} step {step}, new-critic {new_batch_critic[i].item()}")
            os.makedirs(save_dir, exist_ok=True)
        
            mstep = outof_critic(motion.detach().cpu())
            torch.save(mstep, os.path.join(save_dir, f"step{step}.pth"))
            # just do the render!
            render_multi(mstep, device=mstep.device, comments=new_comments, file_paths=new_paths, no_comment=False, isYellow=True)
            # render_multi(mstep, device=mstep.device, comments=new_comments, file_paths=new_paths, no_comment=True, isYellow=False)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        critic_loss = critic_model.module.clipped_critic(motion)

        critic_loss = -scale  * torch.sigmoid(12.0-critic_loss)

        
        # Backward pass
        critic_loss.backward()
        print(f"loss backward() complete. loss is {critic_loss.item()}")

        # Update motion
        optimizer.step()

        

    print(f"Final updated motion: {motion.shape}")

if __name__ == "__main__":
    main()