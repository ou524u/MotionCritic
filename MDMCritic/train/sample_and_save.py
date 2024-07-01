# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""
import sys
import os
import json
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

def main():
    args = tune_args()

    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    critic_device = dist_util.dev()

    print(f"Loading critic model from [{args.critic_model_path}]...")
    critic_model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    critic_model = torch.nn.DataParallel(critic_model)
    critic_model.to(critic_device)
    checkpoint = torch.load(args.critic_model_path)
    critic_model.load_state_dict(checkpoint['model_state_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TuneLoop(args, train_platform, model, diffusion, data, critic_model).sample_and_save_multi(step=args.resume_checkpoint[-7:-3])
    

    train_platform.close()

if __name__ == "__main__":
    main()
