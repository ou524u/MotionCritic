# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
import utils.rotation_conversions as geometry_u


from critic.critic import MotionCritic
from render.render import render_multi
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def outof_mdm(generated_raw):
    # print(f'generated_raw shape is {generated_raw.shape}')
    root_loc = generated_raw[:,-1:,:,:]
    root_loc[:,:,1,:] *= -1
    root_loc[:,:,2,:] *= -1
    rot6d_motion = generated_raw[:,:-1,:,:].permute(0,3,1,2)
    matrix_m = geometry_u.rotation_6d_to_matrix(rot6d_motion)


    rot_m = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],dtype=torch.float, device=generated_raw.device)
    matrix_m[:,:,0,:,:] = torch.matmul(rot_m, matrix_m[:,:,0,:,:])
        
    rot6d_rotated = geometry_u.matrix_to_rotation_6d(matrix_m).permute(0,2,3,1)
    corr_m = torch.cat([rot6d_rotated, root_loc], dim=1)
    return corr_m


def into_critic(generated_motion):
    # generated_raw shape is torch.Size([1, 25, 6, 60])
    root_loc = generated_motion[:,-1:,:3,:].permute(0,3,1,2)
    rot6d_motion = generated_motion[:,:-1,:,:].permute(0,3,1,2)
    axis_angle = geometry_u.matrix_to_axis_angle(geometry_u.rotation_6d_to_matrix(rot6d_motion))
    # axis_angle torch.Size([1, 60, 24, 3])
    # print(f'axis_angle {axis_angle.shape}, root_loc {root_loc.shape}')
    critic_m = torch.cat([axis_angle, root_loc], dim=-2)
    return critic_m

def outof_critic(critic_m):
    # critic_m shape is torch.Size([1, 60, 25, 3])
    print(f"from critic shape {critic_m.shape}")
    axis_angle = critic_m[:, :, :-1, :]
    root_loc = critic_m[:, :, -1:, :]
    # [1,60,24,3] -> [1,60,24,3,3] -> [1,60,24,6]
    # Convert axis_angle back to rotation matrix and then to 6D rotation representation
    rot_matrices = geometry_u.axis_angle_to_matrix(axis_angle)
    rot6d_motion = geometry_u.matrix_to_rotation_6d(rot_matrices)
    
    # Permute dimensions back to original format
    rot6d_motion = rot6d_motion.permute(0, 2, 3, 1) # [1,24, 6, 60]
    root_loc = root_loc.permute(0, 2, 3, 1) # [1,24, 3,60], need to add padding to make it [1,24,6,60]

    # add zero padding to root_loc here please.
    root_loc = torch.cat([root_loc, torch.zeros_like(root_loc)], dim=-2)
    
    # Concatenate rot6d_motion and root_loc
    print(f"shapes rot6d {rot6d_motion.shape}, root {root_loc.shape}")
    generated_motion = torch.cat([rot6d_motion, root_loc], dim=1) # get [1,25,6,60] 
    return generated_motion

# Example usage:
# critic_m = some_critic_m_tensor
# generated_motion = from_critic(critic_m)

def main():
    args = generate_args()
    
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'mya2m_samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)



    # add CFG scale to batch
    if args.guidance_param != 1:
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop

    dump_list = [80, 90, 95, 98, 100]
    # dump_list = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
    # dump_list = [940, 950, 960, 970, 975, 980, 985, 990, 995, 1000]

    
    generated_list = sample_fn(
        model,
        # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
        (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=dump_list, # dump_steps set to true, returning a list, with each step.
        # dump_steps=None,
        noise=None,
        const_noise=False,
    )


    sample_list = []

    for i, sample in enumerate(generated_list):
        sample_list.append(generated_list[i].cpu().clone().detach())


    # real_device = args.device
    real_device = "cpu"



    model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    model = torch.nn.DataParallel(model)


    model.to(real_device)
    checkpoint = torch.load(os.path.join(PROJ_DIR, f"critic/exp8_final.pth"))
    # Load the model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])

    critics = []
    for i, sampled_motion in enumerate(sample_list):
        sample_list[i] = outof_mdm(sampled_motion)
        critic_m = into_critic(sample_list[i])
        critic_m = critic_m.to(real_device)
        critic_val = model.module.clipped_critic(critic_m)
        critic_val = critic_val.cpu().item()
        critics.append(critic_val)


    print(f'critics over steps {critics}')



    sample_list = torch.cat(sample_list, dim=0)
    comment_list = []
    path_list = []

    output_path = os.path.join(PROJ_DIR, f"save/rendered/exp0")
    for i in range(sample_list.shape[0]):
        comment_list.append(f"generate, prompt: 'drink', denoise step: {dump_list[i]}, critic: {critics[i]:.2f}")
        path_list.append(os.path.join(output_path, f"denoise-{dump_list[i]}.mp4"))

    render_multi(sample_list, sample_list.device, comment_list, path_list)
    print(f'sample_list shape is: {sample_list.shape}')





def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              hml_mode='text_only')
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
