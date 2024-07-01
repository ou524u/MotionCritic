import numpy as np
import lib.utils.rotation_conversions as geometry_u
import torch
import os
import sys
import json


import random

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

def into_critic(generated_motion):
    # generated_raw shape is torch.Size([1, 25, 6, 60])
    root_loc = generated_motion[..., -1:, :3, :].permute(-4, -1, -3, -2)
    rot6d_motion = generated_motion[..., :-1, :, :].permute(-4, -1, -3, -2)
    axis_angle = geometry_u.matrix_to_axis_angle(geometry_u.rotation_6d_to_matrix(rot6d_motion))
    # axis_angle torch.Size([1, 60, 24, 3])
    # print(f'axis_angle {axis_angle.shape}, root_loc {root_loc.shape}')
    critic_m = torch.cat([axis_angle, root_loc], dim=-2)
    return critic_m

def putpair(pair_list, file_name, choise='B', type='mdm'):
    npz_file = np.load(file_name, allow_pickle=True)

    motion_list = npz_file['arr_0'].item()['motion']
    # prompt_list = npz_file['arr_0'].item()['prompt']

    processed_motion_list = []
    for motion in motion_list:
        motion = np.transpose(motion, (2,0,1))
        processed_motion_list.append(into_critic(motion))

    if type == 'mdm':
        if choise == 'A':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'B':
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'C':
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[3]})
            return

        elif choise == 'D':
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[2]})
            return
        
    else:

        invalid_count = 0
        if choise == 'A':
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[0]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[0]})
            return

        elif choise == 'B':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[1]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[1]})
            return

        elif choise == 'C':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[2]})
            pair_list.append({'motion_better':processed_motion_list[3], 'motion_worse': processed_motion_list[2]})
            return

        elif choise == 'D':
            pair_list.append({'motion_better':processed_motion_list[0], 'motion_worse': processed_motion_list[3]})
            pair_list.append({'motion_better':processed_motion_list[1], 'motion_worse': processed_motion_list[3]})
            pair_list.append({'motion_better':processed_motion_list[2], 'motion_worse': processed_motion_list[3]})
            return
        

        else:
            invalid_count += 1
            if invalid_count % 20 == 0:
                print(f"invalid count reach: {invalid_count}")


    
            

def load_addfromfile(file_path, result_dict):
    with open(file_path, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        if key not in result_dict:
            result_dict[key] = value
    return result_dict


def load_addfromfolder(folder_path, result_dict):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            result_dict = load_addfromfile(file_path, result_dict)
    return result_dict

def load_shuffle(result_dict, seed=42):
    random.seed(42) 
    # Shuffle the keys in result_dict
    keys = list(result_dict.keys())
    random.shuffle(keys)

    # Create a new dictionary with shuffled keys
    shuffled_result_dict = {key: result_dict[key] for key in keys}
    return shuffled_result_dict


def put_fromdict(result_dict, pair_list, mode='full'):
    motion_dir = os.path.join(PROJ_DIR, 'data')
    
    invalid_cnt = 0
    for i, (file_name, choise) in enumerate(result_dict.items()):
        if mode == 'eval':
            if i%9 != 8:
                continue
        elif mode == 'train':
            if i%9 == 8:
                continue
        putpair(pair_list, os.path.join(motion_dir, file_name), choise, type=file_name[:3])
        if choise not in ['A', 'B', 'C', 'D']:
            invalid_cnt += 1

    print(f"this round, len is {len(result_dict.items())}, in which invalid {invalid_cnt}")
    


# for select_i in range(12):
#     result_dict = {}
#     result_dict = load_addfromfolder(f'marked/mdma/{select_i:02d}', result_dict)
#     result_dict = load_addfromfolder(f'marked/mdma-added/{select_i:02d}', result_dict)
#     pair_list = []
#     # result_dict = load_shuffle(result_dict)
#     mode = 'full' # chooing from full, train, val
#     put_fromdict(result_dict, pair_list, mode=mode)
#     pth_name = f'datasets/humanact12_{select_i:02d}-{mode}.pth'
#     torch.save(pair_list, pth_name)
#     print(f"saving .pth at {pth_name}")

