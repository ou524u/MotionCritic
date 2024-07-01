
import sys
import os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

import torch
import numpy as np
import json
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss


from sklearn.metrics import average_precision_score, brier_score_loss, accuracy_score
from lib.utils.rotation2xyz import Rotation2xyz

from scipy.stats import wilcoxon


evalmdm = False
evalflame = False


# evalflame = True
evalmdm = True

gt_humanact12 = [torch.load(os.path.join(PROJ_DIR, f'datasets/gt/motion-gt{i}.pth'))['motion'] for i in range(12)]
gt_uestc = [torch.load(os.path.join(PROJ_DIR, f'/datasets/gt/motion-gtuestc{i}.pth'))['motion'] for i in range(40)]

gt_humanact12xyz = []
gt_uestcxyz = []


gt_flame = [torch.load(os.path.join(PROJ_DIR, f'/datasets/gt/flame-gt.pth'))]
gt_flamexyz = []

def extract_number_from_filename(file_name):
    first_dash_index = file_name.find('-')
    second_dash_index = file_name.find('-', first_dash_index + 1)
    third_dash_index = file_name.find('-', second_dash_index + 1)
    if second_dash_index == -1 or third_dash_index == -1:
        return None
    number_str = file_name[second_dash_index + 1:third_dash_index]
    try:
        number = int(number_str)
        return number
    except ValueError:
        return None

def choose_gt_dataset_from_filename(file_name):
    if evalmdm:
        action_class = extract_number_from_filename(file_name)

        if file_name[3] == 'a':
            return gt_humanact12[action_class]
        elif file_name[3] == 'u':
            return gt_uestc[action_class]
        
    if evalflame:
        return gt_flame[0]
    

def choose_gtxyz_dataset_from_filename(file_name):
    if evalmdm:
        action_class = extract_number_from_filename(file_name)

        if file_name[3] == 'a':
            return gt_humanact12xyz[action_class]
        elif file_name[3] == 'u':
            return gt_uestcxyz[action_class]
    
    if evalflame:
        return gt_flamexyz[0]


def build_gt_xyz():
    device = 'cpu'
    rot2xyz = Rotation2xyz(device=device)
    if evalmdm:
        for gt in gt_humanact12:

            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2)
            gt_humanact12xyz.append(gt_xyz)
        
        for gt in gt_uestc:
            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2)
            gt_uestcxyz.append(gt_xyz)

    if evalflame:
        for gt in gt_flame:
            gt_xyz = rot2xyz(gt, mask=None,
                            pose_rep='rot6d', translation=True, glob=True,
                            jointstype='smpl', betas=None, beta=0, glob_rot=None,
                            vertstrans=True)
            # shape is [batch_size, 24, 3, 60]
            gt_xyz = gt_xyz.permute(0, 3, 1, 2)
            gt_flamexyz.append(gt_xyz)



def compute_AE(batch_vectors, ground_truth_vectors):
    # batch: [batch_size, 60, 1, 3]
    # gt: [gt_batch_size, 60, 1, 3]
    # print(f"tensor shapes: {batch_vectors.shape}, {ground_truth_vectors.shape}")
    l2_losses = torch.sqrt(torch.sum((batch_vectors.unsqueeze(1) - ground_truth_vectors.unsqueeze(0)) ** 2, dim=-1))
    mean_l2_losses = torch.mean(l2_losses, dim=(1, 2, 3))
    return mean_l2_losses
    

def compute_AVE(batch_vectors, ground_truth_vectors):
    # batch: [batch_size, 60, 1, 3]
    # gt: [gt_batch_size, 60, 1, 3]
    batch_variances = torch.var(batch_vectors, dim=1, keepdim=True)  # [batch_size, 1, 1, 3]
    gt_variances = torch.var(ground_truth_vectors, dim=1, keepdim=True)  # [gt_batch_size, 1, 1, 3]

    l2_losses = torch.sqrt(torch.sum((batch_variances.unsqueeze(1) - gt_variances.unsqueeze(0)) ** 2, dim=-1))
    mean_l2_losses = torch.mean(l2_losses, dim=(1, 2, 3))
    
    return mean_l2_losses


def compute_PFC(batch_vectors):
    # [batchsize, 60, 22, 3]
    delta_t = 1/30
    scores = []
    for batch_vec in  batch_vectors:
        root_v = (batch_vec[1:,0,:] - batch_vec[:-1,0,:])/delta_t
        root_a = (root_v[1:] - root_v[:-1])/delta_t
        root_a = np.linalg.norm(root_a, axis=-1)
        scaling = root_a.max()
        root_a /= scaling


        foot_idx = [7, 10, 8, 11]
        flat_dirs = [0, 2]
        feet = batch_vec[:,foot_idx]
        foot_v = np.linalg.norm(
                feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
            )  
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        # print(f"shape foot_means {foot_mins.shape}, shape root_a {root_a.shape}")
        foot_loss = (
                foot_mins[:, 0] * foot_mins[:, 1] * root_a
            )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        # names.append(pkl)
        # accelerations.append(foot_mins[:, 0].mean())
    scores_tensor = [torch.tensor(score) for score in scores]
    scores_tensor = torch.stack(scores_tensor)
    return scores_tensor


# read all motions from files
motion_location = os.path.join(PROJ_DIR, "datasets")

def rootloc_pairs_from_filename(file_name, choise):
    better_loc = []
    worse_loc = []
    npz_file = np.load(motion_location + file_name, allow_pickle=True)
    motion = npz_file['arr_0'].item()['motion'] # shape:[batch_size,25,6,60]
    motion = np.array(motion).transpose(0, 3, 1, 2) # shape:[batch_size,60,25,6]

    # print(f"motion shape {type(motion)} {motion.shape}")

    
    root_loc = torch.from_numpy(motion[:,:,24:25,0:3]) # shape:[batch_size,60,1,3], batch_size=1

    if evalmdm:
        if choise == 'A':
            better_loc = [root_loc[0], root_loc[0], root_loc[0]]
            worse_loc = [root_loc[1], root_loc[2], root_loc[3]]
        elif choise == 'B':
            better_loc = [root_loc[1], root_loc[1], root_loc[1]]
            worse_loc = [root_loc[0], root_loc[2], root_loc[3]]
        elif choise == 'C':
            better_loc = [root_loc[2], root_loc[2], root_loc[2]]
            worse_loc = [root_loc[0], root_loc[1], root_loc[3]]
        elif choise == 'D':
            better_loc = [root_loc[3], root_loc[3], root_loc[3]]
            worse_loc = [root_loc[0], root_loc[1], root_loc[2]]
    
    if evalflame:
        if choise == 'A':
            worse_loc = [root_loc[0], root_loc[0], root_loc[0]]
            better_loc = [root_loc[1], root_loc[2], root_loc[3]]
        elif choise == 'B':
            worse_loc = [root_loc[1], root_loc[1], root_loc[1]]
            better_loc = [root_loc[0], root_loc[2], root_loc[3]]
        elif choise == 'C':
            worse_loc = [root_loc[2], root_loc[2], root_loc[2]]
            better_loc = [root_loc[0], root_loc[1], root_loc[3]]
        elif choise == 'D':
            worse_loc = [root_loc[3], root_loc[3], root_loc[3]]
            better_loc = [root_loc[0], root_loc[1], root_loc[2]]

    
    better_loc = torch.stack(better_loc, dim=0)
    worse_loc = torch.stack(worse_loc, dim=0)

    return better_loc, worse_loc



def jointxyz_pairs_from_filename(file_name, choise):
    better_loc = []
    worse_loc = []
    npz_file = np.load(motion_location + file_name, allow_pickle=True)
    motion = npz_file['arr_0'].item()['motion'] # shape:[batch_size,25,6,60]

    motion = torch.from_numpy(np.array(motion))
    device = 'cpu'
    rot2xyz = Rotation2xyz(device=device)

    joints_xyz = rot2xyz(motion, mask=None,
                       pose_rep='rot6d', translation=True, glob=True,
                       jointstype='smpl', betas=None, beta=0, glob_rot=None,
                       vertstrans=True)
    # print(f"joints_xyz shape {joints_xyz.shape}") # is [batch_size, 24, 3, 60]
    joints_xyz = joints_xyz.permute(0, 3, 1, 2)

    if evalmdm:
        if choise == 'A':
            better_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            worse_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            better_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            worse_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            better_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            better_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            worse_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    if evalflame:
        if choise == 'A':
            worse_xyz = [joints_xyz[0], joints_xyz[0], joints_xyz[0]]
            better_xyz = [joints_xyz[1], joints_xyz[2], joints_xyz[3]]
        elif choise == 'B':
            worse_xyz = [joints_xyz[1], joints_xyz[1], joints_xyz[1]]
            better_xyz = [joints_xyz[0], joints_xyz[2], joints_xyz[3]]
        elif choise == 'C':
            worse_xyz = [joints_xyz[2], joints_xyz[2], joints_xyz[2]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[3]]
        elif choise == 'D':
            worse_xyz = [joints_xyz[3], joints_xyz[3], joints_xyz[3]]
            better_xyz = [joints_xyz[0], joints_xyz[1], joints_xyz[2]]

    
    better_xyz = torch.stack(better_xyz, dim=0)
    worse_xyz = torch.stack(worse_xyz, dim=0)

    return better_xyz, worse_xyz




def results_from_filename(file_name, choise, metric):
    

    gt = choose_gt_dataset_from_filename(file_name) # gt shape: [batch_size, 60, 25, 3], axis-angle
    if metric == 'Root AE':
        better, worse = rootloc_pairs_from_filename(file_name, choise)
        # gt_loc = gt[:,:,24:25,:] # changed
        gt_loc = gt.permute(0,3,1,2)[:,:,24:25,0:3]
        better_AE = compute_AE(better, gt_loc)
        worse_AE = compute_AE(worse, gt_loc)
        return better_AE, worse_AE
 

    elif metric == 'Root AVE':
        better, worse = rootloc_pairs_from_filename(file_name, choise)
        # gt_loc = gt[:,:,24:25,:]
        gt_loc = gt.permute(0,3,1,2)[:,:,24:25,0:3]
        better_AVE = compute_AVE(better, gt_loc)
        worse_AVE = compute_AVE(worse, gt_loc)
        return better_AVE, worse_AVE
    
    elif metric == 'Joint AE':
        better, worse = jointxyz_pairs_from_filename(file_name, choise)
        gt_xyz = choose_gtxyz_dataset_from_filename(file_name)
        better_AE = compute_AE(better, gt_xyz)
        worse_AE = compute_AE(worse, gt_xyz)
        return better_AE, worse_AE
    
    elif metric == 'Joint AVE':
        better, worse = jointxyz_pairs_from_filename(file_name, choise)
        gt_xyz = choose_gtxyz_dataset_from_filename(file_name)
        better_AVE = compute_AVE(better, gt_xyz)
        worse_AVE = compute_AVE(worse, gt_xyz)
        return better_AVE, worse_AVE
    
    elif metric == 'PFC':
        better, worse = jointxyz_pairs_from_filename(file_name, choise)
        better_PFC = compute_PFC(better)
        worse_PFC = compute_PFC(worse)
        return better_PFC, worse_PFC

        



def data_for_table(critic):
    # less is better!
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff < 0).float())

    # each critic has two scores, 0 for the better and 1 for worse.
    # we want that each pair's better score and worse score go softmax to become to probablities, sum=1
    # true labels are 0
    # we want to calculate acc, log_loss and auc-roc

    target = torch.zeros(critic.shape[0], dtype=torch.long)
    # Compute log_loss
    

    # Compute probabilities with softmax

    # print(f"{critic[:20,:]}")
    probs = F.softmax(critic, dim=1).numpy()
    probs = probs[:,[1,0]]
    target_np = target.numpy()
    log_loss_value = log_loss(y_true=target_np, y_pred=probs, labels=[0, 1])
    print(f"acc is {acc}, log_loss is {log_loss_value}")
    
    # some of the metrics
    differences = probs[:,0] - probs[:,1]
    var = np.var(differences)
    diffmean = np.mean(differences)
    bs = brier_score_loss(y_true=target_np, y_prob=probs[:,0])
    wilcoxon_statistic, p_value = wilcoxon(differences)
    # print(f"probs diff mean {diffmean}, var {var}, bs {bs}")
    # print(f" wilconxon {wilcoxon_statistic}, pval {p_value}")
    
    # compute roc-auc
    # score00 = probs[0][0]
    # score01 = probs[0][1]
    # probs[0][0] = score01
    # probs[0][1] = score00
    # if target_np[0] == 0:
    #     target_np[0] = 1
    # else:
    #     target_np[0] = 0
    # roc_auc_value = roc_auc_score(y_true=target_np, y_score=probs[:,0])

    # return acc, log_loss_value, roc_auc_value




def results_from_json(file_path, metric):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # cut here!
    # data = data[:100]

    better_score = []
    worse_score = []
    cnt = 0
    for file_name, choise in data.items():
        # if cnt > 14:
        #     continue

        if choise not in ['A', 'B', 'C', 'D']:
            continue
        better_pairscore, worse_pairscore = results_from_filename(file_name, choise, metric)
        better_score.append(better_pairscore)
        worse_score.append(worse_pairscore)
        cnt += 1


    print(f"scores' lengths are {len(better_score)}")
    better_score = torch.cat(better_score, dim=0)
    worse_score = torch.cat(worse_score, dim=0)
    
    both_score = torch.stack((better_score, worse_score), dim=1)

    # print(f"better_score shpae {better_score.shape}")
    # print(f"both_score shape {both_score}")

    return  data_for_table(both_score)






if evalmdm:
    file_path = os.path.join(PROJ_DIR, f'marked/mdm-fulleval.json')

else:
    file_path = os.path.join(PROJ_DIR, f'marked/flame-better/flame.json')

def results_from_metric(file_path, metric):
    print(f"building results from metric {metric}")
    results_from_json(file_path, metric)


if evalmdm:
    build_gt_xyz()
    print(f"gt-xyz data built.")
    results_from_metric(file_path, 'Root AVE')
    results_from_metric(file_path, 'Root AE')
    results_from_metric(file_path, 'Joint AVE')
    results_from_metric(file_path, 'Joint AE')
    results_from_metric(file_path, 'PFC')

if evalflame:
    build_gt_xyz()
    print(f"gt-xyz data built.")
    results_from_metric(file_path, 'Root AVE')
    results_from_metric(file_path, 'Root AE')
    results_from_metric(file_path, 'Joint AVE')
    results_from_metric(file_path, 'Joint AE')
    results_from_metric(file_path, 'PFC')
