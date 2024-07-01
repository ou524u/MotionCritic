import sys
import os
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_DIR)

from lib.model.critic import MotionCritic, MotionCritic_s

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.backends import cudnn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.metrics import average_precision_score, brier_score_loss, accuracy_score
import gc
import pytorch_warmup as warmup
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

from scipy.stats import wilcoxon

val_pth_name = "mlist_mdmfull_valshuffle.pth"
# val_pth_name = "mlist_flame_better.pth"


device = torch.device('cuda:0')

model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
model = torch.nn.DataParallel(model)
model.to(device)

# load pretrained model
checkpoint = torch.load(os.path.join(PROJ_DIR,f'pretrained/exp8_final.pth'), map_location=device)
    # Load the model and optimizer
model.load_state_dict(checkpoint['model_state_dict'])


def metric_func(critic):
    # critic's shape is [batch_size,2]
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    # each critic has two scores, 0 for the better and 1 for worse.
    # we want that each pair's better score and worse score go softmax to become to probablities, sum=1
    # true labels are 0
    # we want to calculate acc, log_loss and auc-roc

    # Compute probabilities with softmax
    probs = F.softmax(critic, dim=1).cpu().detach().numpy()
    target_np = target.cpu().numpy()
    # Compute log_loss
    log_loss_value = log_loss(y_true=target_np, y_pred=probs, labels=[0, 1])
    print(f"acc is {acc}, log_loss is {log_loss_value}")

    # some of the other metrics
    differences = probs[:,0] - probs[:,1]
    var = np.var(differences)
    diffmean = np.mean(differences)
    bs = brier_score_loss(y_true=target_np, y_prob=probs[:,0])
    wilcoxon_statistic, p_value = wilcoxon(differences)    
    # print(f"probs diff mean {diffmean}, var {var}, bs {bs}")
    # print(f" wilconxon {wilcoxon_statistic}, pval {p_value}")


    # score00 = probs[0][0]
    # score01 = probs[0][1]
    # probs[0][0] = score01
    # probs[0][1] = score00
    # target_np[0] = 1
    # score10 = probs[1][0]
    # score11 = probs[1][1] 
    # probs[1][0] = score11
    # probs[1][1] = score10
    # target_np[1] = 1
    # roc_auc_value = roc_auc_score(y_true=target_np, y_score=probs[:,0])

    # return acc.item(), log_loss_value, roc_auc_value




class motion_pair_dataset(Dataset):
    def __init__(self, motion_pair_list_name):
        self.data = torch.load(motion_pair_list_name)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

val_pth = os.path.join(PROJ_DIR, 'datasets/'+ val_pth_name)
val_motion_pairs = motion_pair_dataset(motion_pair_list_name=val_pth)


val_loader = DataLoader(val_motion_pairs, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)


all_scores = []
for val_batch_data in val_loader:
    # print(f"model is on {model.module.device}")
    val_batch_data = {key: value.to(device=device) for key, value in val_batch_data.items()}
    scores = model.module.forward(val_batch_data)
    # print(f"scores' shape is {scores.shape}")
    all_scores.append(scores.detach().cpu())
    scores.detach().cpu()

all_scores = torch.cat(all_scores, dim=0)
print(f"all_scores' shape {all_scores.shape}")


metric_func(all_scores)



