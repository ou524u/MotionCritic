import sys
import os
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)

os.environ['WANDB_DIR'] = PROJ_DIR + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = PROJ_DIR + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = PROJ_DIR + '/wandb/.config/'


from lib.model.critic import MotionCritic, MotionCritic_s

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from torch.backends import cudnn
import gc
import pytorch_warmup as warmup
import argparse

# this might be useful
torch.manual_seed(3407)

# Initialize argparse
parser = argparse.ArgumentParser(description='Your description here')

# Add command-line arguments
parser.add_argument('--gpu_indices', type=str, default="0,1",  # Change this to the GPU indices you want to use
                    help='Indices of the GPUs to use, separated by commas (default: "0,1")')



parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training (default: 32)')

parser.add_argument('--epoch', type=int, default=200,
                    help='Batch size for training (default: 32)')

parser.add_argument('--exp_name', type=str, default="exp7_2e-5_decay_seqsplit",
                    help='Experiment name for WandB')

parser.add_argument('--dataset', type=str, default="hfull_shuffle",
                    help='to determine use which train and val dataset')

# arguments
parser.add_argument('--save_checkpoint', action='store_true',
                    help='Whether to save model checkpoints during training (default: False)')

parser.add_argument('--load_model', type=str, default=None,
                    help='Path to the saved model checkpoint to load (default: None)')

parser.add_argument('--save_latest', action='store_true',
                    help='Whether to save model checkpoints during training (default: False)')

parser.add_argument('--lr_warmup', action='store_true',
                    help='Whether to save model checkpoints during training (default: False)')

parser.add_argument('--lr_decay', action='store_true',
                    help='Whether to save model checkpoints during training (default: False)')

parser.add_argument('--big_model', action='store_true',
                    help='Initiate a bigger model')

parser.add_argument('--origin_model', action='store_true',
                    help='use not sigmoid')

# Parse the arguments
args = parser.parse_args()

# Access the value of gpu_indices and convert it to a list of integers
gpu_indices = [int(idx) for idx in args.gpu_indices.split(',')]
gpu_number = len(gpu_indices)


batch_size = args.batch_size
lr = 2e-5 * (batch_size/32) # parallel changing

print(f"training on gpu {gpu_indices}, training starting with batchsize {batch_size}, lr {lr}")

load_model_path = args.load_model
exp_name = args.exp_name
num_epochs = args.epoch
save_checkpoint = args.save_checkpoint
save_latest = args.save_latest
lr_warmup = args.lr_warmup
lr_decay = args.lr_decay
big_model = args.big_model
origin_model = args.origin_model



if args.dataset == 'mdm_prompt':
    train_pth_name = "mlist_mdma_p00_to_p09_mdmu_p00_to_p37.pth"
    val_pth_name = "mlist_mdma_p10_to_p11_mdmu_p38_to_p39.pth"

elif args.dataset == 'mdm_seq':
    train_pth_name = "mlist_mdm_trainseq.pth"
    val_pth_name = "mlist_mdm_valseq.pth"

elif args.dataset == 'mdm_shuffle':
    train_pth_name = "mlist_mdm_trainshuffle.pth"
    val_pth_name = "mlist_mdm_valshuffle.pth"

elif args.dataset == 'flame_seq':
    train_pth_name = "mlist_flame_trainseq.pth"
    val_pth_name = "mlist_flame_valseq.pth"

elif args.dataset == 'flame_shuffle':
    train_pth_name = "mlist_flame_trainshuffle.pth"
    val_pth_name = "mlist_flame_valshuffle.pth"

elif args.dataset == 'hfull_seq':
    train_pth_name = "mlist_hfull_trainseq.pth"
    val_pth_name = "mlist_hfull_valseq.pth"

elif args.dataset == 'hfull_shuffle':
    train_pth_name = "mlist_hfull_trainshuffle.pth"
    val_pth_name = "mlist_hfull_valshuffle.pth"

elif args.dataset == 'full_shuffle':
    train_pth_name = "mlist_full_trainshuffle.pth"
    val_pth_name = "mlist_full_valshuffle.pth"

elif args.dataset == 'mdmfull_seq':
    train_pth_name = "mlist_mdmfull_trainseq.pth"
    val_pth_name = "mlist_mdmfull_valseq.pth"

elif args.dataset == 'mdmfull_shuffle':
    train_pth_name = "mlist_mdmfull_trainshuffle.pth"
    val_pth_name = "mlist_mdmfull_valshuffle.pth"
    
elif args.dataset == 'flamefull_seq':
    train_pth_name = "mlist_flamefull_trainseq.pth"
    val_pth_name = "mlist_flamefull_valseq.pth"

elif args.dataset == 'flamefull_shuffle':
    train_pth_name = "mlist_flamefull_trainshuffle.pth"
    val_pth_name = "mlist_flamefull_valshuffle.pth"


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_indices))
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import wandb

class motion_pair_dataset(Dataset):
    def __init__(self, motion_pair_list_name):
        self.data = torch.load(motion_pair_list_name)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def init_seeds(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
       cudnn.deterministic = True
       cudnn.benchmark = False
    else:  # faster, less reproducible
       cudnn.deterministic = False
       cudnn.benchmark = True


def loss_func(critic):
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = F.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc



wandb.init(project="preexp", name=exp_name, resume=False)





checkpoint_interval = 40  # Save checkpoint every 100 epochs

# Instantiate your dataset
train_pth = os.path.join(PROJ_DIR, 'datasets/'+ train_pth_name)
val_pth = os.path.join(PROJ_DIR, 'datasets/'+ val_pth_name)
train_motion_pairs = motion_pair_dataset(motion_pair_list_name=train_pth)
val_motion_pairs = motion_pair_dataset(motion_pair_list_name=val_pth)



# Instantiate DataLoader
train_loader = DataLoader(train_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_motion_pairs, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)

# Instantiate your model, loss function, and optimizer

if origin_model:
    if big_model:

        model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
        model = torch.nn.DataParallel(model)
        model.to(device)

    else:
        model = MotionCritic(depth=3, dim_feat=128, dim_rep=256, mlp_ratio=2)
        model = torch.nn.DataParallel(model)
        model.to(device)

else:
    if big_model:

        model = MotionCritic_s(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
        model = torch.nn.DataParallel(model)
        model.to(device)

    else:
        model = MotionCritic_s(depth=3, dim_feat=128, dim_rep=256, mlp_ratio=2)
        model = torch.nn.DataParallel(model)
        model.to(device)


criterion = loss_func  # Assuming your loss_func is already defined
# Create your optimizer


optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                             betas=(0.9, 0.999), weight_decay=1e-4)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

final_lr = 1e-5
initial_lr = 5e-4
warmup_type = "radam" 

# lr scheduling
# gamma = (final_lr / initial_lr) ** (1.0 / num_epochs)
gamma = 0.995
scheduler = ExponentialLR(optimizer, gamma)


if warmup_type == 'linear':
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
elif warmup_type == 'exponential':
        warmup_scheduler = warmup.UntunedExponentialWarmup(optimizer)
elif warmup_type == 'radam':
        warmup_scheduler = warmup.RAdamWarmup(optimizer)
elif warmup_type == 'none':
        warmup_scheduler = warmup.LinearWarmup(optimizer, 1)


# Load the model if load_model_path is provided
if load_model_path:
    # Load the checkpoint
    checkpoint = torch.load(load_model_path)
    # Load the model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    best_accuracy = checkpoint['best_accuracy']
    print(f"Model loaded from {load_model_path}, starting training from epoch {start_epoch}")
else:
    start_epoch = 0
    best_accuracy = 0
    print("No model loaded, starting training from scratch")

start_epoch = 0
best_accuracy = 0

# Continue training from the loaded checkpoint
for epoch in range(start_epoch, num_epochs):

    for step, batch_data in enumerate(train_loader):
        # Move batch data to GPU
        # Move each tensor in the dictionary to GPU
        model.train()

        batch_data = {key: value.cuda(device=device) for key, value in batch_data.items()}

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        critic = model(batch_data)

        # Compute the loss
        loss, _, acc = criterion(critic)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        if step % 40 == 0:
            # Log metrics to WandB
            wandb.log({"Loss": loss.item(), "Accuracy": acc.item()})

            # Optionally, print training metrics
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {acc.item()}')
            # Remove batch_data from GPU to save memory

        batch_data = {key: value.detach().cpu() for key, value in batch_data.items()}


    # evaluate the model on a epoch basis
    average_val_loss = 0.0
    average_val_acc = 0.0
    total_val_samples = 0

    if lr_decay:
        scheduler.step()


    if lr_warmup:
        with warmup_scheduler.dampening():
            scheduler.step()


    with torch.no_grad():
        model.eval()

        for val_batch_data in val_loader:
            val_batch_data = {key: value.cuda(device=device) for key, value in val_batch_data.items()}
            val_loss, _, val_acc = criterion(model(val_batch_data))

            val_batch_size = len(val_batch_data)
            average_val_acc += val_acc.item() * val_batch_size
            average_val_loss += val_loss.item() * val_batch_size
            total_val_samples += val_batch_size
            val_batch_data = {key: value.detach().cpu() for key, value in val_batch_data.items()}
            # del val_batch_data
            torch.cuda.empty_cache()

        # Calculate average loss and accuracy
        average_val_loss = average_val_loss / total_val_samples
        average_val_acc = average_val_acc / total_val_samples
        
        wandb.log({"val_Loss": average_val_loss, "val_Accuracy": average_val_acc, "lr": optimizer.param_groups[0]['lr']})

         # Optionally, print training metrics
        print(f'Epoch {epoch + 1}, val_Loss: {average_val_loss}, val_Accuracy: {average_val_acc}, lr: {optimizer.param_groups[0]["lr"]}')


    # Save the best model based on validation accuracy
    if average_val_acc > best_accuracy:
        best_accuracy = average_val_acc
        best_model_state = model.state_dict()
        best_checkpoint_path = f"{exp_name}_best_checkpoint.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'accuracy': acc.item(),
        }, best_checkpoint_path)
        print(f"Best model saved at {best_checkpoint_path}")

    if save_checkpoint:
        # Save checkpoint every k epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{exp_name}_checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_accuracy': best_accuracy,
                'accuracy': acc.item(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    if save_latest:
        # Save latest checkpoint
        checkpoint_path = f"{exp_name}_checkpoint_latest.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'best_accuracy': best_accuracy,
            'accuracy': acc.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


