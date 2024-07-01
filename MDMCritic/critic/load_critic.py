from critic.critic import MotionCritic
import torch

def load_critic(critic_model_path, device):

    critic_model = MotionCritic(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
    critic_model = torch.nn.DataParallel(critic_model)
    checkpoint = torch.load(critic_model_path)
    critic_model.load_state_dict(checkpoint['model_state_dict'])
    critic_model = critic_model.to(device)
    return critic_model