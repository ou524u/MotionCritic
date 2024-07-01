import torch
def loss_func(critic):
    
    target = torch.zeros(critic.shape[0], dtype=torch.long).to(critic.device)
    loss_list = torch.nn.functional.cross_entropy(critic, target, reduction='none')
    loss = torch.mean(loss_list)
    
    critic_diff = critic[:, 0] - critic[:, 1]
    acc = torch.mean((critic_diff > 0).clone().detach().float())
    
    return loss, loss_list, acc