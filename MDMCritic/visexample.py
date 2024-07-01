
from critic.load_critic import load_critic
from render.render import render_multi
from sample.critic_generate import into_critic
import torch

critic_model = load_critic("critic/exp8_final.pth", 'cpu')
example = torch.load("visexample.pth", map_location='cpu')
# get critic scores calculated. 
critic_scores = critic_model.module.batch_critic(into_critic(example['motion']))
print(f"critic scores are {critic_scores}")
# get motions rendered
render_multi(example['motion'], 'cpu', example['comment'], example['path'])