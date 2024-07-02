from lib.model.load_critic import load_critic
from render.render import render_multi
from parsedata import into_critic
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
critic_model = load_critic("critic/motioncritic_pre.pth", device)
example = torch.load("visexample.pth", map_location=device)
# calculate critic score
critic_scores = critic_model.module.batch_critic(into_critic(example['motion']))
print(f"critic scores are {critic_scores}")
# rendering
render_multi(example['motion'], device, example['comment'], example['path'])