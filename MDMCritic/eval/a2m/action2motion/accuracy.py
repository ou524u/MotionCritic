import torch

from pubcode.AlignHP.MDMCritic.sample.critic_generate import outof_mdm, into_critic

def calculate_accuracy(model, motion_loader, num_labels, classifier, device):
    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    with torch.no_grad():
        for batch in motion_loader:
            batch_prob = classifier(batch["output_xyz"], lengths=batch["lengths"])
            # print(f"batch_prob shape is {batch_prob.shape}") is torch.Size([64, 12])
            batch_pred = batch_prob.max(dim=1).indices
            for label, pred in zip(batch["y"], batch_pred):
                confusion[label][pred] += 1

    accuracy = torch.trace(confusion)/torch.sum(confusion)
    return accuracy.item(), confusion



def calculate_critic(model, critic_model, motion_loader, device):
    all_critic = []
    with torch.no_grad():
        for batch in motion_loader:
            # print(f"batch[output] shape is {batch['output'].shape}") is torch.Size([40, 25, 6, 60])
            batch_critic = into_critic(outof_mdm(batch["output"]))
            all_critic.append(batch_critic)

    all_critic = torch.cat(all_critic, dim=0)
    critic_loss = critic_model.module.clipped_critic(all_critic)
    print(f"in calculate_critic critic_loss is {critic_loss.item()}")
    return critic_loss.item()
