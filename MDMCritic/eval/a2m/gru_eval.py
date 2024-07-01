import copy
import os

import numpy as np
from tqdm import tqdm
import torch
import functools
from torch.utils.data import DataLoader

from utils.fixseed import fixseed
from data_loaders.tensors import collate
from eval.a2m.action2motion.evaluate import A2MEvaluation
from eval.unconstrained.evaluate import evaluate_unconstrained_metrics
from .tools import save_metrics, format_metrics
from utils import dist_util

num_samples_unconstrained = 1000

class NewDataloader:
    def __init__(self, mode, model,  device, unconstrained, diffusion=None, dataiterator=None, single_batch_size=None, single_batch_sample=None, single_batch_kwargs=None, num_samples: int=-1):
        assert mode in ["gen", "gt", "batch_gen", "batch_gt"]
        
        if mode in ["batch_gen", "batch_gt"]:
            self.from_single_batch(mode, model, single_batch_size, single_batch_sample, single_batch_kwargs, device, unconstrained, num_samples)
            return

        else:
            self.batches = []
            sample_fn = diffusion.p_sample_loop

            # is_firstround = True
            
            with torch.no_grad():
                for motions, model_kwargs in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                    motions = motions.to(device)
                    if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                        continue  # do not break because it confuses the multiple loaders
                    batch = dict()
                    if mode == "gen":
                        sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                        batch['output'] = sample

                        # if is_firstround:
                        #     is_firstround = False
                            # save the output

                    elif mode == "gt":
                        batch["output"] = motions

                    # mask = torch.ones([batch["output"].shape[0], batch["output"].shape[-1]], dtype=bool).to(device)  # batch_size x num_frames
                    max_n_frames = model_kwargs['y']['lengths'].max()
                    mask = model_kwargs['y']['mask'].reshape(dataiterator.batch_size, max_n_frames).bool()
                    batch["output_xyz"] = model.rot2xyz(x=batch["output"], mask=mask, pose_rep='rot6d', glob=True,
                                                        translation=True, jointstype='smpl', vertstrans=True, betas=None,
                                                        beta=0, glob_rot=None, get_rotations_back=False)
                    batch["lengths"] = model_kwargs['y']['lengths'].to(device)
                    if not unconstrained:  # proceed only if not running unconstrained
                        batch["y"] = model_kwargs['y']['action'].squeeze().long().cpu()  # using torch.long so lengths/action will be used as indices
                    self.batches.append(batch)

                num_samples_last_batch = num_samples % dataiterator.batch_size
                if num_samples_last_batch > 0:
                    for k, v in self.batches[-1].items():
                        self.batches[-1][k] = v[:num_samples_last_batch]

    def __iter__(self):
        return iter(self.batches)
    
    def from_single_batch(self, mode, model, batch_size, sample, model_kwargs, device, unconstrained, num_samples: int=-1):
        assert mode in ["batch_gen", "batch_gt"]
        assert batch_size <= sample.shape[0]
        self.batches = []
        batch = dict()

        batch["output"] = sample
        max_n_frames = model_kwargs['y']['lengths'][:batch_size].max()
        mask = model_kwargs['y']['mask'][:batch_size].reshape(batch_size, max_n_frames).bool()
        batch["output_xyz"] = model.rot2xyz(x=batch["output"], mask=mask, pose_rep='rot6d', glob=True,
                                                    translation=True, jointstype='smpl', vertstrans=True, betas=None,
                                                    beta=0, glob_rot=None, get_rotations_back=False)
        batch["lengths"] = model_kwargs['y']['lengths'][:batch_size].to(device)
        if not unconstrained:
            batch["y"] = model_kwargs['y']['action'][:batch_size].squeeze().long().cpu() 
        self.batches.append(batch)

    
    


# my modify: no gt2 allowed!
def evaluate(args, model, diffusion, data, critic_model):
    num_frames = 60

    # fix parameters for action2motion evaluation
    args.num_frames = num_frames
    args.jointstype = "smpl"
    args.vertstrans = True

    device = dist_util.dev()

    model.eval()

    a2mevaluation = A2MEvaluation(device=device, critic_model=critic_model)
    a2mmetrics = {}

    datasetGT1 = copy.deepcopy(data)
    # datasetGT2 = copy.deepcopy(data)

    allseeds = list(range(args.num_seeds))

    try:
        for index, seed in enumerate(allseeds):
            print(f"Evaluation number: {index+1}/{args.num_seeds}")
            fixseed(seed)

            datasetGT1.reset_shuffle()
            datasetGT1.shuffle()

            # datasetGT2.reset_shuffle()
            # datasetGT2.shuffle()

            dataiterator = DataLoader(datasetGT1, batch_size=args.batch_size,
                                      shuffle=False, num_workers=8, collate_fn=collate)
            # dataiterator2 = DataLoader(datasetGT2, batch_size=args.batch_size,
            #                            shuffle=False, num_workers=8, collate_fn=collate)

            new_data_loader = functools.partial(NewDataloader, model=model, diffusion=diffusion, device=device,
                                                unconstrained=args.unconstrained, num_samples=args.num_samples)
            motionloader = new_data_loader(mode="gen", dataiterator=dataiterator)
            gt_motionloader = new_data_loader("gt", dataiterator=dataiterator)
            # gt_motionloader2 = new_data_loader("gt", dataiterator=dataiterator2)

            # Action2motionEvaluation
            # loaders = {"gen": motionloader,
            #            "gt": gt_motionloader,
            #            "gt2": gt_motionloader2}
            loaders = {"gen": motionloader, 
                       "gt": gt_motionloader}


            a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

            del loaders

        if args.unconstrained:  # unconstrained
            dataset_unconstrained = copy.deepcopy(data)
            dataset_unconstrained.reset_shuffle()
            dataset_unconstrained.shuffle()
            dataiterator_unconstrained = DataLoader(dataset_unconstrained, batch_size=args.batch_size,
                                           shuffle=False, num_workers=8, collate_fn=collate)
            motionloader_unconstrained = new_data_loader(mode="gen", dataiterator=dataiterator_unconstrained, num_samples=num_samples_unconstrained)

            generated_motions = []
            for motion in motionloader_unconstrained:
                idx = [15, 12, 16, 18, 20, 17, 19, 21, 0, 1, 4, 7, 2, 5, 8]
                motion = motion['output_xyz'][:, idx, :, :]
                generated_motions.append(motion.cpu().numpy())
            generated_motions = np.concatenate(generated_motions)
            unconstrained_metrics = evaluate_unconstrained_metrics(generated_motions, device, fast=True)
            unconstrained_metrics = {k+'_unconstrained': v for k, v in unconstrained_metrics.items()}

    except KeyboardInterrupt:
        string = "Saving the evaluation before exiting.."
        print(string)

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}
    if args.unconstrained:
        metrics["feats"] = {**metrics["feats"], **unconstrained_metrics}

    return metrics


def evaluate_batch(args, model, batch_size, motion, sample, model_kwargs, critic_model, sample_kwargs=None):
    num_frames = 60

    # fix parameters for action2motion evaluation
    args.num_frames = num_frames
    args.jointstype = "smpl"
    args.vertstrans = True

    device = dist_util.dev()

    model.eval()
    a2mevaluation = A2MEvaluation(device=device, critic_model=critic_model)
    a2mmetrics = {}

    allseeds = list(range(args.num_seeds))
    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index+1}/{args.num_seeds}")
        fixseed(seed)


        new_data_loader = functools.partial(NewDataloader, model=model, device=device,
                                            unconstrained=args.unconstrained, num_samples=args.num_samples)
        
        if sample_kwargs is not None:
            motionloader = new_data_loader(mode="batch_gen", single_batch_sample=sample, single_batch_size=sample.shape[0], single_batch_kwargs=sample_kwargs)
            gt_motionloader = new_data_loader(mode="batch_gt", single_batch_sample=motion, single_batch_size=motion.shape[0],single_batch_kwargs=model_kwargs)
        else:
            motionloader = new_data_loader(mode="batch_gen", single_batch_sample=sample, single_batch_size=sample.shape[0], single_batch_kwargs=model_kwargs)
            gt_motionloader = new_data_loader(mode="batch_gt", single_batch_sample=motion, single_batch_size=motion.shape[0],single_batch_kwargs=model_kwargs)

        loaders = {"batch_gen": motionloader, 
                    "batch_gt": gt_motionloader}

        a2mmetrics[seed] = a2mevaluation.evaluate(model, loaders)

        del loaders

        # if args.unconstrained:  # unconstrained

    metrics = {"feats": {key: [format_metrics(a2mmetrics[seed])[key] for seed in a2mmetrics.keys()] for key in a2mmetrics[allseeds[0]]}}

    return metrics


