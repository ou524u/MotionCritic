import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader
from diffusion.losses import normal_kl

from torch.utils.data import DataLoader
from data_loaders.tensors import collate


import random
import diffusion.gaussian_diffusion as gsdiff

from pubcode.AlignHP.MDMCritic.sample.critic_generate import outof_mdm, into_critic

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_DIR)
os.environ['WANDB_DIR'] = PROJ_DIR + '/wandb/'
os.environ['WANDB_CACHE_DIR'] = PROJ_DIR + '/wandb/.cache/'
os.environ['WANDB_CONFIG_DIR'] = PROJ_DIR + '/wandb/.config/'

import wandb


class TuneLoop:
    def __init__(self, args, train_platform, model, diffusion, data, critic_model):
        self.args = args
        self.dataset = args.dataset
        self.train_platform = train_platform
        self.model = model

        self.critic_model = critic_model
        self.critic_model.eval()

        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.ddim_sampling = args.ddim_sampling

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
                                                   split=args.eval_split,
                                                   hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
            self.eval_data = {
                'test': lambda: eval_humanml.get_mdm_loader(
                    model, diffusion, args.eval_batch_size,
                    gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples, scale=1.,
                )
            }
        self.use_ddp = False
        self.ddp_model = self.model

        self.render_video = args.render_video
        self.do_sample_when_eval = args.sample_when_eval
        self.add_random_critic_loss = args.random_critic_loss
        self.render_batch_size = args.render_batch_size
        self.critic_loss_scale = args.critic_scale
        self.kl_loss_scale = args.critic_scale
        self.denoise_lower_bound = args.denoise_lower
        self.denoise_upper_bound = args.denoise_upper
        # if args.render_gt:
        #     self.gt_rendered = False
        # else:
        #     self.gt_rendered = True
        self.gt_rendered = False
        
        if args.wandb is not None:
            self.use_wandb=True
            wandb.init(project="preexp", name=args.wandb, resume=False)
        else:
            self.use_wandb=False

        self.no_critic_loss = args.no_critic_loss
        self.use_kl_loss = args.use_kl_loss
        if self.use_kl_loss:
            self.pre_sample = None
        self.relu_loss = args.relu_loss
        self.my_eval_seq_condbatch()
        self.my_eval_critic_condbatch()

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        print(f"### total epoch {self.num_epochs}")
        # Save the last checkpoint if it wasn't already saved.
        if self.step % self.save_interval != 0:
            print(f"saving model to {self.save_dir}")
            self.save()
            self.model.eval()
            self.evaluate()
            if self.do_sample_when_eval:
                self.sample_when_eval()
            self.model.train()
        for epoch in range(self.num_epochs):
            print(f'### epoch {epoch} begin')
            # for motion, cond in tqdm(self.data):
            for motion, cond in self.data:
                if self.step % self.save_interval == 0:
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                    
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

                self.run_step(motion, cond)
                
                if not self.no_critic_loss:
                    self.run_critic_step(motion, cond)


                if self.step % self.log_interval == 0:
                    for k,v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss')

                
                self.step += 1
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
        

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        start_eval = time.time()
        if self.eval_wrapper is not None:
            print('Running evaluation loop: [Should take about 90 min]')
            log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):06d}.log')
            diversity_times = 300
            mm_num_times = 0  # mm is super slow hence we won't run it during training
            eval_dict = eval_humanml.evaluation(
                self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
                replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
            print(eval_dict)
            for k, v in eval_dict.items():
                if k.startswith('R_precision'):
                    for i in range(len(v)):
                        self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
                                                          iteration=self.step + self.resume_step,
                                                          group_name='Eval')
                else:
                    self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
                                                      group_name='Eval')

        elif self.dataset in ['humanact12', 'uestc']:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                        batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                        dataset=self.dataset, unconstrained=self.args.unconstrained,
                                        model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model)
            print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    val = np.array(v).astype(float).mean()
                    self.train_platform.report_scalar(name=k, value=val, iteration=self.step, group_name='Eval')
                    # print(f"k is {k} and v.mean is {np.array(v).astype(float).mean()}")
                    if self.use_wandb:
                        wandb.log({k: val})  
                else:
                    self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        end_eval = time.time()
        print(f'Evaluation time: {round(end_eval-start_eval)}sec')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset=self.data.dataset
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            print(f"mdm loss is {loss} ", end=',')
            if self.use_wandb:
                wandb.log({'loss': loss.item()})
            
            self.mp_trainer.backward(loss)

        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(f'regular training Parameter: {name}, Gradient: None')
        #     else:
        #         print(f'regular training Parameter: {name}, Gradient Norm: {param.grad.norm().item()}')


    def run_critic_step(self, batch, cond):
        self.critic_forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def my_eval_condbatch(self):
        evaldata = copy.deepcopy(self.data.dataset)
        # evaldata.reset_shuffle()
        # evaldata.shuffle()        
        dataiterator = DataLoader(evaldata, batch_size=1,
                                  shuffle=False, num_workers=8, collate_fn=collate)
        prompt_count = {}
        batch_list = []
        cond_list = []
        for i, (batch, cond) in enumerate(dataiterator):
            prompt = cond['y']['action_text'][0]
            if prompt in prompt_count:
                if prompt_count[prompt] >= 10:
                    continue                
                prompt_count[prompt] += 1
            else:
                prompt_count[prompt] = 1
            batch_list.append(batch)
            cond_list.append(cond)


        self.evalbatch = torch.cat(batch_list, dim=0)

        def merge_conds(cond_list):
            merged_cond = {
                'y': {
                    'mask': None,
                    'lengths': None,
                    'action': None,
                    'action_text': []
                }
            }
            mask_list = []
            lengths_list = []
            action_list = []
            action_text_list = []
            for cond in cond_list:
                mask_list.append(cond['y']['mask'])
                lengths_list.append(cond['y']['lengths'])
                action_list.append(cond['y']['action'])
                action_text_list.append(cond['y']['action_text'])
            mask_list = torch.cat(mask_list, dim=0)
            lengths_list = torch.cat(lengths_list, dim=0)
            action_list = torch.cat(action_list, dim=0)
            merged_cond['y']['mask'] = mask_list
            merged_cond['y']['lengths'] = lengths_list
            merged_cond['y']['action'] = action_list
            merged_cond['y']['action_text'] = action_text_list
            return merged_cond
        
        self.evalcond = merge_conds(cond_list)

    def my_eval_seq_condbatch(self):
        evaldata = copy.deepcopy(self.data.dataset)
        # evaldata.reset_shuffle()
        # evaldata.shuffle()        
        dataiterator = DataLoader(evaldata, batch_size=1,
                                  shuffle=False, num_workers=8, collate_fn=collate)
        prompt_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        batch_list = [[] for _ in range(12)]
        cond_list = [[] for _ in range(12)]
        for i, (batch, cond) in enumerate(dataiterator):
            prompt = cond['y']['action'][0][0].item()
            if prompt_count[prompt] >= 10:
                continue
            prompt_count[prompt] = prompt_count[prompt] + 1
            batch_list[prompt].append(batch)
            cond_list[prompt].append(cond)

        batch_list = [item for sublist in batch_list for item in sublist]
        cond_list = [item for sublist in cond_list for item in sublist]
        self.evalbatch = torch.cat(batch_list, dim=0)

        def merge_conds(cond_list):
            merged_cond = {
                'y': {
                    'mask': None,
                    'lengths': None,
                    'action': None,
                    'action_text': []
                }
            }
            mask_list = []
            lengths_list = []
            action_list = []
            action_text_list = []
            for cond in cond_list:
                mask_list.append(cond['y']['mask'])
                lengths_list.append(cond['y']['lengths'])
                action_list.append(cond['y']['action'])
                action_text_list.append(cond['y']['action_text'])
            mask_list = torch.cat(mask_list, dim=0)
            lengths_list = torch.cat(lengths_list, dim=0)
            action_list = torch.cat(action_list, dim=0)
            merged_cond['y']['mask'] = mask_list
            merged_cond['y']['lengths'] = lengths_list
            merged_cond['y']['action'] = action_list
            merged_cond['y']['action_text'] = action_text_list
            return merged_cond
        
        self.evalcond = merge_conds(cond_list)


    def my_eval_gtfullbatch(self):
        evaldata = copy.deepcopy(self.data.dataset)
        # evaldata.reset_shuffle()
        # evaldata.shuffle()        
        dataiterator = DataLoader(evaldata, batch_size=1,
                                  shuffle=False, num_workers=8, collate_fn=collate)
        prompt_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        batch_list = [[] for _ in range(12)]
        cond_list = [[] for _ in range(12)]
        for i, (batch, cond) in enumerate(dataiterator):
            prompt = cond['y']['action'][0][0].item()
            # if prompt_count[prompt] >= 10:
            #     continue
            prompt_count[prompt] = prompt_count[prompt] + 1
            batch_list[prompt].append(batch)
            cond_list[prompt].append(cond)

        # batch_list = [item for sublist in batch_list for item in sublist]
        # cond_list = [item for sublist in cond_list for item in sublist]
        

        def merge_conds(cond_list):
            merged_cond = {
                'y': {
                    'mask': None,
                    'lengths': None,
                    'action': None,
                    'action_text': []
                }
            }
            mask_list = []
            lengths_list = []
            action_list = []
            action_text_list = []
            for cond in cond_list:
                mask_list.append(cond['y']['mask'])
                lengths_list.append(cond['y']['lengths'])
                action_list.append(cond['y']['action'])
                action_text_list.append(cond['y']['action_text'])
            mask_list = torch.cat(mask_list, dim=0)
            lengths_list = torch.cat(lengths_list, dim=0)
            action_list = torch.cat(action_list, dim=0)
            merged_cond['y']['mask'] = mask_list
            merged_cond['y']['lengths'] = lengths_list
            merged_cond['y']['action'] = action_list
            merged_cond['y']['action_text'] = action_text_list
            return merged_cond
        
        for i in range(12):
            batch = torch.cat(batch_list[i], dim=0)
            cond = merge_conds(cond_list[i])
            # batch = into_critic(outof_mdm(batch))
            batch = outof_mdm(batch)
            data = {
                'motion': batch,
                'cond': cond,
            }
            torch.save(data, os.path.join(PROJ_DIR, f"save/critics/motion-gt{i}.pth"))


    def my_savegt_uestc(self):
        evaldata = copy.deepcopy(self.data.dataset)
        # evaldata.reset_shuffle()
        # evaldata.shuffle()        
        dataiterator = DataLoader(evaldata, batch_size=1,
                                  shuffle=False, num_workers=8, collate_fn=collate)
        prompt_count = [0 for i in range(40)]
        batch_list = [[] for _ in range(40)]
        cond_list = [[] for _ in range(40)]
        for i, (batch, cond) in enumerate(dataiterator):
            prompt = cond['y']['action'][0][0].item()
            # if prompt_count[prompt] >= 10:
            #     continue
            prompt_count[prompt] = prompt_count[prompt] + 1
            batch_list[prompt].append(batch)
            cond_list[prompt].append(cond)

        # batch_list = [item for sublist in batch_list for item in sublist]
        # cond_list = [item for sublist in cond_list for item in sublist]
        

        def merge_conds(cond_list):
            merged_cond = {
                'y': {
                    'mask': None,
                    'lengths': None,
                    'action': None,
                    'action_text': []
                }
            }
            mask_list = []
            lengths_list = []
            action_list = []
            action_text_list = []
            for cond in cond_list:
                mask_list.append(cond['y']['mask'])
                lengths_list.append(cond['y']['lengths'])
                action_list.append(cond['y']['action'])
                action_text_list.append(cond['y']['action_text'])
            mask_list = torch.cat(mask_list, dim=0)
            lengths_list = torch.cat(lengths_list, dim=0)
            action_list = torch.cat(action_list, dim=0)
            merged_cond['y']['mask'] = mask_list
            merged_cond['y']['lengths'] = lengths_list
            merged_cond['y']['action'] = action_list
            merged_cond['y']['action_text'] = action_text_list
            return merged_cond
        
        for i in range(40):
            batch = torch.cat(batch_list[i], dim=0)
            cond = merge_conds(cond_list[i])
            # batch = into_critic(outof_mdm(batch))
            batch = outof_mdm(batch)
            data = {
                'motion': batch,
                'cond': cond,
            }
            torch.save(data, os.path.join(PROJ_DIR, f"save/critics/motion-gtuestc{i}.pth"))


    def my_eval_critic_condbatch(self):
        evaldata = copy.deepcopy(self.data.dataset)
        dataiterator = DataLoader(evaldata, batch_size=1,
                                  shuffle=False, num_workers=8, collate_fn=collate)
        batch_list = [[] for _ in range(12)]
        cond_list = [[] for _ in range(12)]
        for i, (batch, cond) in enumerate(dataiterator):
            prompt = cond['y']['action'][0][0].item()
            batch_list[prompt].append(batch)
            cond_list[prompt].append(cond)

        batch_list = [item for sublist in batch_list for item in sublist]
        cond_list = [item for sublist in cond_list for item in sublist]
        merged_batch = torch.cat(batch_list, dim=0)
        merged_batch = into_critic(outof_mdm(merged_batch))
        self.critic_model.to(dist_util.dev())
        merged_batch = merged_batch.to(dist_util.dev())
        critics = self.critic_model.module.batch_critic(merged_batch)
        critics = torch.squeeze(critics)

# critics is a tensor with shape [x]. critic-batch-cond are connected pair. I want to sort and reorder them according to critic value from large to small. how?
        sorted_indices = torch.argsort(critics, descending=True)
    

        # Use the sorted indices to reorder batch_list and cond_list
        sorted_batch_list = [batch_list[i] for i in sorted_indices]
        sorted_cond_list = [cond_list[i] for i in sorted_indices]

        def merge_conds(cond_list):
            merged_cond = {
                'y': {
                    'mask': None,
                    'lengths': None,
                    'action': None,
                    'action_text': []
                }
            }
            mask_list = []
            lengths_list = []
            action_list = []
            action_text_list = []
            for cond in cond_list:
                mask_list.append(cond['y']['mask'])
                lengths_list.append(cond['y']['lengths'])
                action_list.append(cond['y']['action'])
                action_text_list.append(cond['y']['action_text'])
            mask_list = torch.cat(mask_list, dim=0)
            lengths_list = torch.cat(lengths_list, dim=0)
            action_list = torch.cat(action_list, dim=0)
            merged_cond['y']['mask'] = mask_list
            merged_cond['y']['lengths'] = lengths_list
            merged_cond['y']['action'] = action_list
            merged_cond['y']['action_text'] = action_text_list
            return merged_cond
        
        
        sorted_batch = torch.cat(sorted_batch_list, dim=0)
        sorted_cond = merge_conds(sorted_cond_list)
        self.evalfullbatch = sorted_batch
        self.evalfullcond = sorted_cond

        self.evalbatch0 = sorted_batch[:238]
        self.evalcond0 = merge_conds(sorted_cond_list[:238])

        self.evalbatch1 = sorted_batch[238:476]
        self.evalcond1 = merge_conds(sorted_cond_list[238:476])

        self.evalbatch2 = sorted_batch[476:714]
        self.evalcond2 = merge_conds(sorted_cond_list[476:714])

        self.evalbatch3 = sorted_batch[714:952]
        self.evalcond3 = merge_conds(sorted_cond_list[714:952])

        self.evalbatch4 = sorted_batch[952:1190]
        self.evalcond4 = merge_conds(sorted_cond_list[952:1190])
        torch.save(outof_mdm(self.evalbatch4), os.path.join(PROJ_DIR, f"save/direct/motion-evalbatch4.pth"))

        print(f"self.evalbatch0 device is {self.evalbatch0.device}, self.evalfullbatch device is {self.evalfullbatch.device}")

    def sample_and_save_multi(self, step=None):
        # self.sample_and_save(0)
        # self.sample_and_save(1)
        # self.sample_and_save(2)
        # self.sample_and_save(3)
        # self.sample_and_save(4)
        self.sample_and_save(step=step)

    def sample_and_save(self, id=None, step=None):
        if id is None:
            batch = self.evalfullbatch.to(dist_util.dev())
            cond = self.evalfullcond
            micro = batch
            micro_cond = cond

        if id == 0:
            batch = self.evalbatch0.to(dist_util.dev())
            cond = self.evalcond0
            micro = batch
            micro_cond = cond

        if id == 1:
            batch = self.evalbatch1.to(dist_util.dev())
            cond = self.evalcond1
            micro = batch
            micro_cond = cond

        if id == 2:
            batch = self.evalbatch2.to(dist_util.dev())
            cond = self.evalcond2
            micro = batch
            micro_cond = cond

        if id == 3:
            batch = self.evalbatch3.to(dist_util.dev())
            cond = self.evalcond3
            micro = batch
            micro_cond = cond

        if id == 4:
            batch = self.evalbatch4.to(dist_util.dev())
            cond = self.evalcond4
            micro = batch
            micro_cond = cond


        # # pick random time
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        # sample x_t
        x_t = self.diffusion.q_sample(micro, t)# something here
        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.model,
            shape = x_t.shape,
            init_image=None,
            model_kwargs=micro_cond,
            clip_denoised=False,
            skip_timesteps= 0,  # 0 is the default value - i.e. don't skip any step
            # dump_steps=dump_list, # dump_steps set to true, returning a list, with each step.
            # dump_steps=None,
            progress=True,
            # progress=False,
            cond_fn_with_grad=False,
            detach=False,
        )

        
        critics = self.critic_model.module.batch_critic(into_critic(outof_mdm(sample)))
        critics = torch.squeeze(critics)
        # saving motions and critic scores
        # if id is not None:
        #     torch.save(critics, os.path.join(PROJ_DIR, f"save/critics/uncriticed-critics-batch{id}.pth"))
        #     torch.save(outof_mdm(sample), os.path.join(PROJ_DIR, f"save/critics/uncriticed-motions-evalfullbatch{id}.pth"))
        # else:
        #     torch.save(critics, os.path.join(PROJ_DIR, f"save/critics/uncriticed-critics{id}.pth"))
        #     torch.save(outof_mdm(sample), os.path.join(PROJ_DIR, f"save/critics/uncriticed-motions{step}.pth"))

    def eval_fid(self):
        
        if self.dataset in ["humanact12"]:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                    batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                    dataset=self.dataset, unconstrained=self.args.unconstrained,
                                    model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            
            self.evalbatch0 = self.evalbatch0.to(dist_util.dev())
            self.evalfullbatch = self.evalfullbatch.to(dist_util.dev())
            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=1190, motion=self.evalfullbatch, sample=self.evalbatch0, model_kwargs=self.evalfullcond, sample_kwargs=self.evalcond0)
            
            val =  np.array(eval_dict["feats"]["fid_batch_gen"]).astype(float).mean()
            print(f"fid between evalbatch0 and evalfullbatch, fid_batch_gen is {val}")      

            
            # self.evalbatch0 = self.evalbatch0.to(dist_util.dev())
            # self.evalfullbatch = self.evalfullbatch.to(dist_util.dev())
            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=1190, motion=self.evalfullbatch, sample=self.evalbatch1, model_kwargs=self.evalfullcond, sample_kwargs=self.evalcond1)
            
            val =  np.array(eval_dict["feats"]["fid_batch_gen"]).astype(float).mean()
            print(f"fid between evalbatch1 and evalfullbatch, fid_batch_gen is {val}")      


            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=1190, motion=self.evalfullbatch, sample=self.evalbatch2, model_kwargs=self.evalfullcond, sample_kwargs=self.evalcond2)
            
            val =  np.array(eval_dict["feats"]["fid_batch_gen"]).astype(float).mean()
            print(f"fid between evalbatch2 and evalfullbatch, fid_batch_gen is {val}")   


            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=1190, motion=self.evalfullbatch, sample=self.evalbatch3, model_kwargs=self.evalfullcond, sample_kwargs=self.evalcond3)
            
            val =  np.array(eval_dict["feats"]["fid_batch_gen"]).astype(float).mean()
            print(f"fid between evalbatch3 and evalfullbatch, fid_batch_gen is {val}")   

            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=1190, motion=self.evalfullbatch, sample=self.evalbatch4, model_kwargs=self.evalfullcond, sample_kwargs=self.evalcond4)
            
            val =  np.array(eval_dict["feats"]["fid_batch_gen"]).astype(float).mean()
            print(f"fid between evalbatch4 and evalfullbatch, fid_batch_gen is {val}")   
             





    def sample_when_eval(self):
        take_batch_size = 120
        batch = self.evalbatch.to(dist_util.dev())
        cond = self.evalcond
        micro = batch
        micro_cond = cond

        # # pick random time
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        # sample x_t
        x_t = self.diffusion.q_sample(micro, t)# something here
        sample_fn = self.diffusion.p_sample_loop
        # dump_list = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        # dump_list = [930, 970, 990, 1000]
        dump_list = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 980, 990, 995, 1000]
        sampled_list = sample_fn(
            self.model,
            shape = x_t.shape,
            init_image=None,
            model_kwargs=micro_cond,
            clip_denoised=False,
            skip_timesteps= 0,  # 0 is the default value - i.e. don't skip any step
            dump_steps=dump_list, # dump_steps set to true, returning a list, with each step.
            # dump_steps=None,
            progress=True,
            # progress=False,
            cond_fn_with_grad=False,
            detach=False,
        )

# what to do:
# check the denoising process, how critic goes along
        
        # compute complete scores for the dump_list, but only render the sampled video out
        sampled_critic_list = []
        for k, sample in enumerate(sampled_list):
            sampled_critic_list.append(self.critic_model.module.clipped_critic(into_critic(outof_mdm(sample))).item())

        print(f"### eval-sample at step {self.step}, critic list {sampled_critic_list}")
        if self.use_wandb:
            wandb_data = [[x,y] for (x,y) in zip(dump_list, sampled_critic_list)]
            table = wandb.Table(data=wandb_data, columns = ["denoise step", "critic"])
            wandb.log({f"denoising-process at valstep{self.step}" : wandb.plot.line(table, "denoise step", "critic", stroke=None, title=f"denoising-process at valstep{self.step}")})

# what to do:
# save a batch. the saved batch shall be used in user study.
# eval that batch. upon fid and etc.
# render a motion of that batch.
# here we care not the denoising process.

        check_list = []
        text_list = micro_cond['y']['action_text'][:take_batch_size]

        for k, sample in enumerate(sampled_list):
            if k == len(sampled_list) - 1:
                check_list.append(outof_mdm(sample[:take_batch_size]))

        check_list = torch.cat(check_list, dim=0)
        
        score_list = []
        for k in range(check_list.shape[0]):
            # compute separatly each motion's critic
            score_k = self.critic_model.module.clipped_critic(into_critic(check_list[k:k+1]))
            score_list.append(score_k.item())

        # rendering out these motions
        comment_list = []

        
        path_list = []
        pathdir = os.path.join(PROJ_DIR, self.save_dir)
        pathdir = os.path.join(pathdir, f"step{self.step}")
        os.makedirs(pathdir, exist_ok=True)
        for k in range(check_list.shape[0]):
            comment_list.append(f"sampled, step: {self.step}, prompt: {text_list[k]}, critic: {score_list[k]:.3f}")
            path_list.append(os.path.join(pathdir, f"step{self.step}-sample{k//10}-{k%10}.mp4"))
        
        # render only one motion as visual check.
        # render_multi(check_list[:1], check_list.device, comment_list[:1], path_list[:1])

        # save that batch.
        data = {
            'motion': check_list,
            'comment': comment_list,
            'path': path_list,
        }
        batchpath = os.path.join(pathdir, f"step{self.step}-evalbatch.pth")
        torch.save(data, batchpath)
        print(f"batch saved at {batchpath}")


        if self.gt_rendered is False:
            gtcheck_list = outof_mdm(batch[:take_batch_size])
            gttext_list = text_list
            gtscore_list = []
            for k in range(gtcheck_list.shape[0]):
                score_m = self.critic_model.module.clipped_critic(into_critic(gtcheck_list[k:k+1]))
                gtscore_list.append(score_m.item())

            gtcomment_list = []
            gtpath_list = []
            gtpathdir = os.path.join(PROJ_DIR, self.save_dir)
            gtpathdir = os.path.join(gtpathdir, f"gt")
            os.makedirs(gtpathdir, exist_ok=True)
            for k in range(gtcheck_list.shape[0]):
                gtcomment_list.append(f"ground truth, prompt: {gttext_list[k]}, critic: {gtscore_list[k]}")
                gtpath_list.append(os.path.join(gtpathdir, f"gt{k//10}-{k%10}.mp4"))
            
            # render some gt
        
            # render_multi(gtcheck_list[:1], gtcheck_list.device, gtcomment_list[:1], gtpath_list[:1])
            gtdata = {
                'motion': gtcheck_list,
                'comment': gtcomment_list,
                'path': gtpath_list,
            }
            gtpath = os.path.join(gtpathdir, f"gtbatch.pth")
            torch.save(gtdata, gtpath)
            print(f"batch saved at {gtpath}")
            self.gt_rendered = True


# eval that batch.
        if self.dataset in ["humanact12"]:
            eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
                                    batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
                                    dataset=self.dataset, unconstrained=self.args.unconstrained,
                                    model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
            eval_dict =  eval_humanact12_uestc.evaluate(
                eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset, critic_model=self.critic_model,
                single_batch=True, batch_size=take_batch_size, motion=batch[:take_batch_size], sample=sampled_list[-1][:take_batch_size], model_kwargs=cond)
        
            print(f'sample_when_eval, Evaluation results on batch-{self.dataset}: {sorted(eval_dict["feats"].items())}')
            for k, v in eval_dict["feats"].items():
                if 'unconstrained' not in k:
                    val = np.array(v).astype(float).mean()
                    self.train_platform.report_scalar(name=k, value=val, iteration=self.step, group_name='Eval')
                    # print(f"k is {k} and v.mean is {np.array(v).astype(float).mean()}")
                    if self.use_wandb:
                        wandb.log({k: val})                 
        

        

        
    def critic_forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        alpha = self.critic_loss_scale
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond

            # print(f"micro cond is {micro_cond}")
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # # pick random time
            denoise_t = random.randint(self.denoise_lower_bound, self.denoise_upper_bound)
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # print(f"micro shape {micro.shape}, t shape {t.shape}")
            # sample x_t
            x_t = self.diffusion.q_sample(micro, t)# something here
            x_t = x_t.to(self.device)

            # denoise for a few steps for x_t
            
            
            if self.ddim_sampling:
                sample_fn = self.diffusion.ddim_sample_loop
            else:
                sample_fn = self.diffusion.p_sample_loop
            torch.set_grad_enabled(False)
            compute_sample = functools.partial(
                sample_fn,
                self.model,
                shape = x_t.shape,
                init_image=x_t,
                model_kwargs=micro_cond,
                clip_denoised=False,
                skip_timesteps= self.diffusion.num_timesteps - (denoise_t-1),  # 0 is the default value - i.e. don't skip any step
                # progress=True,
                progress=False,
                cond_fn_with_grad=False,
                detach=False,
            )
            sample = compute_sample()

            torch.set_grad_enabled(True)
            compute_sample_pred = functools.partial(
                sample_fn,
                self.model,
                shape = sample.shape,
                init_image=sample,
                model_kwargs=micro_cond,
                clip_denoised=False,
                skip_timesteps= self.diffusion.num_timesteps - 1,  # 0 is the default value - i.e. don't skip any step
                # progress=True,
                progress=False,
                cond_fn_with_grad=True,
                detach=False,
            )
            sample_pred = compute_sample_pred()
            sample_p = sample_pred.clone().detach()

            
            
            sample_pred = into_critic(outof_mdm(sample_pred))
            
                    

            critic_loss = self.critic_model.module.clipped_critic(sample_pred)
           
            # backward
            # print(f"denoise_t is {denoise_t}, critic_output is {critic_loss.item()}", end = ", ")
            print(f"critic_output is {critic_loss.item()}", end = ", ")
            if self.use_wandb:
                wandb.log({'critic_output': critic_loss.item()})

            if self.add_random_critic_loss:
                random_critic_loss = torch.rand(1, device=critic_loss.device)
                critic_loss = critic_loss + random_critic_loss

            # critic_loss method here
            if self.relu_loss:
                critic_loss = alpha * torch.nn.functional.relu(12 - critic_loss)

            else:
                critic_loss = alpha * torch.sigmoid(-critic_loss)


            print(f"critic_loss is {critic_loss.item()}")

            if self.use_kl_loss:
                kl_loss = torch.tensor(0.0, device= self.device, requires_grad=True)
                if self.pre_sample is not None:
                    pre_mean = torch.mean(self.pre_sample)
                    pre_logvar = torch.log(torch.var(self.pre_sample, unbiased=False))
                    curr_mean = torch.mean(sample_pred)
                    curr_logvar = torch.log(torch.var(sample_pred, unbiased=False))
                    # kl_loss = normal_kl(pre_mean, pre_logvar, curr_mean, curr_logvar)  
                    kl_loss = normal_kl(pre_mean, pre_logvar, curr_mean, curr_logvar)  
                critic_loss = critic_loss + self.kl_loss_scale * kl_loss
                if self.use_wandb:
                    wandb.log({'kl_loss': kl_loss.item()})

            if self.use_wandb:
                wandb.log({'critic_loss': critic_loss.item()})

            
            self.mp_trainer.backward(critic_loss)
            self.pre_sample = sample_pred.detach()
            

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):06d}.pt"


    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
