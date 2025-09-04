# ptorch main training loop

import argparse
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import platform
import pdb # for debugging on aws
import cProfile,pstats

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.accelerator import ProjectConfiguration
from accelerate.utils import set_seed, tqdm
from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf,DictConfig
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from transforms.transforms import Compose
# from pc_sam.model.loss import compute_iou
# from pc_sam.model.pc_sam import PointCloudSAM
from utils.torch_utils import replace_with_fused_layernorm, worker_init_fn

from dataset.nnUNet2dDataset import nnUNet2dDataset
from dataset.gNet2dDataset import gNet2dDataset
from model.RadNecClassifier import RadNecClassifier
from model.model import load_statedict
from psam.loss import Criterion

# convenience functions for config .yaml's 
# 
# for testing on different machines
# returns a root directory as a general reference
# originally this had a specific project (radnec2) included
# it should really just reference no more than the general data directory
# and specific projects be listed in the config .yaml's
def get_uname():
    uname = platform.uname()
    if 'xps15' in uname.node:
        return '/media/jbishop/WD4/brainmets/sunnybrook/radnec2'
    elif 'XPS-8950' in uname.node:
        # return '/home/jbishop/data/radnec2'
        return '/home/jbishop/data'
    elif 'amzn' in uname.mode:
        return '/home/ec2-user'
    assert False

def compute_mu_std(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
        # collate_fn = collate_fn 
    )
    mean = 0.0
    std = 0.0
    total_samples = 0

    for data in dataloader:
        images = data['img']
        if torch.max(images) > 1.0:
            raise ValueError('possible image scaling error')
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean, std

OmegaConf.register_new_resolver('getuname',get_uname)
OmegaConf.register_new_resolver('computemustd',compute_mu_std)


# dataset functions
def build_dataset(cfg,decimate=0,num_classes=2,onehot=False,transform=None):
    # dataset = nnUNet2dDataset(cfg.dataset.imgdir,cfg.dataset.lbldir,
    #                                 transform=transform,
    #                                 decimate=decimate,
    #                                 in_memory=cfg.dataset.keep_in_memory,
    #                                 rgb=True,
    #                                 num_classes=num_classes,
    #                                 onehot=onehot)
    dataset = gNet2dDataset(cfg.dataset.datadir,
                                    transform=transform,
                                    decimate=decimate,
                                    in_memory=cfg.dataset.keep_in_memory,
                                    rgb=cfg.dataset.rgb)
    return dataset

def build_datasets(cfg):
    if "dataset_dict" in cfg:
        datasets = DatasetDict()
        for key, dataset_cfg in cfg.dataset_dict.items():
            datasets[key] = build_dataset(dataset_cfg)
        return ConcatDataset(datasets.values())
    else:
        return build_dataset(cfg)


@hydra.main(config_path='configs',config_name='large',version_base=None)
def main(cfg:DictConfig):
    # ---------------------------------------------------------------------------- #
    # Load configuration. option to initialize manually with a comand line arg
    # ---------------------------------------------------------------------------- #
    if False:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", type=str, default="large", help="path to config file"
        )
        parser.add_argument("--config_dir", type=str, default="./configs")
        args, unknown_args = parser.parse_known_args()

        with hydra.initialize(args.config_dir, version_base=None):
            cfg = hydra.compose(config_name=args.config, overrides=unknown_args)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        raise ValueError

    # any mods needed for aws versus local
    if 'amzn' in platform.release():
        cfg['train_dataset']['dataset']['path'] = '/home/ec2-user/psam_training/'
        cfg['val_dataset']['dataset']['path'] = '/home/ec2-user/psam_validation/'


    # Prepare (flat) hyperparameters for logging
    hparams = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "batch_size": cfg.train_dataloader.batch_size * cfg.gradient_accumulation_steps,
    }

    # Check cuda and cudnn settings
    torch.backends.cudnn.benchmark = True
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    set_seed(seed)
    model: RadNecClassifier = hydra.utils.instantiate(cfg.model)
    model.train()
    if False:
        # a keras-like summary, but requires an arg for lbl and (1,0,0) doesn't work
        summary(model,(1,3,256,256),(1,0,0))
    else:
        print(model)

    # ---------------------------------------------------------------------------- #
    # Setup dataloaders
    # ---------------------------------------------------------------------------- #
    if False:
        # augmentation. normalize according to data stats
        pre_dataset_cfg = hydra.utils.instantiate(cfg.train_dataset)
        pre_dataset = build_dataset(pre_dataset_cfg,
                                    num_classes=cfg.model.resnet.num_classes,
                                    decimate=cfg.decimate)
        mu,std = compute_mu_std(pre_dataset)
        for transform in cfg.train_dataset.transforms.transforms:
            if "_target_" in transform and transform["_target_"] == "torchvision.transforms.Normalize":
                transform.mean = mu.tolist()
                transform.std = std.tolist()
                break  

    # now do the initialization with the runtime stats.
    train_dataset_cfg = hydra.utils.instantiate(cfg.train_dataset)
    if cfg.augment: # no augmentation
        train_transform = hydra.utils.instantiate(cfg.train_dataset.transforms)
    else:
        train_transform = None
    train_dataset = build_dataset(train_dataset_cfg,
                                  num_classes=cfg.model.resnet.num_classes,
                                  decimate=cfg.decimate,
                                  onehot=cfg.onehot,
                                  transform=train_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        **cfg.train_dataloader,
        # worker_init_fn=worker_init_fn,
        # generator=torch.Generator().manual_seed(seed)
        # collate_fn = collate_fn 
    )

    if cfg.val_freq > 0:
        if False:
            for transform in cfg.val_dataset.transforms.transforms:
                if "_target_" in transform and transform["_target_"] == "torchvision.transforms.Normalize":
                    transform.mean = mu.tolist()
                    transform.std = std.tolist()
                    break  

        if cfg.augment:
            val_transform = hydra.utils.instantiate(cfg.val_dataset.transforms)
        else:
            val_transform = None
        val_dataset_cfg = hydra.utils.instantiate(cfg.val_dataset)
        val_dataset = build_dataset(val_dataset_cfg,
                                    decimate=cfg.decimate,
                                    onehot=cfg.onehot,
                                    num_classes=cfg.model.resnet.num_classes,
                                    transform=val_transform)
        val_dataloader = DataLoader(
            val_dataset, **cfg.val_dataloader,
            # worker_init_fn=worker_init_fn
        )

    # ---------------------------------------------------------------------------- #
    # Setup optimizer
    # ---------------------------------------------------------------------------- #
    params = []
    for name, module in model.named_children():
        # NOTE: Different learning rates can be set for different modules
        if name == "nnunet_encoder":
            params += [{"params": module.parameters(), "lr": cfg.lr}]
        else:
            params += [{"params": module.parameters(), "lr": cfg.lr}]

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    
    if False: # get a type error trying to update cfg with a tensor
        cfg.loss.pos_weight = pos_weight
        criterion = hydra.utils.instantiate(cfg.loss)
    else: # just make it directly for now
        if cfg.onehot:
            criterion = Criterion(onehot=cfg.onehot,weights=train_dataset.weights)
        else:
            pos_weight = train_dataset.balancedata()
            criterion = Criterion(pos_weight=pos_weight)

    # ---------------------------------------------------------------------------- #
    # Initialize accelerator
    # ---------------------------------------------------------------------------- #
    project_config = ProjectConfiguration(
        output_dir, automatic_checkpoint_naming=True, total_limit=4
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
        log_with=cfg.log_with,
    )
    if True:
        model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        if cfg.val_freq > 0:
            val_dataloader = accelerator.prepare(val_dataloader)

    if cfg.log_with:
        accelerator.init_trackers(
            project_name=cfg.get("project_name", "radnec_classify"),
            config=hparams,
        )
    if cfg.log_with == "tensorboard":
        tboard_tracker = accelerator.get_tracker("tensorboard")
        try:
            file_path = os.path.join(output_dir, "full_config.yaml")
            with open(file_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
            # tboard_tracker.run.save(file_path)
        except:
            pass

    # Define validation function
    @torch.no_grad()
    def validate():
        model.eval()

        pbar = tqdm(total=len(val_dataloader))

        vloss = {}
        for i,data in enumerate(val_dataloader):
            outputs = model(**data, is_eval=True)
            vloss[i] = criterion(outputs,data['lbl'])

            pbar.update(1)

        pbar.close()
        avg_vloss = np.mean(np.array([vloss[l].item() for l in vloss.keys()]))

        metrics = dict(loss=avg_vloss)

        return metrics

    # ---------------------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------------------- #
    step = 0  # Number of batch steps
    global_step = 0  # Number of optimization steps
    start_epoch = 0

    # Restore state
    ckpt_dir = Path(accelerator.project_dir, "checkpoints")
    if ckpt_dir.exists():
        accelerator.load_state()
        global_step = scheduler.scheduler.last_epoch // accelerator.state.num_processes
        get_epoch_fn = lambda x: int(x.name.split("_")[-1])
        last_ckpt_dir = sorted(ckpt_dir.glob("checkpoint_*"), key=get_epoch_fn)[-1]
        start_epoch = get_epoch_fn(last_ckpt_dir) + 1
        accelerator.project_configuration.iteration = start_epoch

    for epoch in range(start_epoch, cfg.max_epochs):

        pbar = tqdm(total=len(train_dataloader))

        if False:
            p = cProfile.Profile()
            p.enable()

        model.train()
        for data in train_dataloader:

            flag = (step + 1) % cfg.gradient_accumulation_steps == 0

            ctx = nullcontext if flag else accelerator.no_sync
            # https://huggingface.co/docs/accelerate/en/concept_guides/gradient_synchronization#solving-the-slowdown-problem
            with ctx(model):
                # NOTE: `forward` method needs to be implemented for `accelerate` to apply autocast
                outputs = model(**data)
                loss = criterion(outputs, data['lbl'])
                # with crossEntropyLoss averaging to mean is already performed
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps
                accelerator.backward(loss)

            if flag:
                if cfg.max_grad_value:
                    nn.utils.clip_grad.clip_grad_value_(
                        model.parameters(), cfg.max_grad_value
                    )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Compute metrics

                metrics = dict(loss=loss.item())

                if cfg.log_with and False:
                    accelerator.log(metrics, step=global_step)

                global_step += 1

            pbar.update(1)
            step += 1
            if global_step >= cfg.max_steps:
                break

        pbar.close()

        # log metrics
        if cfg.log_with:
            accelerator.log(metrics, step=global_step)

        # Save state
        if (epoch + 1) % cfg.get("save_freq", 1) == 0:
            accelerator.save_state()

        if cfg.val_freq > 0 and (epoch + 1) % cfg.val_freq == 0:
            metrics = validate()
            if cfg.log_with:
                metrics = {("val/" + k): v for k, v in metrics.items()}
                accelerator.log(metrics, step=global_step)

        if global_step >= cfg.max_steps:
            break

    accelerator.end_training()


if __name__ == "__main__":
    main()
