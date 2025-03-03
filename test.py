# ptorch run prediction inference

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

import torch
import torch.nn as nn
from accelerate.utils import set_seed, tqdm
from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf,DictConfig
from torch.utils.data import DataLoader

from transforms import Compose
from utils.torch_utils import replace_with_fused_layernorm, worker_init_fn

from dataset.nnUNet2dDataset import nnUNet2dDataset
from model.nnUNetClassifier import nnUNetClassifier

def build_dataset(cfg,decimate=0):

    dataset = nnUNet2dDataset(cfg.dataset.imgdir,cfg.dataset.lbldir,in_memory=cfg.dataset.in_memory,rgb=cfg.dataset.rgb)
    return dataset


@hydra.main(config_path='configs',config_name='test',version_base=None)
def main(cfg:DictConfig):

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        raise ValueError

    # Check cuda and cudnn settings
    torch.backends.cudnn.benchmark = True
    print("flash_sdp_enabled:", torch.backends.cuda.flash_sdp_enabled())
    print("mem_efficient_sdp_enabled:", torch.backends.cuda.mem_efficient_sdp_enabled())
    print("math_sdp_enabled:", torch.backends.cuda.math_sdp_enabled())

    seed = cfg.get("seed", 42)

    # ---------------------------------------------------------------------------- #
    # Setup model
    # ---------------------------------------------------------------------------- #
    # set_seed(seed)
    model: nnUNetClassifier = hydra.utils.instantiate(cfg.model)

    # ---------------------------------------------------------------------------- #
    # Setup dataloaders
    # ---------------------------------------------------------------------------- #
    test_dataset_cfg = hydra.utils.instantiate(cfg.test_dataset)
    test_dataset = build_dataset(test_dataset_cfg)

    test_dataloader = DataLoader(
        test_dataset,
        **cfg.test_dataloader,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    # ---------------------------------------------------------------------------- #
    # test loop
    # ---------------------------------------------------------------------------- #
    
    pbar = tqdm(total=len(test_dataloader))

    res = []
    for i,data in enumerate(test_dataloader):
        outputs = model(**data)
        res.append((data['lbl'].item(),outputs.item()))
        pbar.update(1)

    x = [r[0] for r in res]
    y = [r[1] for r in res]

    plt.scatter(x,y)

    pbar.close()

 


if __name__ == "__main__":
    main()
