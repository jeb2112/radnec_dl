# ptorch run prediction inference

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import platform
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transforms import Compose
from utils.torch_utils import replace_with_fused_layernorm, worker_init_fn

from accelerate.utils import set_seed, tqdm
from datasets import DatasetDict, load_dataset, load_from_disk
from omegaconf import OmegaConf,DictConfig

from dataset.nnUNet2dDataset import nnUNet2dDataset
from model.nnUNetClassifier import nnUNetClassifier

def get_uname():
    uname = platform.uname()
    if 'xps15' in uname.node:
        return '/media/jbishop/WD4/brainmets/sunnybrook/radnec2/radnec_classify'
    elif 'XPS-8950' in uname.node:
        return '/home/jbishop/data/radnec2'
    assert False

def build_dataset(cfg,decimate=0):
    dataset = nnUNet2dDataset(cfg.dataset.imgdir,cfg.dataset.lbldir,
                                    transform=Compose(cfg.transforms),
                                    decimate=decimate,
                                    in_memory=cfg.dataset.keep_in_memory,
                                    rgb=True,
                                    split=cfg.dataset.split)
    return dataset


OmegaConf.register_new_resolver('getuname',get_uname)

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
    model: nnUNetClassifier = hydra.utils.instantiate(cfg.test_dataset.model)

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
    

    outputdir = os.sep+os.path.join(*cfg.test_dataset.model.resnet.ckpt_dir.split(os.sep)[:-2])
    fpath = os.path.join(outputdir,'results.pkl')
    if os.path.exists(fpath):
        with open(fpath,'rb') as fp:
            res = pickle.load(fp)

    else:

        pbar = tqdm(total=len(test_dataloader))
        res = []
        for i,data in enumerate(test_dataloader):
            logits = model(**data)
            probs = F.softmax(logits.data, dim=1)
            arg = torch.argmax(probs,dim=1)
            # pred = np.max(probs)
            lbl = data['lbl'].item()
            res.append((lbl,probs[0,lbl].item()))
            pbar.update(1)
        pbar.close()

        with open(fpath,'wb') as fp:
            pickle.dump(res,fp)

    x = [r[0] for r in res]
    y = [r[1] for r in res]

    data = {'category':x,'value':y}
    # sb.scatterplot(x=x,y=y)
    plt.figure(1),plt.clf()
    plt.subplot(1,2,1)
    sb.stripplot(x='category',y='value',data=data,jitter=True,size=1)
    plt.subplot(1,2,2)
    sb.boxplot(x='category',y='value',data=data)



if __name__ == "__main__":
    main()
