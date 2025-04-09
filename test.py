# ptorch run prediction inference

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import platform
import os
import pickle
import pandas as pd

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

def build_dataset(cfg,decimate=0,onehot=False,transform=None,num_classes=2):
    dataset = nnUNet2dDataset(cfg.dataset.imgdir,cfg.dataset.lbldir,
                                    transform=None,
                                    decimate=decimate,
                                    in_memory=cfg.dataset.keep_in_memory,
                                    rgb=True,
                                    onehot=onehot,
                                    num_classes=num_classes)
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
    model.eval()

    # ---------------------------------------------------------------------------- #
    # Setup dataloaders
    # ---------------------------------------------------------------------------- #
    # test
    dataset_cfg = hydra.utils.instantiate(cfg.test_dataset)
    
    outputdir = os.sep+os.path.join(*cfg.test_dataset.model.resnet.ckpt_dir.split(os.sep)[:-2])
    fpath = os.path.join(outputdir,'results_'+cfg.tag+'.pkl')
    if os.path.exists(fpath):
        with open(fpath,'rb') as fp:
            res = pickle.load(fp)
    else:
        # test_transform = hydra.utils.instantiate(cfg.test_dataset.transforms)
        dataset = build_dataset(dataset_cfg,onehot=cfg.onehot)
        dataloader = DataLoader(
            dataset,
            **cfg.test_dataloader,
            worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(seed),
        )

        # ---------------------------------------------------------------------------- #
        # test loop
        # ---------------------------------------------------------------------------- #
    
        pbar = tqdm(total=len(dataloader))
        res = []
        for i,data in enumerate(dataloader):
            logits = model(**data)
            if cfg.onehot:
                probs = F.softmax(logits.data, dim=1)
            else:
                probs = F.sigmoid(logits.data)
            arg = torch.argmax(probs,dim=1)
            # pred = np.max(probs)
            lbl = data['lbl'].numpy()[0]
            if False:
                plt.imshow(data['img'][0][0])
            res.append((data['lbl'].numpy()[0],probs[0].numpy(),data['fdata']))
            pbar.update(1)
        pbar.close()

        with open(fpath,'wb') as fp:
            pickle.dump(res,fp)


    # one-hot results plots
    if cfg.onehot:
        labels = np.array([r[0] for r in res])
        predictions = np.array([r[1] for r in res])
        files = [r[2] for r in res]
        files = [[f[0] for f in tups] for tups in files]
        file1 = [f[0] for f in files]
        df = pd.DataFrame(predictions,columns=["0", "1"])
        df['Label'] = labels

        # quick check of trated T cases
        if False:
            df['File'] = file1
            print(df[(df['Label']==0) & (~df['File'].str.contains('u'))])

        # Stripplot
        df_melted = df.melt(id_vars=["Label"], var_name="Class", value_name="Probability")
        plt.figure(3,figsize=(5,4)),plt.clf()
        plt.subplot(1,2,1)
        sb.stripplot(x="Label", y="Probability", hue="Class", data=df_melted, jitter=True, dodge=True, s=1)

        plt.xlabel("Label")
        plt.ylabel("Probability")
        plt.title("Prediction Probabilities per Label")
        plt.legend(title="True Label")

        plt.subplot(1,2,2)
        sb.boxplot(x="Label",y='Probability',hue='Class', data=df_melted)
        plt.show()
 
    # mul-hot results plot
    else:
        labels = np.array([r[0] for r in res])
        predictions = np.array([r[1] for r in res])
        categories = ['T','RN']  # Category names
        dx_str = ['T/RN' if np.array_equal(row, [1,1]) else 'RN' for row in labels]

        df = pd.DataFrame({
            'dx':dx_str,"T": labels[:, 0], "RN": labels[:, 1],  
            "Pred T": predictions[:, 0], "Pred RN": predictions[:, 1]
        })


        df_melted = pd.melt(df, 
                            id_vars = ['dx'],
                            value_vars=["T", "RN"], 
                            var_name="category", 
                            value_name="Label")

        df_melted["Prediction"] = pd.melt(df,
                                          id_vars = ['dx'], 
                                        value_vars=["Pred T", "Pred RN"], 
                                        value_name="Prediction")["Prediction"]

        plt.figure(1),plt.clf()
        plt.subplot(1, 2, 1)
        # Stripplot with Separate Categories
        sb.stripplot(x="dx", y="Prediction", hue="category", data=df_melted, jitter=True, dodge=True, s=1)
        plt.xlabel("dx")
        # plt.ylabel("Prediction")
        plt.title("Predictions vs Labels (Per Sample)")
        plt.legend(title="category")

        plt.subplot(1,2,2),plt.cla()
        sb.boxplot(x='dx',y='Prediction', hue='category', data=df_melted)

        plt.show()
        a=1




if __name__ == "__main__":
    main()
