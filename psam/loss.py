from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class Criterion(nn.Module):
    def __init__(self,pos_weight=None,onehot=False):
        super().__init__()
        if onehot:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, outputs: torch.Tensor, lbls):
        loss = self.criterion(outputs,lbls)

        return loss
