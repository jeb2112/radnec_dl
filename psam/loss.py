from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, lbls):
        # criterion=torch.nn.CrossEntropyLoss()
        criterion=torch.nn.BCEWithLogitsLoss()
        loss = criterion(outputs,lbls)

        return loss
