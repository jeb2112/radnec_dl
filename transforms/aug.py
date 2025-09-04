import torch
import kornia.augmentation as K
import torch.nn as nn

import torch
import torch.nn as nn

class BrightnessContrastJitter(nn.Module):
    def __init__(self, brightness=0.2, contrast=0.2, p=1.0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def forward(self, x):
        if torch.rand(1).item() > self.p:
            return x

        b = (torch.rand(1) * 2 - 1) * self.brightness + 1  # scale ~ [1-b, 1+b]
        c = (torch.rand(1) * 2 - 1) * self.contrast + 1    # scale ~ [1-c, 1+c]

        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) * c + mean * b


class KorniaAugmentations(nn.Module):
    def __init__(self,
                 flip_p=0.5,
                 rotation_degrees=15.0,
                 brightness=0.2,
                 contrast=0.2,
                    ):
        super().__init__()
        self.aug = nn.Sequential(
            K.RandomHorizontalFlip(p=flip_p),
            K.RandomRotation(degrees=rotation_degrees),
            BrightnessContrastJitter(
                brightness=brightness,
                contrast=contrast,
                p=1.0  # Always apply if specified
            )
        )

    def forward(self, x):
        return self.aug(x)