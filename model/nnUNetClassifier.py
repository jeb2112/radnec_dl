# pre-trained nnUNet encoder is paired with a dense layer for classification

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torkit3d.nn.functional import batch_index_select

from model.model import nnunet_encoder

class nnUNetClassifier(nn.Module):
    def __init__(
        self,
        encoder: nnunet_encoder
    ):
        super().__init__()
        self.nnunet_encoder = encoder
        self.output_size = 2

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1)),  # Global average pooling to (1) spatial size
            nn.Flatten(),                 # Flatten to (batch_size, num_channels)
            nn.Linear(512, 512),            
            nn.ReLU(),                     
            nn.Dropout(0.5),       
            nn.Linear(512, 1),    # Final classification layer
        )


    def forward(
        self,
        img: torch.Tensor,
        lbl: torch.tensor,
        is_eval: bool = False
    ) -> torch.tensor:

        mf = self.nnunet_encoder(img)[-1]
        mf = self.output_layer(mf)

        return mf

def main():

    # ckpt_dir="/media/jbishop/WD4/brainmets/sunnybrook/radnec2/nnUNet_results/Dataset139_RadNec/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_final.pth"

    encoder = nnunet_encoder(ckpt_dir)
    model = nnUNetClassifier(encoder).cuda()
    print("Model created.")

    points = torch.rand(2, 1024, 3).cuda() * 2 - 1
    point_colors = torch.rand(2, 1024, 3).cuda()
    gt_masks = torch.randint(0, 2, [2, 2, 1024]).bool().cuda()
    with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        outputs = model(points, point_colors, gt_masks)

    for i, output in enumerate(outputs):
        print(f"Iteration {i}")
        print(output["prompt_coords"].shape)
        print(output["prompt_labels"].shape)
        print(output["masks"].shape)
        print(output["iou_preds"].shape)



if __name__ == "__main__":
    main()
