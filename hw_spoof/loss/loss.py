import torch
from torch import Tensor
import torch.nn as nn

class CELoss(nn.Module):
    def __init__(self, spoof_scale=1, bonafide_scale=9):
        super().__init__()
        self.spoof_scale = spoof_scale
        self.bonafide_scale = bonafide_scale
          
    def forward(self,
                prediction,
                label,
                **kwargs) -> Tensor:

        return nn.functional.cross_entropy(prediction, label, weight=torch.Tensor([self.bonafide_scale, self.spoof_scale]).to(prediction.device))

