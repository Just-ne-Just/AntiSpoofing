from typing import List

import torch
from torch import Tensor

from hw_spoof.base.base_metric import BaseMetric
from hw_spoof.metric.utils import compute_eer

class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, prediction: Tensor, label: Tensor, **kwargs):
        bonafide_scores = prediction[label == 0]
        spoof_scores = prediction[label == 1]
        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        print(eer)
        return eer