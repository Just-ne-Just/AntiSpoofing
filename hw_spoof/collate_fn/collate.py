import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    audio = []
    name = []
    label = []

    for item in dataset_items:
        audio.append(item['audio'][0])
        name.append(item['name'])
        label.append(item['label'])
    
    audio = pad_sequence(audio, batch_first=True)
        
    return {
        "audio": audio,
        "name": name,
        "label": torch.Tensor(label)
    }