import json
import logging
import os
import shutil
from curses.ascii import isascii
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SpoofDataset(Dataset):
    def __init__(self, part: str, data_dir: str, limit = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_dir = Path(data_dir) / f"ASVspoof2019_LA_{part}" / "flac"
        protocols_file = Path(data_dir) / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{part}.trl.txt"
        if not data_dir.exists() or not protocols_file.exists():
            raise NotADirectoryError()

        self.index = []
        with open(protocols_file, 'r', encoding='utf-8') as f:
            a = list(map(lambda x: x.strip(), f.readlines()))
        
        for obj in tqdm(a, desc="loading index..."):
            name = obj.strip()[1]
            label = int(obj.strip()[-1] == "spoof")
            self.index.append(
                {
                    "path": str(data_dir / f"{name}.flac"),
                    "label": label
                }
            )
        
        if limit is not None:
            self.index = self.index[:limit]

    def __getitem__(self, index):
        data_dict = self.index[index]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)

        return {
            "audio": audio_wave,
            "audio_length": audio_wave.shape[-1],
            "label": data_dict["label"],
            "name": data_dict["path"].split('/')[-1]
        }

    def __len__(self):
        return len(self.index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor