import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_spoof.model as module_model

from hw_spoof.trainer import Trainer
from hw_spoof.utils import ROOT_PATH
from hw_spoof.utils.object_loading import get_dataloaders
from hw_spoof.utils import MetricTracker
from hw_spoof.utils.parse_config import ConfigParser
import shutil
import torchaudio

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
sr = 16000

def main(config, input_dir, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    print(f"DEVICE: {device}")
    model = model.to(device)
    model.eval()

    with open(out_file, 'a', encoding='utf-8') as f:    
        for file in os.listdir(input_dir):
            audio = torchaudio.load(f"{input_dir}/{file}")[0].unsqueeze(0).to(device)
            out = model(audio)
            prob = torch.softmax(out["prediction"], dim=-1)
            f.write(f"{file}: bona-{prob[0][0].item()} spoof-{prob[0][1].item()}\n")
    



if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.txt",
        type=str,
        help="output file",
    )
    args.add_argument(
        "-i",
        "--input",
        default="./test_data",
        type=str,
        help="path to input dir",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set

    main(config, args.input, args.output)