import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner
from nemo.utils import logging

from nemo.utils.exp_manager import exp_manager

from pathlib import Path
import torch

# offline inference packages
import ctc_decoders
from plotly import graph_objects as go
import numpy as np
import librosa

EXPERIMENT_PROJECT = "NLP_SS_22"
ANNOTATION_FILE = "transcriptions_{}.json"

MAIN_DATA_DIR = Path("/data")
MODEL_DIR = MAIN_DATA_DIR.joinpath("models")
DATA_DIR = MAIN_DATA_DIR.joinpath("voice")

@hydra_runner(config_path="test", config_name="test")
def main(cfg):

    # load pretrained model determined by config name of model as ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.name, map_location=torch.device("cpu")
        )
    logging.info(f'Model loaded')



if __name__ == "__main__":
    main()

