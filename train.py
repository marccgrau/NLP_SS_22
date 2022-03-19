import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig

import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from pathlib import Path
import torch

EXPERIMENT_PROJECT = "NLP_SS_22"
ANNOTATION_FILE = "transcriptions_{}.json"

MAIN_DATA_DIR = Path("/data")
MODEL_DIR = MAIN_DATA_DIR.joinpath("models")
DATA_DIR = MAIN_DATA_DIR.joinpath("voice")

@hydra_runner(config_path="test", config_name="test")
def main(cfg):
    # logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # define directories & names
    source_config = cfg.source_config
    target_config = cfg.target_config
    tokenizer_name = cfg.tokenizer_name

    target_voice_dir = DATA_DIR.joinpath(target_config)

    train_manifest = target_voice_dir.joinpath(ANNOTATION_FILE.format("train"))
    val_manifest = target_voice_dir.joinpath(ANNOTATION_FILE.format("val"))
    test_manifest = target_voice_dir.joinpath(ANNOTATION_FILE.format("test"))

    tokenizer_dir = target_voice_dir.joinpath("tokenizers").joinpath(tokenizer_name)

    # set experiment name
    experiment_name = f"{cfg.name}-{source_config}-TO-{target_config}"

    # manually determine names and paths for data & tokenizer, setup names for experience manager
    cfg.model.train_ds.manifest_filepath = f"{train_manifest}"
    cfg.model.test_ds.manifest_filepath = f"{test_manifest}"
    cfg.model.validation_ds.manifest_filepath = f"{val_manifest}"

    if 'tokenizer' in cfg.model:
        cfg.model.tokenizer.dir = f"{tokenizer_dir}"

    cfg.exp_manager.exp_dir = f"experiments/"
    cfg.exp_manager.name = experiment_name
    cfg.exp_manager.create_wandb_logger = True
    cfg.exp_manager.wandb_logger_kwargs.name = experiment_name
    cfg.exp_manager.wandb_logger_kwargs.project = EXPERIMENT_PROJECT

    # instantiate trainer
    trainer = pl.Trainer(**cfg.trainer)
    logging.info(f'Trainer instantiated')
    # instantiate experience manager
    exp_manager(trainer, cfg.get("exp_manager", None))
    logging.info(f'Experience Manager instantiated')
    # load pretrained model determined by config name of model as ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
            cfg.name, map_location=torch.device("cpu")
        )
    logging.info(f'Model loaded')
    # set trainer for model from config trainer
    asr_model.set_trainer(trainer)
    logging.info(f'Trainer set')
    # set hydra config as model config
    asr_model.cfg = cfg
    # setup data loaders for training, testing & validation
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_validation_data(cfg.model.validation_ds)
    asr_model.setup_test_data(cfg.model.test_ds)
    logging.info(f'Data loaders set up')
    # setup the optimization procedure
    asr_model.setup_optimization(cfg.model.optim)
    logging.info(f'Optimizer set up')
    # check for validity of model config
    logging.info(f'Model config: {OmegaConf.to_yaml(asr_model.cfg)}')

    # train the model
    logging.info(f'Start training')
    trainer.fit(asr_model)
    logging.info(f'Training finished')

    # test model
    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)

    # save model
    model_save_path = MODEL_DIR.joinpath(
        f"{experiment_name}.nemo"
    )
    asr_model.save_to(f"{model_save_path}")
    logging.info(f"Saved trained model to '{model_save_path}")

if __name__ == "__main__":
    main()

