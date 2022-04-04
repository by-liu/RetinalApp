import os
import sys
import argparse
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from retinal.config.registry import Registry
from retinal.utils import mkdir, set_random_seed, setup_logging
from retinal.engine import DRTrainer

logger = logging.getLogger(__name__)

TRAINERS = {
    "dr": DRTrainer
}


@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    with open_dict(cfg):
        cfg.work_dir = os.getcwd()
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    set_random_seed(
        cfg.seed if cfg.seed is not None else None,
        deterministic=True if cfg.seed is not None else False
    )

    trainer = TRAINERS[cfg.task](cfg)
    trainer.run()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
