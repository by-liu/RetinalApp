import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from retinal.engine import DRTester

logger = logging.getLogger(__name__)

TESTERS = {
    "dr": DRTester,
}


@hydra.main(config_path="../configs", config_name="defaults")
def main(cfg: DictConfig):
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    with open_dict(cfg):
        cfg.work_dir = os.getcwd()
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    tester = TESTERS[cfg.task](cfg)
    tester.test()

    logger.info("Job complete !\n")


if __name__ == "__main__":
    main()
