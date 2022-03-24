import sys
import argparse
import logging

from retinal.config.registry import Registry
from retinal.utils import mkdir, set_random_seed, setup_logging
from retinal.engine import load_config, SegmentTrainer, DRTrainer

logger = logging.getLogger(__name__)

TRAINER_REGISTRY = Registry("trainer")
TRAINER_REGISTRY.register("segment", SegmentTrainer)
TRAINER_REGISTRY.register("dr", DRTrainer)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Segmentation Pipeline')
    parser.add_argument("task", type=str, default=None,
                        choices=["segment", "dr"],
                        help="The target task")
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using command-line")

    return parser


def setup(args):
    cfg = load_config(args)
    mkdir(cfg.OUTPUT_DIR)
    setup_logging(output_dir=cfg.OUTPUT_DIR)
    set_random_seed(
        seed=None if cfg.RNG_SEED < 0 else cfg.RNG_SEED,
        deterministic=False if cfg.RNG_SEED < 0 else True
    )
    return cfg


def main():
    args = argument_parser().parse_args()
    cfg = setup(args)
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    trainer = TRAINER_REGISTRY.get(args.task)(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
