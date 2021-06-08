import sys
import argparse
import logging

from retinal.config.registry import Registry
from retinal.utils import mkdir, setup_logging
from retinal.engine import load_config, SegmentTester

logger = logging.getLogger(__name__)

TESTER_REGISTRY = Registry("tester")
TESTER_REGISTRY.register("segment", SegmentTester)


def argument_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Segmentation Pipeline')
    parser.add_argument("task", type=str, default=None,
                        choices=["segment"],
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
    return cfg


def main():
    args = argument_parser().parse_args()
    cfg = setup(args)
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    tester = TESTER_REGISTRY.get(args.task)(cfg)
    tester.test()


if __name__ == "__main__":
    main()
