import sys
import logging
import argparse

from retinal.config.registry import Registry
from retinal.utils import mkdir, setup_logging
from retinal.engine import load_config, ImageFolderTester, DRFolderTester

logger = logging.getLogger(__name__)

TESTER_REGISTRY = Registry("folder_tester")
TESTER_REGISTRY.register("segment", ImageFolderTester)
TESTER_REGISTRY.register("dr", DRFolderTester)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Segmentation Pipeline')
    parser.add_argument("--task", type=str, default="segment",
                        choices=["segment", "dr"],
                        help="The target task")
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--image-path", default="", metavar="FILE",
                        help="path of images")
    parser.add_argument("--save-path", default="", metavar="FILE",
                        help="save path of prediction masks")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using command-line")

    args = parser.parse_args()

    return args


def setup(args):
    cfg = load_config(args)
    cfg.DATA.NAME = "image-folder"
    cfg.DATA.DATA_ROOT = args.image_path
    mkdir(cfg.OUTPUT_DIR)
    setup_logging(output_dir=cfg.OUTPUT_DIR)
    return cfg


def main():
    args = parse_args()
    cfg = setup(args)
    logger.info("Launch command : ")
    logger.info(" ".join(sys.argv))
    tester = TESTER_REGISTRY.get(args.task)(cfg, args.save_path)
    tester.test()


if __name__ == "__main__":
    main()
