import argparse
import json
import pathlib

from .data_class import SenticASGCNTrainArgs


def parse_args_and_load_config(
    config_path: str = "config/sentic_asgcn_config.json",
) -> SenticASGCNTrainArgs:
    """Get config from config file using argparser

    Returns:
        SenticASGCNTrainArgs: SenticASGCNTrainArgs instance populated from config
    """
    parser = argparse.ArgumentParser(description="SenticASGCN Training")
    parser.add_argument("--config", type=str, default=config_path)
    args = parser.parse_args()

    cfg_path = pathlib.Path(__file__).parent / args.config
    with open(cfg_path, "r") as cfg_file:
        cfg = json.load(cfg_file)

    sentic_asgcn_args = SenticASGCNTrainArgs(**cfg)
    return sentic_asgcn_args
