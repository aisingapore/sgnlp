import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .data_class import BaseArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def check_path_exists(file_type: str, file_path: Path) -> None:
    """Exit if file path does not exist."""
    if not Path.exists(file_path):
        raise FileNotFoundError(f"{file_type} file not found at {file_path}")


def load_stance_classification_config() -> BaseArguments:
    """Load arguments in the stance classification configuration file."""
    return _parse_args_and_load_config("config/stance_classification_config.json")


def load_rumour_verification_config() -> BaseArguments:
    """Load arguments in the rumour verification configuration file."""
    return _parse_args_and_load_config("config/rumour_verification_config.json")


def _parse_args_and_load_config(config_path: str) -> BaseArguments:
    """Load arguments in the configuration file.

    Args:
        config_path (str): Path of the configuration file.

    Returns:
        BaseArguments: Arguments of the configuration file.
    """
    with open(Path(__file__).parent / config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return BaseArguments(**cfg)


def set_device_and_seed(
    no_cuda: bool = False,
    seed: int = 64,
) -> Dict[str, Any]:
    """Set seeds and store configuration related to CUDA usage and distributed training.

    Args:
        no_cuda (bool, optional): Whether to not use CUDA even when it is available or not. Defaults to False.
        seed (int, optional): Random seed for initialization. Defaults to 64.

    Returns:
        Dict[str, Any]: Configuration related to CUDA usage and distributed training.
    """
    env: Dict[str, Any] = {}

    env["local_rank"] = int(os.getenv("LOCAL_RANK", -1))

    if env["local_rank"] == -1 or no_cuda:
        env["device"] = torch.device(
            "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
        )
        env["n_gpu"] = torch.cuda.device_count()
    else:
        torch.cuda.set_device(env["local_rank"])
        env["device"] = torch.device("cuda", env["local_rank"])
        env["n_gpu"] = 1
        torch.distributed.init_process_group(backend="nccl")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env["n_gpu"] > 0:
        torch.cuda.manual_seed_all(seed)

    return env
