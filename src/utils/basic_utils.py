import os
import random
import string
from datetime import datetime
from typing import Any, Dict, Union

import torch
import yaml


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_run_name(
    model_name: str,
    dset_name: str,
    lr: float,
    batch_size: int,
    experiment_type: str = "",
    random_suffix: bool = True,
    random_suffix_len: int = 6,
) -> str:
    """Generate a unique run name with timestamp."""
    now = datetime.now().strftime("%m%d-%H%M")
    rand_suffx = (
        "".join(
            random.Random().choices(
                string.ascii_letters + string.digits, k=random_suffix_len
            )
        )
        if random_suffix
        else ""
    )
    return f"{model_name}_{dset_name}_{experiment_type}_lr{lr}_bs{batch_size}_{now}_{rand_suffx}"


def write_yaml(
    data: Union[Dict[str, Any], list],
    file_path: Union[str, os.PathLike],
    default_flow_style: bool = False,
) -> None:
    """
    Write a dictionary or list to a YAML file.

    Args:
        data: The data to write (usually a dictionary).
        file_path: The path to the output YAML file.
        default_flow_style: If False, uses block style for formatting.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=False,
            allow_unicode=True,
        )


def read_yaml(file_path: Union[str, os.PathLike]) -> Union[Dict[str, Any], list]:
    """
    Read a YAML file into a dictionary or list.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
