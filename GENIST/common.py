from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_TOPK_VALUES = (10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000, 6000)
MODE_ALIASES = {
    "spot": "spot",
    "single_cell": "single_cell",
    "single-cell": "single_cell",
    "singlecell": "single_cell",
    "cell": "single_cell",
}


def normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in MODE_ALIASES:
        raise ValueError(f"Unsupported mode: {mode}. Expected one of: {sorted(MODE_ALIASES)}")
    return MODE_ALIASES[normalized]


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def build_logger(
    log_dir: str | Path | None = None,
    verbose: bool = False,
    log_filename: str = "run.log",
) -> logging.Logger:
    logger = logging.getLogger("genist")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    while logger.handlers:
        handler = logger.handlers.pop()
        handler.close()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_dir = ensure_directory(log_dir)
        file_handler = logging.FileHandler(file_dir / log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def list_run_directories(experiment_root: str | Path) -> list[Path]:
    root = ensure_directory(experiment_root)
    indexed_runs: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = re.fullmatch(r"(?:run_)?(\d+)", path.name) if path.is_dir() else None
        if match is not None:
            indexed_runs.append((int(match.group(1)), path))
    return [path for _, path in sorted(indexed_runs)]


def next_run_directory(experiment_root: str | Path) -> Path:
    root = ensure_directory(experiment_root)
    existing_runs = list_run_directories(root)
    indices = [int(re.search(r"\d+$", path.name).group()) for path in existing_runs]
    run_dir = root / f"run_{(max(indices, default=-1) + 1):03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(make_json_serializable(payload), handle, indent=2, ensure_ascii=False)
    return output_path


def save_arguments(path: str | Path, args: Any, extra: dict[str, Any] | None = None) -> Path:
    payload = dict(vars(args))
    if extra:
        payload.update(extra)
    return write_json(path, payload)


def resolve_optional_path(base_dir: str | Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(base_dir).expanduser().resolve() / candidate).resolve()
