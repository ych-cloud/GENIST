from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from .common import normalize_mode, resolve_optional_path


SPOT_EMBEDDING_LAYOUT = {
    "uni": {
        "base_dir": "embeddings/spot/uni/original",
        "base_suffix": ".pt",
        "aug_dir": "embeddings/spot/uni/augmented",
        "aug_suffix": ".pt",
    },
    "conch": {
        "base_dir": "embeddings/spot/conch/original",
        "base_suffix": ".pt",
        "aug_dir": "embeddings/spot/conch/augmented",
        "aug_suffix": ".pt",
    },
}


class ExpressionConditionDataset(Dataset):
    def __init__(self, expressions: torch.Tensor, conditions: torch.Tensor) -> None:
        if expressions.shape[0] != conditions.shape[0]:
            raise ValueError(
                "Expression rows and condition rows do not match: "
                f"{expressions.shape[0]} vs {conditions.shape[0]}"
            )
        self.expressions = expressions
        self.conditions = conditions

    def __len__(self) -> int:
        return self.expressions.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expressions[index], self.conditions[index]


@dataclass
class DatasetBundle:
    dataset: Dataset | None
    input_gene_size: int
    condition_size: int
    gene_names: list[str]
    entity_ids: list[str]
    raw_conditions: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthBundle:
    matrix: np.ndarray
    gene_names: list[str]
    entity_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_feature_root(data_path: str | Path, feature_root: str | Path | None) -> Path:
    data_root = Path(data_path).expanduser().resolve()
    if feature_root is not None:
        return Path(feature_root).expanduser().resolve()
    return (data_root / "derived_features").resolve()


def parse_slide_names(slide_names: str | Sequence[str]) -> list[str]:
    if isinstance(slide_names, str):
        items = slide_names.split(",")
    else:
        items = list(slide_names)
    parsed = [str(item).strip() for item in items if str(item).strip()]
    if not parsed:
        raise ValueError("At least one slide name must be provided.")
    return parsed


def require_single_slide(slide_names: str | Sequence[str]) -> str:
    parsed = parse_slide_names(slide_names)
    if len(parsed) != 1:
        raise ValueError(f"Exactly one slide must be provided, received: {parsed}")
    return parsed[0]


def read_text_list(path: str | Path) -> list[str]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_gene_list(path: str | Path) -> list[str]:
    gene_path = Path(path).expanduser().resolve()
    if not gene_path.exists():
        raise FileNotFoundError(f"Gene list file not found: {gene_path}")
    gene_frame = pd.read_csv(gene_path, header=None)
    if gene_frame.empty:
        raise ValueError(f"Gene list file is empty: {gene_path}")
    genes = [str(value).strip() for value in gene_frame.iloc[:, 0].tolist() if str(value).strip()]
    if genes and genes[0].lower() == "gene":
        genes = genes[1:]
    if not genes:
        raise ValueError(f"Gene list file does not contain any valid genes: {gene_path}")
    return genes


def _to_dense_array(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _validate_gene_coverage(available_genes: Sequence[str], selected_genes: Sequence[str], source: str) -> None:
    missing = [gene for gene in selected_genes if gene not in available_genes]
    if missing:
        raise ValueError(
            f"{len(missing)} selected genes are missing in {source}. "
            f"Examples: {missing[:10]}"
        )


def _load_spot_expression_frame(st_dir: Path, slide_name: str, selected_genes: Sequence[str]) -> pd.DataFrame:
    adata = anndata.read_h5ad(st_dir / f"{slide_name}.h5ad")
    _validate_gene_coverage(adata.var_names.tolist(), selected_genes, f"{slide_name}.h5ad")
    matrix = _to_dense_array(adata[:, list(selected_genes)].X)
    return pd.DataFrame(
        matrix,
        columns=list(selected_genes),
        index=adata.obs_names.astype(str),
    )


def _load_spot_condition_matrix(
    processed_dir: Path,
    slide_name: str,
    encoder_names: Sequence[str],
    augmented: bool = False,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for encoder_name in dict.fromkeys(encoder_names):
        if encoder_name not in SPOT_EMBEDDING_LAYOUT:
            raise ValueError(
                f"Unsupported spot encoder '{encoder_name}'. Supported encoders: "
                f"{sorted(SPOT_EMBEDDING_LAYOUT)}"
            )
        layout = SPOT_EMBEDDING_LAYOUT[encoder_name]
        directory_name = layout["aug_dir"] if augmented else layout["base_dir"]
        suffix = layout["aug_suffix"] if augmented else layout["base_suffix"]
        tensor_path = processed_dir / directory_name / f"{slide_name}{suffix}"
        if not tensor_path.exists():
            raise FileNotFoundError(f"Missing embedding tensor: {tensor_path}")
        parts.append(torch.load(tensor_path, map_location="cpu").float())

    if not parts:
        raise ValueError("At least one spot condition source must be requested.")
    if len(parts) == 1:
        return parts[0]
    concat_dim = -1 if augmented else 1
    return torch.cat(parts, dim=concat_dim)


def _sample_augmented_features(augmented_tensor: torch.Tensor, num_aug_ratio: int) -> torch.Tensor:
    if num_aug_ratio <= 0:
        raise ValueError("num_aug_ratio must be a positive integer.")
    if augmented_tensor.ndim != 3:
        raise ValueError(
            f"Expected augmented embedding tensor with shape [N, T, D], got {tuple(augmented_tensor.shape)}"
        )
    num_entities, num_transforms, condition_size = augmented_tensor.shape
    if num_aug_ratio > num_transforms:
        raise ValueError(
            f"Requested {num_aug_ratio} augmentations per entity, but only {num_transforms} are available."
        )

    sampled = torch.empty((num_entities * num_aug_ratio, condition_size), dtype=augmented_tensor.dtype)
    for entity_index in range(num_entities):
        selected_indices = np.random.choice(num_transforms, size=num_aug_ratio, replace=False)
        start = entity_index * num_aug_ratio
        sampled[start : start + num_aug_ratio] = augmented_tensor[entity_index, selected_indices, :]
    return sampled


def _prepare_expression_tensor(expression_frame: pd.DataFrame) -> torch.Tensor:
    expression_values = np.log2(expression_frame.to_numpy(dtype=np.float32) + 1.0)
    return torch.from_numpy(expression_values).float()


def _filter_invalid_rows(
    expression_frame: pd.DataFrame,
    condition_tensor: torch.Tensor,
) -> tuple[pd.DataFrame, torch.Tensor]:
    invalid_mask = expression_frame.isna().all(axis=1) | (expression_frame.fillna(0).sum(axis=1) == 0)
    filtered_frame = expression_frame.loc[~invalid_mask].fillna(0.0).copy()
    filtered_conditions = condition_tensor[~invalid_mask.to_numpy()]
    return filtered_frame, filtered_conditions


def _build_spot_training_bundle(args, logger) -> DatasetBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    st_dir = data_root / "st"

    slide_list_path = resolve_optional_path(processed_dir, args.slide_list_file)
    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if slide_list_path is None or gene_list_path is None:
        raise ValueError("Spot training requires both --slide-list-file and --gene-list.")

    all_slides = read_text_list(slide_list_path)
    held_out_slides = set(parse_slide_names(args.slide_out))
    training_slides = [slide_name for slide_name in all_slides if slide_name not in held_out_slides]
    if not training_slides:
        raise ValueError("No training slides remain after excluding held-out slides.")

    selected_genes = load_gene_list(gene_list_path)
    logger.info("Spot mode: %d training slides, %d selected genes.", len(training_slides), len(selected_genes))

    base_expression_frames: list[pd.DataFrame] = []
    base_conditions: list[torch.Tensor] = []
    aug_expression_frames: list[pd.DataFrame] = []
    aug_conditions: list[torch.Tensor] = []

    for slide_name in training_slides:
        expression_frame = _load_spot_expression_frame(st_dir, slide_name, selected_genes)
        condition_tensor = _load_spot_condition_matrix(
            processed_dir=processed_dir,
            slide_name=slide_name,
            encoder_names=args.spot_condition_sources,
            augmented=False,
        )
        if condition_tensor.shape[0] != expression_frame.shape[0]:
            raise ValueError(
                f"Spot condition rows for {slide_name} do not match expression rows: "
                f"{condition_tensor.shape[0]} vs {expression_frame.shape[0]}"
            )

        base_expression_frames.append(expression_frame)
        base_conditions.append(condition_tensor)
        logger.info(
            "Loaded spot slide %s with %d entities and condition size %d.",
            slide_name,
            expression_frame.shape[0],
            condition_tensor.shape[1],
        )

        if args.num_aug_ratio > 0:
            augmented_tensor = _load_spot_condition_matrix(
                processed_dir=processed_dir,
                slide_name=slide_name,
                encoder_names=args.spot_condition_sources,
                augmented=True,
            )
            sampled_augmented = _sample_augmented_features(augmented_tensor, args.num_aug_ratio)
            aug_conditions.append(sampled_augmented)
            repeated_expression = pd.DataFrame(
                np.repeat(expression_frame.to_numpy(dtype=np.float32), args.num_aug_ratio, axis=0),
                columns=expression_frame.columns,
                index=[
                    f"{entity_id}|aug{aug_index}"
                    for entity_id in expression_frame.index
                    for aug_index in range(args.num_aug_ratio)
                ],
            )
            aug_expression_frames.append(repeated_expression)

    expression_frame = pd.concat(base_expression_frames + aug_expression_frames, axis=0)
    condition_tensor = torch.cat(base_conditions + aug_conditions, dim=0)
    expression_frame, condition_tensor = _filter_invalid_rows(expression_frame, condition_tensor)
    dataset = ExpressionConditionDataset(
        expressions=_prepare_expression_tensor(expression_frame),
        conditions=condition_tensor.float(),
    )
    return DatasetBundle(
        dataset=dataset,
        input_gene_size=len(selected_genes),
        condition_size=condition_tensor.shape[1],
        gene_names=selected_genes,
        entity_ids=expression_frame.index.astype(str).tolist(),
        metadata={
            "mode": "spot",
            "data_root": str(data_root),
            "feature_root": str(processed_dir),
            "training_slides": training_slides,
            "held_out_slides": sorted(held_out_slides),
            "spot_condition_sources": list(args.spot_condition_sources),
            "num_aug_ratio": int(args.num_aug_ratio),
        },
    )


def _load_single_cell_split_expression(
    split_dir: Path,
    selected_genes: Sequence[str],
) -> pd.DataFrame:
    expression_path = split_dir / "expression.csv"
    if not expression_path.exists():
        raise FileNotFoundError(f"Missing single-cell expression file: {expression_path}")
    expression_frame = pd.read_csv(expression_path, index_col=0)
    _validate_gene_coverage(expression_frame.columns.tolist(), selected_genes, str(expression_path))
    return expression_frame.loc[:, list(selected_genes)].copy()


def _load_single_cell_condition_matrix(
    split_dir: Path,
    slide_name: str,
    include_extra_condition: bool,
) -> torch.Tensor:
    uni_path = split_dir / "embeddings" / "single_cell" / "uni" / "original" / f"{slide_name}.pt"
    if not uni_path.exists():
        raise FileNotFoundError(f"Missing single-cell UNI embedding file: {uni_path}")
    condition_tensor = torch.load(uni_path, map_location="cpu").float()

    if include_extra_condition:
        extra_condition_path = split_dir / "extra_cond.npy"
        if not extra_condition_path.exists():
            raise FileNotFoundError(
                f"Missing extra condition file while --use-extra-condition is enabled: {extra_condition_path}"
            )
        extra_condition = torch.from_numpy(np.load(extra_condition_path).astype(np.float32))
        if extra_condition.shape[0] != condition_tensor.shape[0]:
            raise ValueError(
                "single-cell extra condition rows do not match UNI embedding rows: "
                f"{extra_condition.shape[0]} vs {condition_tensor.shape[0]}"
            )
        condition_tensor = torch.cat([condition_tensor, extra_condition], dim=1)

    return condition_tensor


def _load_single_cell_augmented_condition_matrix(
    split_dir: Path,
    slide_name: str,
    include_extra_condition: bool,
    num_aug_ratio: int,
) -> torch.Tensor:
    aug_path = split_dir / "embeddings" / "single_cell" / "uni" / "augmented" / f"{slide_name}.pt"
    if not aug_path.exists():
        raise FileNotFoundError(f"Missing single-cell augmented UNI embedding file: {aug_path}")
    augmented_condition = torch.load(aug_path, map_location="cpu").float()
    sampled_augmented = _sample_augmented_features(augmented_condition, num_aug_ratio)

    if include_extra_condition:
        extra_condition_path = split_dir / "extra_cond.npy"
        extra_condition = np.load(extra_condition_path).astype(np.float32)
        repeated_extra = torch.from_numpy(np.repeat(extra_condition, num_aug_ratio, axis=0)).float()
        sampled_augmented = torch.cat([sampled_augmented, repeated_extra], dim=1)

    return sampled_augmented


def _build_single_cell_training_bundle(args, logger) -> DatasetBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    split_dir = processed_dir / f"fold_{args.fold}" / "train"

    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if gene_list_path is None:
        raise ValueError("Single-cell training requires --gene-list.")

    selected_genes = load_gene_list(gene_list_path)
    expression_frame = _load_single_cell_split_expression(split_dir, selected_genes)
    condition_tensor = _load_single_cell_condition_matrix(
        split_dir=split_dir,
        slide_name=args.slide_name,
        include_extra_condition=args.use_extra_condition,
    )
    if condition_tensor.shape[0] != expression_frame.shape[0]:
        raise ValueError(
            "Single-cell condition rows do not match expression rows: "
            f"{condition_tensor.shape[0]} vs {expression_frame.shape[0]}"
        )

    expression_frames = [expression_frame]
    condition_tensors = [condition_tensor]

    if args.num_aug_ratio > 0:
        augmented_condition = _load_single_cell_augmented_condition_matrix(
            split_dir=split_dir,
            slide_name=args.slide_name,
            include_extra_condition=args.use_extra_condition,
            num_aug_ratio=args.num_aug_ratio,
        )
        condition_tensors.append(augmented_condition)
        augmented_expression = pd.DataFrame(
            np.repeat(expression_frame.to_numpy(dtype=np.float32), args.num_aug_ratio, axis=0),
            columns=expression_frame.columns,
            index=[
                f"{entity_id}|aug{aug_index}"
                for entity_id in expression_frame.index
                for aug_index in range(args.num_aug_ratio)
            ],
        )
        expression_frames.append(augmented_expression)

    merged_expression = pd.concat(expression_frames, axis=0)
    merged_conditions = torch.cat(condition_tensors, dim=0)
    merged_expression, merged_conditions = _filter_invalid_rows(merged_expression, merged_conditions)
    dataset = ExpressionConditionDataset(
        expressions=_prepare_expression_tensor(merged_expression),
        conditions=merged_conditions.float(),
    )

    logger.info(
        "Single-cell mode: fold=%s, slide=%s, entities=%d, genes=%d.",
        args.fold,
        args.slide_name,
        len(merged_expression),
        len(selected_genes),
    )
    return DatasetBundle(
        dataset=dataset,
        input_gene_size=len(selected_genes),
        condition_size=merged_conditions.shape[1],
        gene_names=selected_genes,
        entity_ids=merged_expression.index.astype(str).tolist(),
        metadata={
            "mode": "single_cell",
            "data_root": str(data_root),
            "feature_root": str(processed_dir),
            "fold": int(args.fold),
            "slide_name": args.slide_name,
            "use_extra_condition": bool(args.use_extra_condition),
            "num_aug_ratio": int(args.num_aug_ratio),
        },
    )


def build_training_bundle(args, logger) -> DatasetBundle:
    mode = normalize_mode(args.mode)
    if mode == "spot":
        return _build_spot_training_bundle(args, logger)
    return _build_single_cell_training_bundle(args, logger)


def _build_spot_sampling_bundle(args, logger) -> DatasetBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    st_dir = data_root / "st"
    slide_name = require_single_slide(args.slide_out)

    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if gene_list_path is None:
        raise ValueError("Spot sampling requires --gene-list.")

    adata = anndata.read_h5ad(st_dir / f"{slide_name}.h5ad")
    raw_conditions = _load_spot_condition_matrix(
        processed_dir=processed_dir,
        slide_name=slide_name,
        encoder_names=args.spot_condition_sources,
        augmented=False,
    )
    selected_genes = load_gene_list(gene_list_path)
    if raw_conditions.shape[0] != adata.shape[0]:
        raise ValueError(
            f"Spot sampling condition rows do not match hold-out entities: {raw_conditions.shape[0]} vs {adata.shape[0]}"
        )

    repeated_conditions = raw_conditions.repeat_interleave(args.samples_per_condition, dim=0)
    logger.info(
        "Spot sampling bundle ready for %s with %d entities and %d samples per entity.",
        slide_name,
        raw_conditions.shape[0],
        args.samples_per_condition,
    )
    return DatasetBundle(
        dataset=None,
        input_gene_size=len(selected_genes),
        condition_size=raw_conditions.shape[1],
        gene_names=selected_genes,
        entity_ids=adata.obs_names.astype(str).tolist(),
        raw_conditions=repeated_conditions.float(),
        metadata={
            "mode": "spot",
            "slide_name": slide_name,
            "samples_per_condition": int(args.samples_per_condition),
            "spot_condition_sources": list(args.spot_condition_sources),
        },
    )


def _build_single_cell_sampling_bundle(args, logger) -> DatasetBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    split_dir = processed_dir / f"fold_{args.fold}" / args.split
    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if gene_list_path is None:
        raise ValueError("Single-cell sampling requires --gene-list.")

    expression_frame = _load_single_cell_split_expression(split_dir, load_gene_list(gene_list_path))
    raw_conditions = _load_single_cell_condition_matrix(
        split_dir=split_dir,
        slide_name=args.slide_name,
        include_extra_condition=args.use_extra_condition,
    )
    if raw_conditions.shape[0] != expression_frame.shape[0]:
        raise ValueError(
            "Single-cell sampling condition rows do not match expression rows: "
            f"{raw_conditions.shape[0]} vs {expression_frame.shape[0]}"
        )

    repeated_conditions = raw_conditions.repeat_interleave(args.samples_per_condition, dim=0)
    logger.info(
        "Single-cell sampling bundle ready for fold=%s split=%s slide=%s with %d entities.",
        args.fold,
        args.split,
        args.slide_name,
        raw_conditions.shape[0],
    )
    return DatasetBundle(
        dataset=None,
        input_gene_size=expression_frame.shape[1],
        condition_size=raw_conditions.shape[1],
        gene_names=expression_frame.columns.astype(str).tolist(),
        entity_ids=expression_frame.index.astype(str).tolist(),
        raw_conditions=repeated_conditions.float(),
        metadata={
            "mode": "single_cell",
            "fold": int(args.fold),
            "split": args.split,
            "slide_name": args.slide_name,
            "samples_per_condition": int(args.samples_per_condition),
            "use_extra_condition": bool(args.use_extra_condition),
        },
    )


def build_sampling_bundle(args, logger) -> DatasetBundle:
    mode = normalize_mode(args.mode)
    if mode == "spot":
        return _build_spot_sampling_bundle(args, logger)
    return _build_single_cell_sampling_bundle(args, logger)


def _build_spot_ground_truth(args) -> GroundTruthBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    st_dir = data_root / "st"
    slide_name = require_single_slide(args.slide_out)
    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if gene_list_path is None:
        raise ValueError("Spot evaluation requires --gene-list.")

    selected_genes = load_gene_list(gene_list_path)
    expression_frame = _load_spot_expression_frame(st_dir, slide_name, selected_genes)
    matrix = np.log2(expression_frame.to_numpy(dtype=np.float32) + 1.0)
    return GroundTruthBundle(
        matrix=matrix,
        gene_names=selected_genes,
        entity_ids=expression_frame.index.astype(str).tolist(),
        metadata={"mode": "spot", "slide_name": slide_name},
    )


def _build_single_cell_ground_truth(args) -> GroundTruthBundle:
    data_root = Path(args.data_path).expanduser().resolve()
    processed_dir = resolve_feature_root(data_root, args.feature_root)
    split_dir = processed_dir / f"fold_{args.fold}" / args.split
    gene_list_path = resolve_optional_path(processed_dir, args.gene_list)
    if gene_list_path is None:
        raise ValueError("Single-cell evaluation requires --gene-list.")

    selected_genes = load_gene_list(gene_list_path)
    expression_frame = _load_single_cell_split_expression(split_dir, selected_genes)
    matrix = np.log2(expression_frame.to_numpy(dtype=np.float32) + 1.0)
    return GroundTruthBundle(
        matrix=matrix,
        gene_names=selected_genes,
        entity_ids=expression_frame.index.astype(str).tolist(),
        metadata={
            "mode": "single_cell",
            "fold": int(args.fold),
            "split": args.split,
            "slide_name": args.slide_name,
        },
    )


def load_ground_truth(args) -> GroundTruthBundle:
    mode = normalize_mode(args.mode)
    if mode == "spot":
        return _build_spot_ground_truth(args)
    return _build_single_cell_ground_truth(args)
