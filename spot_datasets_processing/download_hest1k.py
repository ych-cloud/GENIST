#!/usr/bin/env python
"""Download the HEST-1k subsets used by GENIST.

The downloader selects sample IDs from a pinned HEST metadata table and uses
the Hugging Face Hub snapshot API to retain the upstream directory layout.
By default, only the WSI and spatial-transcriptomics files required by the
GENIST spot pipeline are downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

try:
    from spot_datasets_processing.common import build_logger, default_dataset_root, ensure_directory
except ImportError:
    from common import build_logger, default_dataset_root, ensure_directory


HEST_REPOSITORY = "MahmoodLab/hest"
DEFAULT_METADATA_FILE = "HEST_v1_1_0.csv"
DEFAULT_COMPONENTS = ("wsis", "st")
AVAILABLE_SUBSETS = ("her2st", "kidney", "prad", "mouse_brain")
AVAILABLE_COMPONENTS = (
    "wsis",
    "st",
    "metadata",
    "spatial_plots",
    "thumbnails",
    "tissue_seg",
    "patches",
    "patches_vis",
    "pixel_size_vis",
    "transcripts",
    "xenium_seg",
    "cellvit_seg",
)

KIDNEY_TITLE_PREFIX = (
    "Spatial localization with Spatial Transcriptomics for an atlas of healthy "
    "and injured cell states"
)
HER2ST_TITLE_PREFIX = "Spatial deconvolution of"
MOUSE_BRAIN_TITLE_PREFIX = "Spatial Multimodal Analysis"
PRAD_SAMPLE_IDS = tuple(f"MEND{index}" for index in range(139, 163))
HER2ST_SLIDE_PATTERN = re.compile(r"\bSPA\d+\b", flags=re.IGNORECASE)


def import_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "HEST-1k download requires huggingface_hub. Install preprocessing "
            "dependencies with: pip install -e '.[preprocessing]'"
        ) from exc
    return hf_hub_download, snapshot_download


def read_metadata(
    output_dir: Path,
    metadata_file: str,
    revision: str,
    token: str | None,
    force_download: bool,
) -> tuple[pd.DataFrame, Path]:
    hf_hub_download, _ = import_huggingface_hub()
    metadata_path = Path(
        hf_hub_download(
            repo_id=HEST_REPOSITORY,
            repo_type="dataset",
            filename=metadata_file,
            revision=revision,
            local_dir=output_dir,
            token=token,
            force_download=force_download,
        )
    ).resolve()
    metadata = pd.read_csv(metadata_path)
    required_columns = {"id", "dataset_title", "disease_state"}
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(
            f"HEST metadata {metadata_path} is missing required columns: {missing_columns}"
        )
    metadata = metadata.copy()
    metadata["id"] = metadata["id"].astype(str).str.strip()
    if metadata["id"].duplicated().any():
        duplicates = metadata.loc[metadata["id"].duplicated(), "id"].tolist()
        raise ValueError(f"HEST metadata contains duplicated sample IDs: {duplicates[:10]}")
    return metadata, metadata_path


def subset_mask(metadata: pd.DataFrame, subset: str) -> pd.Series:
    titles = metadata["dataset_title"].fillna("").astype(str)
    disease_states = metadata["disease_state"].fillna("").astype(str)

    if subset == "her2st":
        return titles.str.startswith(HER2ST_TITLE_PREFIX)
    if subset == "kidney":
        return titles.str.startswith(KIDNEY_TITLE_PREFIX)
    if subset == "mouse_brain":
        return titles.str.startswith(MOUSE_BRAIN_TITLE_PREFIX) & disease_states.eq("Healthy")
    if subset == "prad":
        return metadata["id"].isin(PRAD_SAMPLE_IDS)
    raise ValueError(f"Unsupported HEST subset: {subset}")


def select_samples(
    metadata: pd.DataFrame,
    subsets: Sequence[str],
    explicit_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    sample_to_labels: dict[str, list[str]] = {}
    ordered_ids: list[str] = []

    if explicit_ids:
        requested_ids = [str(sample_id).strip() for sample_id in explicit_ids if str(sample_id).strip()]
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in set(metadata["id"])]
        if missing_ids:
            raise ValueError(f"Requested HEST sample IDs were not found: {missing_ids}")
        for sample_id in requested_ids:
            if sample_id not in sample_to_labels:
                ordered_ids.append(sample_id)
                sample_to_labels[sample_id] = ["custom"]
    else:
        for subset in subsets:
            matched_ids = metadata.loc[subset_mask(metadata, subset), "id"].tolist()
            if not matched_ids:
                raise ValueError(f"Subset '{subset}' did not match any samples in the HEST metadata.")
            for sample_id in matched_ids:
                if sample_id not in sample_to_labels:
                    ordered_ids.append(sample_id)
                    sample_to_labels[sample_id] = []
                sample_to_labels[sample_id].append(subset)

    selected = metadata.set_index("id").loc[ordered_ids].reset_index()
    selected.insert(
        1,
        "genist_subset",
        [",".join(sample_to_labels[sample_id]) for sample_id in ordered_ids],
    )
    selected.insert(
        2,
        "genist_slide_id",
        [
            resolve_genist_slide_id(row, sample_to_labels[str(row["id"])])
            for _, row in selected.iterrows()
        ],
    )
    return selected


def resolve_genist_slide_id(row: pd.Series, subset_labels: Sequence[str]) -> str:
    sample_id = str(row["id"])
    if "her2st" not in subset_labels or "subseries" not in row.index:
        return sample_id

    subseries = str(row.get("subseries", ""))
    match = HER2ST_SLIDE_PATTERN.search(subseries)
    return match.group(0).upper() if match is not None else sample_id


def build_download_patterns(sample_ids: Sequence[str], components: Sequence[str]) -> list[str]:
    patterns: list[str] = []
    for component in components:
        for sample_id in sample_ids:
            # Upstream files use either <id>.<suffix> or <id>_<suffix>.
            patterns.append(f"{component}/{sample_id}.*")
            patterns.append(f"{component}/{sample_id}_*")
    return patterns


def create_slide_aliases(
    output_dir: Path,
    selected_samples: pd.DataFrame,
    components: Sequence[str],
    logger,
) -> list[dict[str, str]]:
    aliases: list[dict[str, str]] = []
    for row in selected_samples.itertuples(index=False):
        source_id = str(row.id)
        slide_id = str(row.genist_slide_id)
        if source_id == slide_id:
            continue

        for component in components:
            component_dir = output_dir / component
            if not component_dir.exists():
                continue
            source_paths = sorted(
                set(component_dir.glob(f"{source_id}.*"))
                | set(component_dir.glob(f"{source_id}_*"))
            )
            for source_path in source_paths:
                target_path = component_dir / f"{slide_id}{source_path.name[len(source_id):]}"
                if target_path.exists():
                    if os.path.samefile(source_path, target_path):
                        continue
                    raise FileExistsError(
                        f"Cannot create slide alias because the target already exists: {target_path}"
                    )
                try:
                    os.link(source_path.resolve(), target_path)
                    link_type = "hardlink"
                except OSError:
                    try:
                        target_path.symlink_to(source_path.resolve())
                        link_type = "symlink"
                    except OSError as exc:
                        raise OSError(
                            f"Could not create alias {target_path} for {source_path}. "
                            "Use the HEST id from genist_hest_samples.csv or rerun with "
                            "--no-slide-aliases."
                        ) from exc
                aliases.append(
                    {
                        "source": str(source_path.resolve()),
                        "alias": str(target_path.resolve()),
                        "type": link_type,
                    }
                )
                logger.info("Created %s alias: %s -> %s", link_type, target_path.name, source_path.name)
    return aliases


def write_download_record(
    output_dir: Path,
    selected_samples: pd.DataFrame,
    metadata_path: Path,
    args: argparse.Namespace,
    patterns: Sequence[str],
) -> tuple[Path, Path]:
    manifest_path = output_dir / "genist_hest_samples.csv"
    record_path = output_dir / "genist_hest_download.json"
    selected_samples.to_csv(manifest_path, index=False, encoding="utf-8")
    record = {
        "repository": HEST_REPOSITORY,
        "revision": args.revision,
        "metadata_file": args.metadata_file,
        "metadata_path": str(metadata_path),
        "subsets": list(args.subset),
        "components": list(args.components),
        "sample_count": int(len(selected_samples)),
        "sample_ids": selected_samples["id"].tolist(),
        "genist_slide_ids": selected_samples["genist_slide_id"].tolist(),
        "allow_patterns": list(patterns),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with record_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path, record_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected HEST-1k spot datasets for GENIST."
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=AVAILABLE_SUBSETS,
        default=["her2st"],
        help="Paper subset(s) to download; defaults to HER2ST.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Optional explicit HEST IDs; when set, overrides --subset selection.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=AVAILABLE_COMPONENTS,
        default=list(DEFAULT_COMPONENTS),
        help="HEST components to download; defaults to wsis and st.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dataset_root(),
        help="Download directory. The upstream HEST folder layout is preserved.",
    )
    parser.add_argument(
        "--metadata-file",
        default=DEFAULT_METADATA_FILE,
        help="Pinned HEST metadata table used to define the paper subsets.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision, tag, or commit hash.",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing a Hugging Face access token.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent Hugging Face downloads.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files instead of reusing the local Hub cache.",
    )
    parser.add_argument(
        "--no-slide-aliases",
        action="store_true",
        help="Do not create SPA-style aliases for HER2ST files after download.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve metadata and write the sample manifest without downloading bulk data.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")

    logger = build_logger(args.verbose)
    output_dir = ensure_directory(args.output_dir)
    token = os.getenv(args.token_env) or None
    if token is None:
        logger.info(
            "No token found in %s; a token saved by `hf auth login` will be used if available.",
            args.token_env,
        )

    metadata, metadata_path = read_metadata(
        output_dir=output_dir,
        metadata_file=args.metadata_file,
        revision=args.revision,
        token=token,
        force_download=args.force_download,
    )
    selected_samples = select_samples(metadata, args.subset, args.sample_ids)
    sample_ids = selected_samples["id"].tolist()
    patterns = build_download_patterns(sample_ids, args.components)
    manifest_path, record_path = write_download_record(
        output_dir,
        selected_samples,
        metadata_path,
        args,
        patterns,
    )

    logger.info("Selected %d HEST samples: %s", len(sample_ids), ", ".join(sample_ids))
    logger.info("Sample manifest: %s", manifest_path)
    if args.dry_run:
        logger.info("Dry run complete; no WSI or expression files were downloaded.")
        return 0

    _, snapshot_download = import_huggingface_hub()
    try:
        snapshot_download(
            repo_id=HEST_REPOSITORY,
            repo_type="dataset",
            revision=args.revision,
            local_dir=output_dir,
            allow_patterns=patterns,
            token=token,
            max_workers=args.max_workers,
            force_download=args.force_download,
        )
    except Exception as exc:
        raise RuntimeError(
            "HEST-1k download failed. Confirm that you accepted the dataset terms at "
            "https://huggingface.co/datasets/MahmoodLab/hest and that your Hugging "
            f"Face token is available through {args.token_env} or `hf auth login`."
        ) from exc

    logger.info("HEST download complete: %s", output_dir)
    if not args.no_slide_aliases:
        aliases = create_slide_aliases(
            output_dir=output_dir,
            selected_samples=selected_samples,
            components=args.components,
            logger=logger,
        )
        logger.info("Created %d GENIST slide-name aliases.", len(aliases))
    logger.info("Download record: %s", record_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
