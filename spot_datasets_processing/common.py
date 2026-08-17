from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import anndata


DEFAULT_SLIDE_NAMES = tuple(f"SPA{i}" for i in range(119, 155))


@dataclass(frozen=True)
class Her2stPaths:
    dataset_root: Path
    wsi_dir: Path
    st_dir: Path

    @classmethod
    def from_dataset_root(cls, dataset_root: Path) -> "Her2stPaths":
        resolved_root = Path(dataset_root).resolve()
        return cls(
            dataset_root=resolved_root,
            wsi_dir=resolved_root / "wsis",
            st_dir=resolved_root / "st",
        )


def default_dataset_root() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "her2st"


def build_logger(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    return logging.getLogger("her2st_preprocess")


def ensure_directory(path: Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_slide_names(
    slides: Sequence[str] | None = None,
    slides_file: str | Path | None = None,
) -> list[str]:
    collected: list[str] = []

    if slides_file is not None:
        with Path(slides_file).expanduser().resolve().open("r", encoding="utf-8") as handle:
            collected.extend(
                line.strip()
                for line in handle
                if line.strip()
            )

    if slides:
        collected.extend(str(slide).strip() for slide in slides if str(slide).strip())

    if not collected:
        return list(DEFAULT_SLIDE_NAMES)

    unique_slides: list[str] = []
    seen: set[str] = set()
    for slide_name in collected:
        if slide_name not in seen:
            unique_slides.append(slide_name)
            seen.add(slide_name)
    return unique_slides


def write_lines(path: Path, lines: Iterable[str]) -> None:
    output_path = Path(path).resolve()
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def read_slide_adata(st_dir: Path, slide_name: str) -> anndata.AnnData:
    slide_path = Path(st_dir).resolve() / f"{slide_name}.h5ad"
    if not slide_path.exists():
        raise FileNotFoundError(f"Missing HER2ST slide file: {slide_path}")
    return anndata.read_h5ad(slide_path)


def compute_common_genes(
    st_dir: Path,
    slide_names: Sequence[str],
    logger: logging.Logger,
) -> list[str]:
    common_genes: set[str] | None = None

    for slide_name in slide_names:
        adata = read_slide_adata(st_dir, slide_name)
        slide_genes = set(map(str, adata.var_names))
        common_genes = slide_genes if common_genes is None else common_genes & slide_genes
        logger.info(
            "Loaded %s with shape %s; current common genes=%d",
            slide_name,
            adata.shape,
            0 if common_genes is None else len(common_genes),
        )

    return sorted(common_genes or [])


def load_aligned_adata(
    st_dir: Path,
    slide_name: str,
    common_genes: Sequence[str],
) -> anndata.AnnData:
    adata = read_slide_adata(st_dir, slide_name)
    return adata[:, list(common_genes)].copy()
