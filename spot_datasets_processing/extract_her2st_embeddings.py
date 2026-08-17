#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from tqdm import tqdm

try:
    from spot_datasets_processing.common import (
        Her2stPaths,
        build_logger,
        default_dataset_root,
        ensure_directory,
        read_slide_adata,
        read_slide_names,
    )
except ImportError:
    from common import (
        Her2stPaths,
        build_logger,
        default_dataset_root,
        ensure_directory,
        read_slide_adata,
        read_slide_names,
    )


Image.MAX_IMAGE_PIXELS = None

ENCODER_OUTPUT_DIRS = {
    "conch": "embeddings/spot/conch/original",
    "uni": "embeddings/spot/uni/original",
    "conch_aug": "embeddings/spot/conch/augmented",
    "uni_aug": "embeddings/spot/uni/augmented",
}
SUPPORTED_ENCODERS = ("conch", "uni")
AUGMENT_TRANSFORMS = (
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


@dataclass
class EncoderBundle:
    name: str
    dim: int
    encode: Callable[[Image.Image], torch.Tensor]


def import_conch_factory():
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ModuleNotFoundError(
            "CONCH is an optional external dependency. Install the official package with "
            "`pip install git+https://github.com/mahmoodlab/CONCH.git`, request access to "
            "MahmoodLab/conch on Hugging Face, and provide --conch-weights."
        ) from exc
    return create_model_from_pretrained


def import_uni_encoder():
    try:
        from uni import get_encoder
    except ImportError as exc:
        raise ModuleNotFoundError(
            "UNI is an optional external dependency. Install the official package with "
            "`pip install git+https://github.com/mahmoodlab/UNI.git` and request access "
            "to MahmoodLab/UNI on Hugging Face."
        ) from exc
    return get_encoder


def load_conch_encoder(weights_path: Path, device: torch.device) -> EncoderBundle:
    create_model_from_pretrained = import_conch_factory()
    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        str(weights_path),
        device=device,
    )
    model.eval()

    def encode(patch: Image.Image) -> torch.Tensor:
        tensor = preprocess(
            patch.resize((256, 256), Image.Resampling.LANCZOS)
        ).unsqueeze(0)
        with torch.inference_mode():
            features = model.encode_image(
                tensor.to(device),
                proj_contrast=False,
                normalize=False,
            )
        return features.squeeze(0).detach().cpu()

    return EncoderBundle(name="conch", dim=512, encode=encode)


def load_uni_encoder(device: torch.device) -> EncoderBundle:
    get_encoder = import_uni_encoder()
    model, transform = get_encoder(enc_name="uni", device=device)
    model.eval()

    def encode(patch: Image.Image) -> torch.Tensor:
        tensor = transform(
            patch.resize((224, 224), Image.Resampling.LANCZOS)
        ).unsqueeze(0)
        with torch.inference_mode():
            features = model(tensor.to(device))
        return features.squeeze(0).detach().cpu()

    return EncoderBundle(name="uni", dim=1024, encode=encode)


def load_requested_encoders(
    encoder_names: Sequence[str],
    device: torch.device,
    conch_weights: Path | None,
) -> list[EncoderBundle]:
    encoders: list[EncoderBundle] = []
    for name in dict.fromkeys(encoder_names):
        if name == "conch":
            if conch_weights is None:
                raise ValueError(
                    "CONCH encoder requested but --conch-weights was not provided."
                )
            if not Path(conch_weights).expanduser().resolve().exists():
                raise FileNotFoundError(f"CONCH weights not found: {conch_weights}")
            encoders.append(load_conch_encoder(conch_weights, device))
        elif name == "uni":
            encoders.append(load_uni_encoder(device))
        else:
            raise ValueError(f"Unsupported encoder: {name}")
    return encoders


def get_spot_radius(adata) -> int:
    spot_diameter = float(
        adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    )
    return 112 if spot_diameter < 224 else int(spot_diameter // 2)


def crop_spot_patch(image: Image.Image, x_coord: float, y_coord: float, radius: int) -> Image.Image:
    left = int(round(x_coord - radius))
    upper = int(round(y_coord - radius))
    right = int(round(x_coord + radius))
    lower = int(round(y_coord + radius))
    return image.crop((left, upper, right, lower))


def encode_augmented_patch(
    patch: Image.Image,
    encoder: EncoderBundle,
    num_augments: int,
) -> torch.Tensor:
    if num_augments < 1 or num_augments > len(AUGMENT_TRANSFORMS):
        raise ValueError(
            "num_augments must be between "
            f"1 and {len(AUGMENT_TRANSFORMS)}, got {num_augments}"
        )
    augmented = [
        encoder.encode(patch.transpose(transform))
        for transform in AUGMENT_TRANSFORMS[:num_augments]
    ]
    return torch.stack(augmented, dim=0)


def extract_slide_embeddings(
    image: Image.Image,
    adata,
    encoders: Sequence[EncoderBundle],
    num_augments: int,
    slide_name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    coords = adata.obsm["spatial"]
    if len(coords) == 0:
        raise ValueError(f"Slide {slide_name} contains no spatial spots.")

    radius = get_spot_radius(adata)
    base_features: dict[str, list[torch.Tensor]] = {encoder.name: [] for encoder in encoders}
    aug_features: dict[str, list[torch.Tensor]] = {
        f"{encoder.name}_aug": [] for encoder in encoders
    }

    for x_coord, y_coord in tqdm(coords, desc=f"{slide_name} spots", leave=False):
        patch = crop_spot_patch(image, x_coord, y_coord, radius)
        for encoder in encoders:
            base_features[encoder.name].append(encoder.encode(patch))
            aug_features[f"{encoder.name}_aug"].append(
                encode_augmented_patch(patch, encoder, num_augments)
            )

    base_tensors = {
        name: torch.stack(features, dim=0)
        for name, features in base_features.items()
    }
    aug_tensors = {
        name: torch.stack(features, dim=0)
        for name, features in aug_features.items()
    }
    return base_tensors, aug_tensors


def expected_output_paths(
    output_root: Path,
    slide_name: str,
    encoder_names: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for name in encoder_names:
        paths.append(output_root / ENCODER_OUTPUT_DIRS[name] / f"{slide_name}.pt")
        paths.append(
            output_root / ENCODER_OUTPUT_DIRS[f"{name}_aug"] / f"{slide_name}.pt"
        )
    return paths


def save_embeddings(
    output_root: Path,
    slide_name: str,
    base_tensors: dict[str, torch.Tensor],
    aug_tensors: dict[str, torch.Tensor],
) -> None:
    for name, tensor in base_tensors.items():
        output_dir = ensure_directory(output_root / ENCODER_OUTPUT_DIRS[name])
        torch.save(tensor.cpu(), output_dir / f"{slide_name}.pt")

    for name, tensor in aug_tensors.items():
        output_dir = ensure_directory(output_root / ENCODER_OUTPUT_DIRS[name])
        torch.save(tensor.cpu(), output_dir / f"{slide_name}.pt")


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract HER2ST spot image embeddings."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root(),
        help="HER2ST dataset root containing wsis/ and st/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory used to store extracted embeddings.",
    )
    parser.add_argument(
        "--slides",
        nargs="*",
        default=None,
        help="Optional slide names, for example: SPA119 SPA120.",
    )
    parser.add_argument(
        "--slides-file",
        type=Path,
        default=None,
        help="Optional text file containing one slide name per line.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=SUPPORTED_ENCODERS,
        default=list(SUPPORTED_ENCODERS),
        help="Foundation models used for embedding extraction.",
    )
    parser.add_argument(
        "--conch-weights",
        type=Path,
        default=Path(os.environ["CONCH_WEIGHTS"]) if "CONCH_WEIGHTS" in os.environ else None,
        help="Path to the CONCH pretrained weights file.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Falls back to cpu if CUDA is unavailable.",
    )
    parser.add_argument(
        "--num-augments",
        type=int,
        default=7,
        help="Number of transpose-based augmentations to save per spot.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a slide when all requested outputs already exist.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = build_logger(args.verbose)

    paths = Her2stPaths.from_dataset_root(args.dataset_root)
    output_root = ensure_directory(args.output_root or (paths.dataset_root / "derived_features"))
    slide_names = read_slide_names(args.slides, args.slides_file)
    device = choose_device(args.device)
    if args.device.startswith("cuda") and device.type != "cuda":
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")

    requested_encoder_names = list(dict.fromkeys(args.encoders))
    encoders = load_requested_encoders(requested_encoder_names, device, args.conch_weights)
    logger.info(
        "Loaded encoders=%s on device=%s for %d slides",
        ",".join(encoder.name for encoder in encoders),
        device,
        len(slide_names),
    )

    for slide_name in slide_names:
        output_paths = expected_output_paths(output_root, slide_name, requested_encoder_names)
        if args.skip_existing and all(path.exists() for path in output_paths):
            logger.info("Skipping %s because all outputs already exist.", slide_name)
            continue

        adata = read_slide_adata(paths.st_dir, slide_name)
        image_path = paths.wsi_dir / f"{slide_name}.tif"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing HER2ST WSI file: {image_path}")

        logger.info("Processing %s with %d spots", slide_name, adata.n_obs)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            base_tensors, aug_tensors = extract_slide_embeddings(
                rgb_image,
                adata,
                encoders,
                args.num_augments,
                slide_name,
            )

        save_embeddings(output_root, slide_name, base_tensors, aug_tensors)
        for name, tensor in base_tensors.items():
            logger.info("%s %s shape=%s", slide_name, name, tuple(tensor.shape))
        for name, tensor in aug_tensors.items():
            logger.info("%s %s shape=%s", slide_name, name, tuple(tensor.shape))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Embedding extraction completed. Outputs saved to %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
