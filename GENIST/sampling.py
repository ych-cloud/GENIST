from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from GENIST.ar_mask import get_attn_mask_single_seq, sample_random_ar_splits
from GENIST.diffusion import create_diffusion
from .common import build_logger, ensure_directory, normalize_mode, save_arguments, write_json
from .data import build_sampling_bundle


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    resolved_path = Path(checkpoint_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")
    checkpoint = torch.load(resolved_path, map_location=device)
    if "ema" in checkpoint:
        return checkpoint["ema"]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def default_output_dir(checkpoint_path: str | Path) -> Path:
    resolved = Path(checkpoint_path).expanduser().resolve()
    return ensure_directory(resolved.parent.parent / "predictions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample gene expression from a trained GENIST checkpoint.")
    parser.add_argument("--mode", type=str, required=True, help="Sampling mode: spot or single_cell.")
    parser.add_argument("--checkpoint", "--ckpt", dest="ckpt", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, default=Path("datasets/her2st"), help="Dataset root directory.")
    parser.add_argument("--feature-root", "--processed-dir", "--processed_dir", dest="feature_root", type=Path, default=None, help="Optional override for the derived feature directory.")
    parser.add_argument("--gene-order-file", "--gene-list", "--gene_list_filename", dest="gene_list", type=str, default=None, help="Ordered gene file filename or path.")
    parser.add_argument("--output-dir", "--save_path", dest="output_dir", type=Path, default=None, help="Directory where generated samples are stored.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Sampling device.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    parser.add_argument("--depth", "--dit-num-blocks", "--DiT_num_blocks", dest="depth", type=int, default=4, help="Transformer depth.")
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=384, help="Transformer hidden size.")
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--num-sampling-steps", "--num_sampling_steps", dest="num_sampling_steps", type=int, default=1000, help="Diffusion sampling steps.")
    parser.add_argument("--sampling-batch-size", "--sampling_batch_size", dest="sampling_batch_size", type=int, default=512, help="Sampling batch size.")
    parser.add_argument(
        "--samples-per-condition",
        "--sample-num-per-cond",
        dest="samples_per_condition",
        type=int,
        default=20,
        help="Number of generated samples per condition vector.",
    )
    parser.add_argument("--ar-num-steps-min", type=int, default=2, help="Minimum number of AR partitions.")
    parser.add_argument("--ar-num-steps-max", type=int, default=3, help="Maximum number of AR partitions; use -1 for no cap.")
    parser.add_argument("--ar-num-steps-decay", type=float, default=0.9, help="Decay used when sampling AR partition counts.")
    parser.add_argument(
        "--ar-clamp",
        dest="ar_clamp",
        action="store_true",
        default=True,
        help="Use AR clamped sampling. Enabled by default.",
    )
    parser.add_argument(
        "--no-ar-clamp",
        dest="ar_clamp",
        action="store_false",
        help="Disable AR clamped sampling and use a single generalized causal mask.",
    )

    parser.add_argument("--slide-out", "--slide_out", dest="slide_out", type=str, default="", help="Held-out spot slide for spot mode.")
    parser.add_argument(
        "--spot-condition-sources",
        nargs="+",
        default=("uni", "conch"),
        help="Spot condition tensors to concatenate. Supported: uni, conch",
    )

    parser.add_argument("--fold", type=int, default=0, help="Single-cell fold index.")
    parser.add_argument("--split", type=str, default="val", help="Single-cell split name.")
    parser.add_argument("--slide-name", type=str, default="", help="Single-cell slide/sample name.")
    parser.add_argument(
        "--use-extra-condition",
        action="store_true",
        help="Concatenate extra_cond.npy with UNI embeddings in single-cell mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.mode = normalize_mode(args.mode)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    output_dir = ensure_directory(args.output_dir or default_output_dir(args.ckpt))
    logger = build_logger(output_dir, args.verbose, log_filename="sampling.log")
    logger.info("Output directory: %s", output_dir)

    bundle = build_sampling_bundle(args, logger)
    if bundle.raw_conditions is None:
        raise ValueError("Sampling bundle does not provide any condition tensor.")

    from GENIST.models import GENIST

    model = GENIST(
        input_size=bundle.input_gene_size,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=bundle.condition_size,
    )
    model.load_state_dict(load_checkpoint(args.ckpt, device), strict=True)
    model.to(device)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    condition_loader = DataLoader(bundle.raw_conditions, batch_size=args.sampling_batch_size, shuffle=False)

    all_samples: list[torch.Tensor] = []
    for batch_index, condition_batch in enumerate(condition_loader, start=1):
        condition_batch = condition_batch.to(device)
        sequence_length = bundle.input_gene_size
        max_ar_steps = args.ar_num_steps_max if args.ar_num_steps_max > 0 else None
        split_sizes, cumulative_sizes = sample_random_ar_splits(
            sequence_length,
            decay=float(args.ar_num_steps_decay),
            min_steps=int(args.ar_num_steps_min),
            max_steps=max_ar_steps,
            device=device,
        )
        noise = torch.randn(condition_batch.shape[0], 1, sequence_length, device=device)
        model_kwargs = {"y": condition_batch}

        if args.ar_clamp:
            samples = diffusion.p_sample_loop_ar_clamped(
                model.forward,
                noise.shape,
                split_sizes=split_sizes,
                cond_len=0,
                noise=noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
        else:
            attention_mask = get_attn_mask_single_seq(
                sequence_length,
                split_sizes=split_sizes,
                cumsum=cumulative_sizes,
            ).bool().to(device)
            model_kwargs["attn_mask"] = attention_mask
            samples = diffusion.p_sample_loop(
                model.forward,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )

        all_samples.append(samples.detach().cpu())
        logger.info("Sampled batch %d/%d.", batch_index, len(condition_loader))

    stacked_samples = torch.cat(all_samples, dim=0)
    checkpoint_stem = Path(args.ckpt).expanduser().resolve().stem
    output_path = output_dir / f"expression_samples_{checkpoint_stem}_n{args.samples_per_condition}.pt"
    torch.save(stacked_samples, output_path)
    save_arguments(output_dir / "sampling_config.json", args, extra={"output_path": str(output_path)})
    write_json(output_dir / "sampling_metadata.json", bundle.metadata)
    logger.info("Saved samples to %s", output_path)


if __name__ == "__main__":
    main()
