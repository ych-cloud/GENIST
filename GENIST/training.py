from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from GENIST.ar_mask import get_attn_mask_at_step, sample_random_ar_splits
from GENIST.diffusion import create_diffusion
from GENIST.train_helper import requires_grad, update_ema

from .common import (
    build_logger,
    ensure_directory,
    list_run_directories,
    next_run_directory,
    normalize_mode,
    save_arguments,
    set_random_seed,
    write_json,
)
from .data import DatasetBundle, build_training_bundle


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GenistTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        args: argparse.Namespace,
        run_dir: Path,
        logger,
    ) -> None:
        self.model = model.to(device)
        self.ema = deepcopy(model).to(device)
        requires_grad(self.ema, False)
        update_ema(self.ema, self.model, decay=0)

        self.train_loader = train_loader
        self.device = device
        self.args = args
        self.logger = logger
        self.run_dir = run_dir
        self.checkpoint_dir = ensure_directory(run_dir / "checkpoints")

        self.diffusion = create_diffusion(timestep_respacing="")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=0.0)
        self.train_steps = 0
        self.log_steps = 0
        self.running_loss = 0.0

        logger.info("Model parameters: %s", f"{sum(parameter.numel() for parameter in model.parameters()):,}")

    def _run_batch(self, expressions: torch.Tensor, timesteps: torch.Tensor, model_kwargs: dict[str, torch.Tensor], noise=None) -> None:
        loss_mask = model_kwargs.pop("loss_mask", None)
        loss_weight = model_kwargs.pop("loss_weight", None)
        loss_dict = self.diffusion.training_losses(
            self.model,
            expressions,
            timesteps,
            model_kwargs,
            noise=noise,
            loss_mask=loss_mask,
            loss_weight=loss_weight,
        )
        loss = loss_dict["loss"].mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema(self.ema, self.model)

        self.running_loss += float(loss.item())
        self.train_steps += 1
        self.log_steps += 1

        if self.log_steps % self.args.log_every == 0:
            if expressions.is_cuda:
                torch.cuda.synchronize()
            average_loss = self.running_loss / self.log_steps
            self.logger.info("step=%07d train_loss=%.6f", self.train_steps, average_loss)
            self.running_loss = 0.0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
            self.save_checkpoint()

    def _run_epoch(self, epoch_index: int) -> None:
        batch_size = self.train_loader.batch_size or 0
        self.logger.info("epoch=%03d batch_size=%d num_batches=%d", epoch_index, batch_size, len(self.train_loader))

        for expressions, conditions in self.train_loader:
            expressions = expressions.unsqueeze(1).to(self.device)
            conditions = conditions.to(self.device)
            timesteps = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (expressions.size(0),),
                device=self.device,
            )

            sequence_length = expressions.shape[-1]
            max_ar_steps = self.args.ar_num_steps_max if self.args.ar_num_steps_max > 0 else None
            split_sizes, cumulative_sizes = sample_random_ar_splits(
                sequence_length,
                decay=float(self.args.ar_num_steps_decay),
                min_steps=int(self.args.ar_num_steps_min),
                max_steps=max_ar_steps,
                device=self.device,
            )
            num_splits = len(split_sizes)
            step_decay = float(self.args.ar_step_decay)
            if step_decay <= 0:
                step_decay = 1.0
            split_indices = torch.arange(num_splits, device=self.device, dtype=torch.float32)
            if abs(step_decay - 1.0) < 1e-8:
                probabilities = torch.ones(num_splits, device=self.device) / num_splits
            else:
                weights = torch.pow(torch.tensor(step_decay, device=self.device), split_indices)
                probabilities = weights / weights.sum()

            step_index = int(torch.multinomial(probabilities, num_samples=1).item())
            attention_mask = get_attn_mask_at_step(
                sequence_length,
                split_sizes=split_sizes,
                cumsum=cumulative_sizes,
                step_index=step_index,
            ).bool().to(self.device)

            start_index = cumulative_sizes[step_index]
            current_block_size = split_sizes[step_index]

            noise = torch.zeros_like(expressions)
            noise[:, :, start_index : start_index + current_block_size] = torch.randn_like(
                expressions[:, :, start_index : start_index + current_block_size]
            )

            loss_mask = torch.zeros_like(expressions)
            loss_mask[:, :, start_index : start_index + current_block_size] = 1.0

            if self.args.ar_weight_schedule == "cosine":
                loss_weight = (
                    math.cos(math.pi * start_index / sequence_length)
                    * (self.args.ar_weight_max - self.args.ar_weight_min)
                    / 2.0
                    + (self.args.ar_weight_max + self.args.ar_weight_min) / 2.0
                )
            else:
                loss_weight = self.args.ar_weight_max - (
                    (self.args.ar_weight_max - self.args.ar_weight_min) * (start_index / sequence_length)
                )

            self._run_batch(
                expressions,
                timesteps,
                {
                    "y": conditions,
                    "attn_mask": attention_mask,
                    "loss_mask": loss_mask,
                    "loss_weight": loss_weight,
                },
                noise=noise,
            )

    def save_checkpoint(self) -> Path:
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.train_steps:07d}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizer.state_dict(),
                "train_steps": self.train_steps,
            },
            checkpoint_path,
        )
        self.logger.info("Saved checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def resume_from(self, checkpoint_path: str | Path, strict: bool = False) -> None:
        checkpoint = torch.load(Path(checkpoint_path).expanduser().resolve(), map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        self.ema.load_state_dict(checkpoint["ema"], strict=strict)
        if "opt" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["opt"])
            except Exception as exc:
                self.logger.warning("Optimizer state could not be restored: %s", exc)
        self.train_steps = int(checkpoint.get("train_steps", self.train_steps))
        self.logger.info("Resumed from %s", checkpoint_path)

    def train(self, total_epochs: int) -> None:
        self.model.train()
        self.ema.eval()
        for epoch_index in range(total_epochs):
            self._run_epoch(epoch_index)
        self.save_checkpoint()


def build_model(bundle: DatasetBundle, args: argparse.Namespace) -> torch.nn.Module:
    from GENIST.models import GENIST

    return GENIST(
        input_size=bundle.input_gene_size,
        depth=args.depth,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        label_size=bundle.condition_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GENIST on spot-level or single-cell gene expression data.")
    parser.add_argument("--mode", type=str, required=True, help="Training mode: spot or single_cell.")
    parser.add_argument("--data-path", "--data_path", dest="data_path", type=Path, default=Path("datasets/her2st"), help="Dataset root directory.")
    parser.add_argument("--feature-root", "--processed-dir", "--processed_dir", dest="feature_root", type=Path, default=None, help="Optional override for the derived feature directory.")
    parser.add_argument(
        "--gene-order-file",
        "--gene-list",
        "--gene_list_filename",
        dest="gene_list",
        type=str,
        default=None,
        help="Ordered gene file filename or path.",
    )
    parser.add_argument(
        "--experiment-root",
        "--results-dir",
        "--results_dir",
        dest="experiment_root",
        type=Path,
        default=Path("experiments"),
        help="Root directory used to store experiment runs.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device, for example cuda:0 or cpu.")
    parser.add_argument("--seed", "--global-seed", dest="seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--total-epochs", "--epochs", "--total_epochs", dest="total_epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--global-batch-size", "--batch_size", "--global_batch_size", dest="global_batch_size", type=int, default=512, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--ckpt-every", "--ckpt_every", dest="ckpt_every", type=int, default=30000, help="Checkpoint interval in optimizer steps.")
    parser.add_argument("--log-every", type=int, default=500, help="Log interval in optimizer steps.")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path or 'latest' to resume training.")
    parser.add_argument("--resume-strict", action="store_true", help="Load checkpoints with strict state-dict matching.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    parser.add_argument("--depth", "--dit-num-blocks", "--DiT_num_blocks", dest="depth", type=int, default=4, help="Transformer depth.")
    parser.add_argument("--hidden-size", "--hidden_size", dest="hidden_size", type=int, default=384, help="Transformer hidden size.")
    parser.add_argument("--num-heads", "--num_heads", dest="num_heads", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--ar-num-steps-min", type=int, default=4, help="Minimum number of AR partitions.")
    parser.add_argument("--ar-num-steps-max", type=int, default=8, help="Maximum number of AR partitions; use -1 for no cap.")
    parser.add_argument("--ar-num-steps-decay", type=float, default=0.9, help="Decay used when sampling AR partition counts.")
    parser.add_argument("--ar-step-decay", type=float, default=0.9, help="Decay used when sampling the current AR step.")
    parser.add_argument("--ar-weight-max", type=float, default=2.0, help="Maximum loss weight for early AR steps.")
    parser.add_argument("--ar-weight-min", type=float, default=1.0, help="Minimum loss weight for late AR steps.")
    parser.add_argument(
        "--ar-weight-schedule",
        choices=("linear", "cosine"),
        default="linear",
        help="Schedule used to map AR step position to loss weight.",
    )
    parser.add_argument("--slide-out", "--slide_out", dest="slide_out", type=str, default="", help="Held-out spot slide(s), comma-separated for spot mode.")
    parser.add_argument(
        "--sample-id-file",
        "--slide-list-file",
        "--folder_list_filename",
        dest="slide_list_file",
        type=str,
        default="samples_her2st.txt",
        help="Sample identifier file for spot mode.",
    )
    parser.add_argument(
        "--spot-condition-sources",
        nargs="+",
        default=("uni", "conch"),
        help="Spot condition tensors to concatenate. Supported: uni, conch.",
    )
    parser.add_argument("--num-aug-ratio", type=int, default=0, help="Number of augmentations sampled per entity.")

    parser.add_argument("--fold", type=int, default=0, help="Single-cell fold index.")
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

    requested_device = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    set_random_seed(args.seed)

    run_root = ensure_directory(args.experiment_root / args.mode)
    if args.resume == "latest":
        existing_runs = list_run_directories(run_root)
        run_dir = existing_runs[-1] if existing_runs else next_run_directory(run_root)
    else:
        run_dir = next_run_directory(run_root)
    ensure_directory(run_dir / "predictions")
    logger = build_logger(run_dir, args.verbose, log_filename="training.log")
    logger.info("Run directory: %s", run_dir)

    bundle = build_training_bundle(args, logger)
    write_json(run_dir / "dataset_manifest.json", bundle.metadata)
    save_arguments(run_dir / "run_config.json", args, extra={"run_dir": str(run_dir)})

    model = build_model(bundle, args)
    train_loader = DataLoader(
        bundle.dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    trainer = GenistTrainer(
        model=model,
        train_loader=train_loader,
        device=device,
        args=args,
        run_dir=run_dir,
        logger=logger,
    )

    if args.resume:
        if args.resume == "latest":
            checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
            if checkpoints:
                trainer.resume_from(checkpoints[-1], strict=args.resume_strict)
            else:
                logger.info("No checkpoints found in %s. Training starts from scratch.", run_dir / "checkpoints")
        else:
            trainer.resume_from(args.resume, strict=args.resume_strict)

    logger.info("Starting training on %s mode.", args.mode)
    trainer.train(args.total_epochs)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
