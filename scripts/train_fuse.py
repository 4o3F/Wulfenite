"""Phase 0a training: fuse-layer-only fine-tune of BSRNN+CAM++ on AISHELL-1.

This script:
  1. Builds the BSRNNCampPlus wrapper (English BSRNN + Chinese CAM++).
  2. Freezes everything except the single SpeakerFuseLayer ``Linear(192,128)``.
  3. Trains on-the-fly AISHELL-1 2-speaker mixtures with SI-SDR loss.
  4. Periodically writes checkpoints to ``assets/campplus/train_phase0a/``.

Run:
    uv run python train_fuse.py --aishell-root /path/to/data_aishell

Required files (the script will fail early with a clear error otherwise):
    ../assets/avg_model.pt                    (the released WeSep BSRNN weights)
    ../assets/campplus/campplus_cn_common.bin (from `uv run python download_campplus.py`)
    --aishell-root path containing data_aishell/wav/{train,dev,test}/S0xxx/*.wav
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aishell_mixer import AishellMixDataset, collate_mix_batch
from bsrnn_campplus import build_bsrnn_campplus


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Negative scale-invariant SDR, averaged over the batch.

    estimate, target: [B, T] float tensors in time domain.
    """
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Optimal scale
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    scale = dot / target_energy
    projection = scale * target

    noise = estimate - projection
    ratio = (projection * projection).sum(dim=-1) / (
        (noise * noise).sum(dim=-1) + eps
    )
    si_sdr = 10.0 * torch.log10(ratio + eps)  # [B]
    return -si_sdr.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"[init] device={device}")

    repo_root = Path(__file__).resolve().parent.parent
    bsrnn_ckpt = args.bsrnn_ckpt or (repo_root / "assets" / "avg_model.pt")
    campplus_ckpt = args.campplus_ckpt or (
        repo_root / "assets" / "campplus" / "campplus_cn_common.bin"
    )

    if not bsrnn_ckpt.exists():
        raise FileNotFoundError(
            f"WeSep BSRNN checkpoint not found at {bsrnn_ckpt}. "
            "Expected the released avg_model.pt under assets/."
        )
    if not campplus_ckpt.exists():
        raise FileNotFoundError(
            f"CAM++ checkpoint not found at {campplus_ckpt}. "
            "Run `uv run python download_campplus.py` first."
        )

    model = build_bsrnn_campplus(bsrnn_ckpt, campplus_ckpt, device=device)
    model.freeze_all_except_fuse_layer()
    print(f"[init] trainable parameters: {count_trainable_params(model)}")

    def pin_bn_eval(m: torch.nn.Module) -> None:
        """Keep BatchNorm running stats frozen. Must be called after .train()."""
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()

    # --- Data ---
    train_ds = AishellMixDataset(
        aishell_root=args.aishell_root,
        split="train",
        samples_per_epoch=args.samples_per_epoch,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        seed=None,  # fresh mixtures every __getitem__ call
    )
    val_ds = AishellMixDataset(
        aishell_root=args.aishell_root,
        split="dev",
        samples_per_epoch=args.val_samples,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        seed=args.seed,  # reproducible validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # virtual epoch with seeded __getitem__
        num_workers=args.num_workers,
        collate_fn=collate_mix_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate_mix_batch,
        drop_last=False,
    )

    # --- Optimizer ---
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"output dir: {out_dir}")
    log(f"trainable params: {count_trainable_params(model)}")
    log(f"train steps/epoch: {len(train_loader)}  val steps: {len(val_loader)}")

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pin_bn_eval(model)

        epoch_loss = 0.0
        epoch_count = 0
        epoch_t0 = time.time()
        for step, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            enrollment = batch["enrollment"].to(device, non_blocking=True)

            estimate = model(mixture, enrollment)
            loss = si_sdr_loss(estimate, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            opt.step()
            sched.step()

            epoch_loss += loss.item()
            epoch_count += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                log(
                    f"epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}"
                )

        train_avg = epoch_loss / max(1, epoch_count)
        dur = time.time() - epoch_t0
        log(f"epoch {epoch} train si-sdr-loss={train_avg:.4f} time={dur:.1f}s")

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                mixture = batch["mixture"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                enrollment = batch["enrollment"].to(device, non_blocking=True)
                estimate = model(mixture, enrollment)
                loss = si_sdr_loss(estimate, target)
                val_loss += loss.item()
                val_count += 1
        val_avg = val_loss / max(1, val_count)
        log(f"epoch {epoch} val   si-sdr-loss={val_avg:.4f}")

        # Save checkpoints — we only need the trainable fuse-layer weights,
        # but we also dump the full state_dict for easy swapping at infer time.
        # Convert any Path values in args to strings before saving so the
        # checkpoint can be loaded on a different OS (Linux-saved PosixPath
        # cannot be unpickled on Windows).
        args_for_save = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        }
        ckpt = {
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_avg,
            "val_loss": val_avg,
            "state_dict": model.state_dict(),
            "fuse_state_dict": {
                k: v.detach().cpu()
                for k, v in model.separator.separation[0].fc.linear.state_dict().items()
            },
            "args": args_for_save,
        }
        torch.save(ckpt, out_dir / f"epoch{epoch:03d}.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ckpt, out_dir / "best.pt")
            log(f"[best] val improved to {best_val:.4f} — saved best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", type=Path, required=True,
                        help="Path to AISHELL-1 root (contains data_aishell/wav/)")
    parser.add_argument("--bsrnn-ckpt", type=Path, default=None,
                        help="WeSep BSRNN checkpoint (default: ../assets/avg_model.pt)")
    parser.add_argument("--campplus-ckpt", type=Path, default=None,
                        help="CAM++ zh-cn checkpoint (default: ../assets/campplus/campplus_cn_common.bin)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "campplus" / "train_phase0a",
                        help="Where to save checkpoints + logs")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="8 is safe for 24 GB VRAM with 4 s chunks; raise for more")
    parser.add_argument("--samples-per-epoch", type=int, default=10000,
                        help="Virtual epoch size for the on-the-fly mixer")
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=[-5.0, 5.0])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
