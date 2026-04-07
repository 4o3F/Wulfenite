"""Phase 0b training: unfreeze the BSRNN separator + train with augmentation.

Motivation
----------
Phase 0a (fuse-layer-only) converged cleanly on AISHELL-1 mixtures
(val SI-SDR-loss ~-9.4 dB, diagnose script confirmed +11.5 dB average
SI-SDR improvement on dev), but the frozen English BSRNN separator
does not generalize to real small-room Chinese streaming audio — the
mask magnitudes collapse and the output is audibly unseparated.

Phase 0b fixes both the ceiling and the training objective:

1. **Unfreeze the separator** (keep CAM++ frozen). Now ~6M params move,
   starting from the Phase 0a best checkpoint so we keep the fuse-layer
   learning.
2. **Acoustic augmentation** — synthetic room impulse responses applied
   per-speaker plus mild broadband noise, bridging AISHELL studio
   recordings to small-room streaming conditions.
3. **Loss = SI-SDR (for shape) + log-magnitude penalty (for level)**,
   eliminating the scale-invariance artifact that made Phase 0a produce
   chronically low-level outputs.
4. **Lower learning rate** (1e-4 vs 1e-3) since we are now moving the
   full separator stack instead of a single Linear layer.

Usage
-----
    uv run python train_phase0b.py \
        --aishell-root /path/to/aishell \
        --init-from ../assets/campplus/train_phase0a/best.pt \
        --batch-size 12 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aishell_mixer import AishellMixDataset, collate_mix_batch
from augmentation import AugmentationConfig, ReverbConfig
from bsrnn_campplus import build_bsrnn_campplus


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor,
                eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR (shape only)."""
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target * target).sum(dim=-1, keepdim=True) + eps
    scale = dot / target_energy
    projection = scale * target
    noise = estimate - projection
    ratio = (projection * projection).sum(dim=-1) / (
        (noise * noise).sum(dim=-1) + eps
    )
    return -10.0 * torch.log10(ratio + eps).mean()


def log_mag_penalty(estimate: torch.Tensor, target: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Absolute difference of log-RMS between estimate and target.

    Forces the estimate to have the same overall level as the target,
    counteracting SI-SDR's scale invariance. ``| log(rms_est / rms_tgt) |``
    averaged over the batch.
    """
    est_rms = torch.sqrt((estimate * estimate).mean(dim=-1) + eps)
    tgt_rms = torch.sqrt((target * target).mean(dim=-1) + eps)
    return torch.abs(torch.log(est_rms + eps) - torch.log(tgt_rms + eps)).mean()


def phase0b_loss(estimate: torch.Tensor, target: torch.Tensor,
                 si_weight: float = 1.0, mag_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    si = si_sdr_loss(estimate, target)
    mag = log_mag_penalty(estimate, target)
    total = si_weight * si + mag_weight * mag
    return total, {"si_sdr": float(si.detach()), "log_mag": float(mag.detach())}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pin_bn_eval(model: torch.nn.Module) -> None:
    """Keep BatchNorm running stats frozen even though the affine params
    are in the trainable set. Running stats from the English pretraining
    are closer to "right" than whatever small batches would produce."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"[init] device={device}")

    repo_root = Path(__file__).resolve().parent.parent
    bsrnn_ckpt = args.bsrnn_ckpt or (repo_root / "assets" / "avg_model.pt")
    campplus_ckpt = args.campplus_ckpt or (
        repo_root / "assets" / "campplus" / "campplus_cn_common.bin"
    )
    init_from = args.init_from or (
        repo_root / "assets" / "campplus" / "train_phase0a" / "best.pt"
    )

    for p in (bsrnn_ckpt, campplus_ckpt):
        if not p.exists():
            raise FileNotFoundError(p)

    model = build_bsrnn_campplus(bsrnn_ckpt, campplus_ckpt, device=device)

    # Init from Phase 0a best checkpoint so we keep the learned fuse layer
    # and start Phase 0b from a known-good CAM++->separator mapping.
    if init_from and Path(init_from).exists():
        ckpt = torch.load(init_from, map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(
                ckpt["state_dict"], strict=False
            )
            print(
                f"[init] loaded Phase 0a state_dict from {init_from} "
                f"(epoch={ckpt.get('epoch')}); missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        else:
            print(f"[init] WARN: {init_from} has no 'state_dict' key, skipping")
    else:
        print(f"[init] no Phase 0a checkpoint (init_from={init_from}), "
              "starting from raw English BSRNN + fresh fuse layer")

    # Phase 0b freezing: only CAM++ is frozen
    model.freeze_only_campplus()
    n_train = count_trainable_params(model)
    print(f"[init] Phase 0b trainable parameters: {n_train}")

    # --- Data (with augmentation) ---
    aug_cfg = AugmentationConfig(
        reverb=ReverbConfig(rt60_range=tuple(args.rt60_range)),
        noise_snr_range_db=tuple(args.noise_snr_range_db),
        p_reverb=args.p_reverb,
        p_noise=args.p_noise,
        reverb_enrollment=args.reverb_enrollment,
    )
    train_ds = AishellMixDataset(
        aishell_root=args.aishell_root,
        split="train",
        samples_per_epoch=args.samples_per_epoch,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        seed=None,
        augment=True,
        aug_config=aug_cfg,
    )
    val_ds = AishellMixDataset(
        aishell_root=args.aishell_root,
        split="dev",
        samples_per_epoch=args.val_samples,
        segment_seconds=args.segment_seconds,
        enrollment_seconds=args.enrollment_seconds,
        snr_range_db=tuple(args.snr_range_db),
        seed=args.seed,
        augment=True,
        aug_config=aug_cfg,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_mix_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate_mix_batch, drop_last=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=True,
    )

    # --- Optimizer ---
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(0.03 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

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
    log(f"trainable params: {n_train}")
    log(f"train steps/epoch: {len(train_loader)}  val steps: {len(val_loader)}")
    log(f"augmentation: rt60={args.rt60_range} noise_snr={args.noise_snr_range_db} "
        f"p_reverb={args.p_reverb} p_noise={args.p_noise}")
    log(f"loss weights: si_sdr={args.si_weight} log_mag={args.mag_weight}")

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pin_bn_eval(model)

        epoch_total = 0.0
        epoch_si = 0.0
        epoch_mag = 0.0
        epoch_count = 0
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            enrollment = batch["enrollment"].to(device, non_blocking=True)

            estimate = model(mixture, enrollment)
            loss, parts = phase0b_loss(
                estimate, target,
                si_weight=args.si_weight, mag_weight=args.mag_weight,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            opt.step()
            sched.step()

            epoch_total += float(loss.detach())
            epoch_si += parts["si_sdr"]
            epoch_mag += parts["log_mag"]
            epoch_count += 1
            global_step += 1

            if global_step % args.log_interval == 0:
                log(
                    f"epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"loss={float(loss.detach()):.4f} "
                    f"si={parts['si_sdr']:+.3f} mag={parts['log_mag']:.4f} "
                    f"lr={opt.param_groups[0]['lr']:.2e}"
                )

        dur = time.time() - t0
        log(
            f"epoch {epoch} train total={epoch_total / epoch_count:.4f} "
            f"si={epoch_si / epoch_count:+.3f} "
            f"mag={epoch_mag / epoch_count:.4f}  time={dur:.1f}s"
        )

        # Validation
        model.eval()
        val_total = 0.0
        val_si = 0.0
        val_mag = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                mixture = batch["mixture"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                enrollment = batch["enrollment"].to(device, non_blocking=True)
                estimate = model(mixture, enrollment)
                loss, parts = phase0b_loss(
                    estimate, target,
                    si_weight=args.si_weight, mag_weight=args.mag_weight,
                )
                val_total += float(loss)
                val_si += parts["si_sdr"]
                val_mag += parts["log_mag"]
                val_count += 1

        val_avg_total = val_total / max(1, val_count)
        val_avg_si = val_si / max(1, val_count)
        val_avg_mag = val_mag / max(1, val_count)
        log(
            f"epoch {epoch} val   total={val_avg_total:.4f} "
            f"si={val_avg_si:+.3f} mag={val_avg_mag:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "step": global_step,
            "train_total": epoch_total / epoch_count,
            "val_total": val_avg_total,
            "val_si_sdr": val_avg_si,
            "val_log_mag": val_avg_mag,
            "state_dict": model.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / f"epoch{epoch:03d}.pt")
        if val_avg_total < best_val:
            best_val = val_avg_total
            torch.save(ckpt, out_dir / "best.pt")
            log(f"[best] val improved to {best_val:.4f} — saved best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", type=Path, required=True)
    parser.add_argument("--bsrnn-ckpt", type=Path, default=None)
    parser.add_argument("--campplus-ckpt", type=Path, default=None)
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Phase 0a checkpoint to start from "
                             "(default: ../assets/campplus/train_phase0a/best.pt)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "assets" / "campplus" / "train_phase0b",
                        help="Where to save checkpoints + logs")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="12 is safe for 24 GB VRAM with 4 s chunks and full separator grads")
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--enrollment-seconds", type=float, default=4.0)
    parser.add_argument("--snr-range-db", type=float, nargs=2, default=[-5.0, 5.0])

    # Augmentation knobs
    parser.add_argument("--rt60-range", type=float, nargs=2, default=[0.08, 0.35],
                        help="Synthetic RIR RT60 range in seconds (small room)")
    parser.add_argument("--noise-snr-range-db", type=float, nargs=2,
                        default=[15.0, 30.0],
                        help="Additive broadband noise SNR range")
    parser.add_argument("--p-reverb", type=float, default=0.85)
    parser.add_argument("--p-noise", type=float, default=0.80)
    parser.add_argument("--reverb-enrollment", action="store_true", default=True,
                        help="Also reverberate the enrollment signal")

    # Loss knobs
    parser.add_argument("--si-weight", type=float, default=1.0)
    parser.add_argument("--mag-weight", type=float, default=0.5,
                        help="Weight for the log-magnitude penalty that "
                             "kills SI-SDR's scale invariance")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader worker processes. Bump higher if "
                             "the server has many cores and GPU is still idle.")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Per-worker batches to prefetch. 4 keeps GPU "
                             "fed even when one worker is briefly slow.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
