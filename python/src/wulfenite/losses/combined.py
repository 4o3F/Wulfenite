"""Combined separation loss with routing-aware regularizers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .inactive import target_inactive_loss
from .mr_stft import MultiResolutionSTFTLoss
from .presence import presence_loss
from .recall import target_recall_loss
from .sdr import sdr_loss
from .silence import target_absent_loss


VIEW_ROLE_A = 0
VIEW_ROLE_B = 1


@dataclass
class LossWeights:
    """Per-component weights for :class:`WulfeniteLoss`."""

    sdr: float = 1.0
    mr_stft: float = 1.0
    absent: float = 0.5
    presence: float = 0.1
    recall: float = 0.0
    inactive: float = 0.25
    route: float = 0.5
    overlap_route: float = 0.25
    ae: float = 0.0


@dataclass
class LossParts:
    """Per-component scalar breakdown for logging."""

    total: float
    sdr: float
    mr_stft: float
    recall: float
    inactive: float
    absent: float
    presence: float
    route: float
    overlap_route: float
    ae: float
    n_present: int
    n_absent: int
    n_route_pairs: int


def _frame_energy(signal: torch.Tensor, frame_size: int) -> torch.Tensor:
    if signal.dim() != 2:
        raise ValueError(f"signal must be [B, T]; got {tuple(signal.shape)}")
    batch, total = signal.shape
    n_frames = total // frame_size
    usable = n_frames * frame_size
    return signal[:, :usable].reshape(batch, n_frames, frame_size).pow(2).mean(dim=-1)


def _align_frame_mask(mask: torch.Tensor, n_frames: int) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError(f"mask must be [B, F]; got {tuple(mask.shape)}")
    aligned = mask.bool()
    if aligned.shape[1] == n_frames:
        return aligned
    if aligned.shape[1] % n_frames != 0:
        raise ValueError(
            f"mask length {aligned.shape[1]} cannot align to {n_frames} frames"
        )
    factor = aligned.shape[1] // n_frames
    return aligned.reshape(aligned.shape[0], n_frames, factor).any(dim=-1)


def _mean_or_zero(
    values: list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not values:
        return torch.zeros((), device=device, dtype=dtype)
    return torch.stack(values, dim=0).mean()


def compute_scene_routing_stats(
    estimate: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    *,
    target_active_frames: torch.Tensor | None,
    nontarget_active_frames: torch.Tensor | None,
    overlap_frames: torch.Tensor | None,
    background_frames: torch.Tensor | None,
    scene_id: torch.Tensor | None,
    view_role_id: torch.Tensor | None,
    frame_size: int = 160,
    route_margin: float = 0.05,
    overlap_margin: float = 0.02,
    overlap_dominance_margin: float = 0.02,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute same-scene routing losses and validation metrics."""
    device = estimate.device
    dtype = estimate.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if (
        target_active_frames is None
        or nontarget_active_frames is None
        or overlap_frames is None
        or background_frames is None
        or scene_id is None
        or view_role_id is None
    ):
        return {
            "route_loss": zero,
            "overlap_route_loss": zero,
            "target_only_energy_true": zero,
            "target_only_energy_wrong": zero,
            "other_only_energy_true": zero,
            "overlap_energy_true": zero,
            "overlap_energy_wrong": zero,
            "route_margin_target_only": zero,
            "route_margin_overlap": zero,
            "wrong_enrollment_leakage": zero,
            "n_pairs": torch.zeros((), device=device, dtype=torch.long),
        }

    est_energy = _frame_energy(estimate, frame_size)
    tgt_energy = _frame_energy(target, frame_size)
    mix_energy = _frame_energy(mixture, frame_size)
    n_frames = est_energy.shape[1]
    target_active = _align_frame_mask(target_active_frames.to(device), n_frames)
    nontarget_active = _align_frame_mask(nontarget_active_frames.to(device), n_frames)
    overlap = _align_frame_mask(overlap_frames.to(device), n_frames)
    background = _align_frame_mask(background_frames.to(device), n_frames)
    scene_ids = scene_id.to(device).long()
    role_ids = view_role_id.to(device).long()

    route_terms: list[torch.Tensor] = []
    overlap_terms: list[torch.Tensor] = []
    target_true_vals: list[torch.Tensor] = []
    target_wrong_vals: list[torch.Tensor] = []
    overlap_true_vals: list[torch.Tensor] = []
    overlap_wrong_vals: list[torch.Tensor] = []
    pair_count = 0

    for sid in torch.unique(scene_ids).tolist():
        scene_mask = scene_ids == sid
        a_idx = torch.nonzero(scene_mask & (role_ids == VIEW_ROLE_A), as_tuple=False)
        b_idx = torch.nonzero(scene_mask & (role_ids == VIEW_ROLE_B), as_tuple=False)
        if a_idx.numel() == 0 or b_idx.numel() == 0:
            continue
        a = int(a_idx[0].item())
        b = int(b_idx[0].item())
        pair_count += 1

        mix_ref = mix_energy[a]
        a_ratio = est_energy[a] / (mix_ref + eps)
        b_ratio = est_energy[b] / (mix_ref + eps)

        a_only = target_active[a] & ~nontarget_active[a]
        b_only = target_active[b] & ~nontarget_active[b]
        bg = background[a] & background[b]
        ov = overlap[a] & overlap[b]

        if bool(a_only.any().item()):
            delta = a_ratio[a_only] - b_ratio[a_only]
            route_terms.append(torch.relu(route_margin - delta).square().mean())
            target_true_vals.append(a_ratio[a_only].mean())
            target_wrong_vals.append(b_ratio[a_only].mean())
        if bool(b_only.any().item()):
            delta = b_ratio[b_only] - a_ratio[b_only]
            route_terms.append(torch.relu(route_margin - delta).square().mean())
            target_true_vals.append(b_ratio[b_only].mean())
            target_wrong_vals.append(a_ratio[b_only].mean())
        if bool(bg.any().item()):
            route_terms.append((a_ratio[bg] + b_ratio[bg]).mean())

        if bool(ov.any().item()):
            a_overlap = a_ratio[ov]
            b_overlap = b_ratio[ov]
            tgt_gap = (tgt_energy[a][ov] - tgt_energy[b][ov]) / (mix_ref[ov] + eps)
            dom_a = tgt_gap > overlap_dominance_margin
            dom_b = tgt_gap < -overlap_dominance_margin
            if bool(dom_a.any().item()):
                overlap_terms.append(
                    torch.relu(
                        overlap_margin - (a_overlap[dom_a] - b_overlap[dom_a])
                    ).square().mean()
                )
                overlap_true_vals.append(a_overlap[dom_a].mean())
                overlap_wrong_vals.append(b_overlap[dom_a].mean())
            if bool(dom_b.any().item()):
                overlap_terms.append(
                    torch.relu(
                        overlap_margin - (b_overlap[dom_b] - a_overlap[dom_b])
                    ).square().mean()
                )
                overlap_true_vals.append(b_overlap[dom_b].mean())
                overlap_wrong_vals.append(a_overlap[dom_b].mean())

    target_true = _mean_or_zero(target_true_vals, device=device, dtype=dtype)
    target_wrong = _mean_or_zero(target_wrong_vals, device=device, dtype=dtype)
    overlap_true = _mean_or_zero(overlap_true_vals, device=device, dtype=dtype)
    overlap_wrong = _mean_or_zero(overlap_wrong_vals, device=device, dtype=dtype)
    return {
        "route_loss": _mean_or_zero(route_terms, device=device, dtype=dtype),
        "overlap_route_loss": _mean_or_zero(
            overlap_terms, device=device, dtype=dtype,
        ),
        "target_only_energy_true": target_true,
        "target_only_energy_wrong": target_wrong,
        "other_only_energy_true": target_wrong,
        "overlap_energy_true": overlap_true,
        "overlap_energy_wrong": overlap_wrong,
        "route_margin_target_only": target_true - target_wrong,
        "route_margin_overlap": overlap_true - overlap_wrong,
        "wrong_enrollment_leakage": target_wrong,
        "n_pairs": torch.tensor(pair_count, device=device, dtype=torch.long),
    }


class WulfeniteLoss(nn.Module):
    """End-to-end loss module used during training."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        mr_stft_loss: MultiResolutionSTFTLoss | None = None,
        recall_frame_size: int = 320,
        recall_floor: float = 0.3,
        inactive_threshold: float = 0.05,
        inactive_topk_fraction: float = 0.25,
        route_frame_size: int = 160,
        route_margin: float = 0.05,
        overlap_margin: float = 0.02,
        overlap_dominance_margin: float = 0.02,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.mr_stft = mr_stft_loss or MultiResolutionSTFTLoss()
        self.recall_frame_size = recall_frame_size
        self.recall_floor = recall_floor
        self.inactive_threshold = inactive_threshold
        self.inactive_topk_fraction = inactive_topk_fraction
        self.route_frame_size = route_frame_size
        self.route_margin = route_margin
        self.overlap_margin = overlap_margin
        self.overlap_dominance_margin = overlap_dominance_margin

    def forward(
        self,
        clean: torch.Tensor,
        target: torch.Tensor,
        mixture: torch.Tensor,
        target_present: torch.Tensor,
        presence_logit: torch.Tensor | None = None,
        target_active_frames: torch.Tensor | None = None,
        nontarget_active_frames: torch.Tensor | None = None,
        overlap_frames: torch.Tensor | None = None,
        background_frames: torch.Tensor | None = None,
        scene_id: torch.Tensor | None = None,
        view_role_id: torch.Tensor | None = None,
        ae_reconstruction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LossParts]:
        """Compute the full training loss."""
        if clean.shape != target.shape or clean.shape != mixture.shape:
            raise ValueError(
                "clean / target / mixture must all share shape "
                f"[B, T]; got {tuple(clean.shape)}, {tuple(target.shape)}, "
                f"{tuple(mixture.shape)}"
            )
        if target_present.shape != clean.shape[:1]:
            raise ValueError(
                f"target_present must be shape [B]; got "
                f"{tuple(target_present.shape)}"
            )

        device = clean.device
        dtype = clean.dtype
        zero = torch.zeros((), device=device, dtype=dtype)

        present_mask = target_present.to(device).bool()
        absent_mask = ~present_mask
        n_present = int(present_mask.sum().item())
        n_absent = int(absent_mask.sum().item())

        l_sdr = zero
        l_stft = zero
        l_recall = zero
        l_inactive = zero
        if n_present > 0:
            l_sdr = sdr_loss(clean[present_mask], target[present_mask])
            l_stft = self.mr_stft(clean[present_mask], target[present_mask])
            if self.weights.recall > 0.0:
                active_frames = None
                if target_active_frames is not None:
                    active_frames = target_active_frames[present_mask]
                    if overlap_frames is not None:
                        active_frames = (
                            active_frames.bool()
                            & ~overlap_frames[present_mask].bool()
                        )
                l_recall = target_recall_loss(
                    clean[present_mask],
                    target[present_mask],
                    frame_size=self.recall_frame_size,
                    active_frames=active_frames,
                    floor=self.recall_floor,
                )
            if (
                self.weights.inactive > 0.0
                and target_active_frames is not None
                and nontarget_active_frames is not None
            ):
                inactive_frames = (
                    nontarget_active_frames[present_mask].bool()
                    & ~target_active_frames[present_mask].bool()
                )
                if bool(inactive_frames.any().item()):
                    l_inactive = target_inactive_loss(
                        clean[present_mask],
                        mixture[present_mask],
                        inactive_frames=inactive_frames,
                        threshold=self.inactive_threshold,
                        topk_fraction=self.inactive_topk_fraction,
                    )

        l_absent = zero
        if n_absent > 0:
            l_absent = target_absent_loss(clean[absent_mask], mixture[absent_mask])

        l_presence = zero
        if presence_logit is not None:
            l_presence = presence_loss(presence_logit, target_present.to(device))

        routing_stats = compute_scene_routing_stats(
            clean,
            target,
            mixture,
            target_active_frames=target_active_frames,
            nontarget_active_frames=nontarget_active_frames,
            overlap_frames=overlap_frames,
            background_frames=background_frames,
            scene_id=scene_id,
            view_role_id=view_role_id,
            frame_size=self.route_frame_size,
            route_margin=self.route_margin,
            overlap_margin=self.overlap_margin,
            overlap_dominance_margin=self.overlap_dominance_margin,
        )
        l_route = routing_stats["route_loss"]
        l_overlap_route = routing_stats["overlap_route_loss"]

        l_ae = zero
        if self.weights.ae > 0.0:
            if ae_reconstruction is None:
                raise ValueError(
                    "ae_reconstruction is required when loss weight ae > 0"
                )
            l_ae = F.l1_loss(ae_reconstruction, mixture)

        w = self.weights
        total = (
            w.sdr * l_sdr
            + w.mr_stft * l_stft
            + w.recall * l_recall
            + w.inactive * l_inactive
            + w.absent * l_absent
            + w.presence * l_presence
            + w.route * l_route
            + w.overlap_route * l_overlap_route
            + w.ae * l_ae
        )

        parts = LossParts(
            total=float(total.detach()),
            sdr=float(l_sdr.detach()),
            mr_stft=float(l_stft.detach()),
            recall=float(l_recall.detach()),
            inactive=float(l_inactive.detach()),
            absent=float(l_absent.detach()),
            presence=float(l_presence.detach()),
            route=float(l_route.detach()),
            overlap_route=float(l_overlap_route.detach()),
            ae=float(l_ae.detach()),
            n_present=n_present,
            n_absent=n_absent,
            n_route_pairs=int(routing_stats["n_pairs"].item()),
        )
        return total, parts
