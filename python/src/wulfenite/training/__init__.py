"""wulfenite.training — PSE training entry points."""

from .config import TrainConfig
from .kd_dataset import TinyECAPAKDBatch, TinyECAPAKDDataset, split_speakers_for_kd
from .train_pdfnet2 import run_pdfnet2_epoch, scheduled_batch_size, train_pdfnet2
from .train_tiny_ecapa import (
    ContrastiveKDLoss,
    augment_speaker_batch,
    load_tiny_ecapa_checkpoint,
    run_tiny_ecapa_epoch,
    train_tiny_ecapa,
)

__all__ = [
    "TrainConfig",
    "TinyECAPAKDBatch",
    "TinyECAPAKDDataset",
    "split_speakers_for_kd",
    "scheduled_batch_size",
    "run_pdfnet2_epoch",
    "train_pdfnet2",
    "ContrastiveKDLoss",
    "augment_speaker_batch",
    "load_tiny_ecapa_checkpoint",
    "run_tiny_ecapa_epoch",
    "train_tiny_ecapa",
]
