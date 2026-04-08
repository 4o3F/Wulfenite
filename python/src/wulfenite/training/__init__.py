"""wulfenite.training — training loop, config, and checkpoint utilities.

Library-level re-exports. ``train.py`` is NOT imported here because
it is an entry-point module meant to be run as ``python -m
wulfenite.training.train``; importing it at package-init time would
preload it into ``sys.modules`` and trigger a ``runpy`` warning
("found in sys.modules after import of package ..."). Code that
wants the training loop from Python should do::

    from wulfenite.training.train import run_training

instead of ``from wulfenite.training import run_training``.
"""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TrainingConfig

__all__ = [
    "TrainingConfig",
    "load_checkpoint",
    "save_checkpoint",
]
