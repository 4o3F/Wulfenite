"""wulfenite.inference — whole-utterance and streaming inference scripts."""

from .streaming import run_streaming
from .whole import run_whole

__all__ = ["run_streaming", "run_whole"]
