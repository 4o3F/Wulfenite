"""wulfenite.inference — whole-utterance and streaming inference scripts.

Intentionally empty of re-exports. Both ``whole.py`` and
``streaming.py`` are entry-point modules meant to be invoked as
``python -m wulfenite.inference.whole`` or ``python -m
wulfenite.inference.streaming``. Re-exporting ``run_whole`` /
``run_streaming`` here would preload those modules into
``sys.modules`` and produce a ``runpy`` warning at launch time.

Code that wants to call the inference functions from Python should
import them directly::

    from wulfenite.inference.whole import run_whole
    from wulfenite.inference.streaming import run_streaming
"""
