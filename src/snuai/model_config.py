"""Shared model selection for inference and legacy training entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-Reranker-8B"
"""Official multimodal reranker used by the public inference pipeline."""

MODEL_ID_ENV = "SNUAI_MODEL_ID"
"""Environment override for offline reranker mirrors and local checkpoints."""

LEGACY_TRAINING_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
"""Generative checkpoint retained for historical score24 SFT/DPO experiments."""

LEGACY_MODEL_ID_ENV = "SNUAI_LEGACY_MODEL_ID"


def _model_id(default: str, env_name: str,
              environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get(env_name, "").strip() or default


def default_model_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured Qwen3-VL-Reranker checkpoint."""
    return _model_id(DEFAULT_MODEL_ID, MODEL_ID_ENV, environ)


def legacy_training_model_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the generative checkpoint for legacy training-only commands."""
    return _model_id(LEGACY_TRAINING_MODEL_ID, LEGACY_MODEL_ID_ENV, environ)


def add_model_id_argument(parser, **kwargs):
    """Add the shared reranker ``--model-id`` option to an argument parser."""
    kwargs.setdefault("default", default_model_id())
    kwargs.setdefault(
        "help",
        f"reranker checkpoint (default: {DEFAULT_MODEL_ID}; override with {MODEL_ID_ENV})",
    )
    return parser.add_argument("--model-id", **kwargs)


def add_legacy_model_id_argument(parser, **kwargs):
    """Add a clearly separated model option for score24 SFT/DPO utilities."""
    kwargs.setdefault("default", legacy_training_model_id())
    kwargs.setdefault(
        "help",
        "legacy generative checkpoint for score24 experiments "
        f"(default: {LEGACY_TRAINING_MODEL_ID}; override with {LEGACY_MODEL_ID_ENV})",
    )
    return parser.add_argument("--model-id", **kwargs)
