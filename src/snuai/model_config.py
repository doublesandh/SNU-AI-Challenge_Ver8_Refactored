"""Shared model selection for training, inference, and maintenance scripts.

Keep the default in one place: model-specific CLI defaults used to be duplicated
across seven entry points and had already drifted between local 8B and VESSL 32B
checkpoints.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
"""Official reasoning-enhanced Qwen3-VL checkpoint used by default."""

MODEL_ID_ENV = "SNUAI_MODEL_ID"
"""Environment override for offline mirrors and local checkpoint paths."""


def default_model_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured default model ID.

    ``SNUAI_MODEL_ID`` lets GPU machines use a local Hugging Face mirror without
    changing source code. Empty or whitespace-only values fall back to the
    canonical public checkpoint.
    """

    env = os.environ if environ is None else environ
    return env.get(MODEL_ID_ENV, "").strip() or DEFAULT_MODEL_ID


def add_model_id_argument(parser, **kwargs):
    """Add the shared ``--model-id`` option to an argparse parser."""

    kwargs.setdefault("default", default_model_id())
    kwargs.setdefault(
        "help",
        f"base model/checkpoint (default: {DEFAULT_MODEL_ID}; "
        f"override with {MODEL_ID_ENV})",
    )
    return parser.add_argument("--model-id", **kwargs)
