"""Shared model configuration regression tests."""

import argparse

from snuai.model_config import (DEFAULT_MODEL_ID, MODEL_ID_ENV,
                                add_model_id_argument, default_model_id)


def test_default_is_qwen3_vl_thinking():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-VL-8B-Thinking"
    assert default_model_id({}) == DEFAULT_MODEL_ID


def test_environment_can_point_to_local_mirror():
    assert default_model_id({MODEL_ID_ENV: " /models/qwen3-vl-thinking "}) == \
        "/models/qwen3-vl-thinking"


def test_empty_environment_value_falls_back():
    assert default_model_id({MODEL_ID_ENV: "  "}) == DEFAULT_MODEL_ID


def test_argparse_helper_uses_shared_default():
    parser = argparse.ArgumentParser()
    add_model_id_argument(parser, default=default_model_id({}))
    assert parser.parse_args([]).model_id == DEFAULT_MODEL_ID
    assert parser.parse_args(["--model-id", "local/checkpoint"]).model_id == \
        "local/checkpoint"
