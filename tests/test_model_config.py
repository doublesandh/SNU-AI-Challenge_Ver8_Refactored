"""Shared model configuration regression tests."""

import argparse

from snuai.model_config import (DEFAULT_MODEL_ID, LEGACY_MODEL_ID_ENV,
                                LEGACY_TRAINING_MODEL_ID, MODEL_ID_ENV,
                                add_legacy_model_id_argument,
                                add_model_id_argument, default_model_id,
                                legacy_training_model_id)


def test_default_is_qwen3_vl_reranker():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen3-VL-Reranker-8B"
    assert default_model_id({}) == DEFAULT_MODEL_ID


def test_environment_can_point_to_local_reranker_mirror():
    assert default_model_id({MODEL_ID_ENV: " /models/qwen3-vl-reranker "}) == \
        "/models/qwen3-vl-reranker"


def test_empty_environment_value_falls_back():
    assert default_model_id({MODEL_ID_ENV: "  "}) == DEFAULT_MODEL_ID


def test_argparse_helper_uses_reranker_default():
    parser = argparse.ArgumentParser()
    add_model_id_argument(parser, default=default_model_id({}))
    assert parser.parse_args([]).model_id == DEFAULT_MODEL_ID
    assert parser.parse_args(["--model-id", "local/checkpoint"]).model_id == \
        "local/checkpoint"


def test_legacy_training_model_is_explicitly_separate():
    assert legacy_training_model_id({}) == LEGACY_TRAINING_MODEL_ID
    assert legacy_training_model_id({LEGACY_MODEL_ID_ENV: " /legacy/model "}) == \
        "/legacy/model"
    parser = argparse.ArgumentParser()
    add_legacy_model_id_argument(parser, default=legacy_training_model_id({}))
    assert parser.parse_args([]).model_id == LEGACY_TRAINING_MODEL_ID
