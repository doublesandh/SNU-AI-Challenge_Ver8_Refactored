"""Training and inference must enforce identical model invariants."""

from types import SimpleNamespace

from snuai.model_checks import (verify_lora_only_on_language,
                                verify_vision_not_quantized)
from snuai.train.qlora import (
    verify_lora_only_on_language as qlora_verify_lora,
    verify_vision_not_quantized as qlora_verify_quantization,
)


Linear4bit = type("Linear4bit", (), {})


class _Model:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return iter(self._modules)


def test_qlora_reexports_shared_checks():
    assert qlora_verify_lora is verify_lora_only_on_language
    assert qlora_verify_quantization is verify_vision_not_quantized


def test_quantization_counts_language_modules():
    model = _Model([
        ("model.language_model.layers.0.q_proj", Linear4bit()),
        ("model.visual.blocks.0", SimpleNamespace()),
    ])
    assert verify_vision_not_quantized(model) == {
        "n_quant_lang": 1,
        "n_quant_vision": 0,
    }


def test_lora_counts_language_modules():
    model = _Model([
        ("model.language_model.layers.0.q_proj.lora_A", SimpleNamespace()),
        ("model.visual.blocks.0", SimpleNamespace()),
    ])
    assert verify_lora_only_on_language(model) == {
        "n_lora_lang": 1,
        "n_lora_vision": 0,
    }
