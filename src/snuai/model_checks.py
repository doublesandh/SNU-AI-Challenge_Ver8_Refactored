"""Model invariant checks shared by training and inference."""

from __future__ import annotations


VISION_PATH_MARKERS = ("visual", "vision")
QUANTIZED_MODULE_TYPES = ("Linear4bit", "Params4bit")


def _is_vision_path(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in VISION_PATH_MARKERS)


def verify_vision_not_quantized(model) -> dict[str, int]:
    """Ensure 4-bit modules are confined to the language model."""

    n_lang = n_vis = 0
    for name, module in model.named_modules():
        if type(module).__name__ not in QUANTIZED_MODULE_TYPES:
            continue
        if _is_vision_path(name):
            n_vis += 1
        else:
            n_lang += 1
    if n_vis:
        raise RuntimeError(
            f"vision tower에 4bit 모듈 {n_vis}개 — 규약 위반(vision 비양자화)"
        )
    return {"n_quant_lang": n_lang, "n_quant_vision": n_vis}


def verify_lora_only_on_language(model) -> dict[str, int]:
    """Ensure LoRA modules exist and are confined to the language model."""

    n_lang = n_vis = 0
    for name, _ in model.named_modules():
        if "lora_" not in name:
            continue
        if _is_vision_path(name):
            n_vis += 1
        else:
            n_lang += 1
    if n_vis:
        raise RuntimeError(
            f"vision tower에 LoRA {n_vis}개 — select_lora_targets를 우회했는지 확인"
        )
    if not n_lang:
        raise RuntimeError("LoRA가 하나도 안 붙음")
    return {"n_lora_lang": n_lang, "n_lora_vision": n_vis}
