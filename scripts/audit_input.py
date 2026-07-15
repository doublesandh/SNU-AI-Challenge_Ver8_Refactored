#!/usr/bin/env python
"""§17 입력 감사 — 모델에 실제로 들어갈 input_ids/픽셀/타임스탬프를 디코드해 덤프하고,
이전 덤프와 구조 필드를 diff한다 (원 구현계획 §17, Ver2 사고 재발 방지용 자동화).

Ver2 사고(비디오 인코딩이 셔플된 무관 프레임을 병합 → LB -16.9pp)의 진단에 실제로
쓴 것과 같은 종류의 확인(프레임당 토큰/픽셀, "Image k:" 라벨 개수, 타임스탬프 개수)을
매번 손으로 하는 대신 게이트로 제도화한다. **모델 가중치 없이(프로세서만) 실행
가능** — GPU 불필요.

사용:
  python scripts/audit_input.py --csv data/train.csv --image-dir data/train \
      --model-id Qwen/Qwen3-VL-Reranker-8B \
      --video-mode --video-dup-factor 2 \
      --out runs/audit/reranker_videodup.json

  # 이전 덤프와 구조 diff (허용 목록 밖 변화가 있으면 exit 1)
  python scripts/audit_input.py ... --out runs/audit/ver7_videodup.json \
      --diff runs/audit/ver3_image_baseline.json \
      --allow-diff frame_count,timestamp_count,vision_tokens_total,text_tokens_total,total_len

운영 규칙: 인코딩/프롬프트를 바꿀 때마다 이전 Ver 덤프 대비 --diff로 돌려서,
config에 명시한 --allow-diff 밖의 구조 변화가 있으면 학습을 시작하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TIMESTAMP_RE = re.compile(r"<\d+\.?\d*\s*seconds?>")
_IMAGE_LABEL_RE = re.compile(r"Image \d+:")
_CANDIDATE_LABEL_RE = re.compile(
    r"Candidate (?:earliest|second|third|latest|position \d+):")


# ---------------------------------------------------------------------------
# 순수 함수 (pytest 가능 — 실제 프로세서/모델 불필요)
# ---------------------------------------------------------------------------

def count_timestamps(decoded_text: str) -> int:
    return len(_TIMESTAMP_RE.findall(decoded_text))


def count_image_labels(decoded_text: str) -> int:
    return len(_IMAGE_LABEL_RE.findall(decoded_text))


def collapse_vision_runs(ids: list[int], vision_ids: set[int]) -> str:
    """비전 자리표시 토큰의 연속 런을 [VISION×N]으로 접어 사람이 읽기 좋게."""
    out: list[str] = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] in vision_ids:
            j = i
            while j < n and ids[j] in vision_ids:
                j += 1
            out.append(f"[VISION×{j - i}]")
            i = j
        else:
            out.append(str(ids[i]))
            i += 1
    return " ".join(out)


def diff_facts(prev: dict, curr: dict, allow: set[str]) -> list[str]:
    """recipe(설명용 메타)를 제외한 구조 필드 중, allow에 없는데 달라진 것 나열."""
    diffs = []
    keys = (set(prev.keys()) | set(curr.keys())) - {"recipe", "label_check"}
    for k in sorted(keys):
        if k in allow:
            continue
        if prev.get(k) != curr.get(k):
            diffs.append(f"{k}: {prev.get(k)!r} -> {curr.get(k)!r}")
    return diffs


# ---------------------------------------------------------------------------
# 실제 프로세서 연동 (main()에서만 호출)
# ---------------------------------------------------------------------------

def load_one_sample(args):
    if args.synthetic:
        from snuai.data.synthetic import make_dataset
        return make_dataset(1, seed=42)[0]
    from snuai.data.sample import load_csv
    samples = load_csv(args.csv, args.image_dir, caption_col=args.caption_col)
    if args.holdout_val:
        from snuai.data.split import split_samples
        _, samples = split_samples(samples, val_frac=args.val_frac)
    return samples[0]


def build_messages(sample, args):
    from snuai.prompting import build_reranker_messages
    return build_reranker_messages(
        sample.caption, sample.load_images(), video_mode=args.video_mode,
        dup_factor=args.video_dup_factor)


def _vision_token_count_by_grid(enc, processor) -> int | None:
    """image_grid_thw/video_grid_thw로부터 비전 토큰 수 산출 (하드코딩 없는 1차 근거)."""
    for grid_key, proc_attr in (("image_grid_thw", "image_processor"),
                               ("video_grid_thw", "video_processor")):
        if grid_key in enc:
            grid_thw = enc[grid_key]
            sub = getattr(processor, proc_attr, None)
            merge_size = getattr(sub, "merge_size", None)
            if merge_size:
                return int((grid_thw.prod(dim=-1) // (merge_size ** 2)).sum().item())
    return None


def audit_one(processor, sample, args) -> tuple[dict, list[int], set[int]]:
    from snuai.prompting import call_processor, extract_media

    msgs = build_messages(sample, args)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    images, videos = extract_media(msgs)
    enc = call_processor(processor, [text], images, videos, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    decoded = processor.tokenizer.decode(ids)

    # vision 토큰 id: processor가 직접 노출하는 image_token_id/video_token_id를
    # 1순위로 쓰고(하드코딩 아님 — transformers Qwen3VLProcessor가 초기화 시점에
    # tokenizer.image_token_id 폴백까지 처리해서 설정함), grid_thw 기반 산출과 교차검증.
    image_tok_id = getattr(processor, "image_token_id", None)
    video_tok_id = getattr(processor, "video_token_id", None)
    vision_ids = {i for i in (image_tok_id, video_tok_id) if i is not None}
    vision_count_by_id = sum(1 for t in ids if t in vision_ids) if vision_ids else None
    vision_count_by_grid = _vision_token_count_by_grid(enc, processor)

    if vision_count_by_id is not None and vision_count_by_grid is not None \
            and vision_count_by_id != vision_count_by_grid:
        print(f"[warn] vision 토큰 수 불일치: placeholder-id={vision_count_by_id} "
              f"grid={vision_count_by_grid} — 프로세서 버전/속성명 확인 필요", file=sys.stderr)
    vision_total = vision_count_by_grid if vision_count_by_grid is not None else vision_count_by_id

    grid_thw = enc.get("image_grid_thw", enc.get("video_grid_thw"))
    facts = {
        "frame_count": len(images) if images else (len(videos[0]) if videos else 0),
        "grid_thw": grid_thw.tolist() if grid_thw is not None else None,
        "vision_tokens_total": vision_total,
        "vision_tokens_by_placeholder_id": vision_count_by_id,
        "vision_tokens_by_grid": vision_count_by_grid,
        "text_tokens_total": len(ids) - vision_total if vision_total is not None else None,
        "total_len": len(ids),
        "image_label_count": count_image_labels(decoded),
        "candidate_label_count": len(_CANDIDATE_LABEL_RE.findall(decoded)),
        "timestamp_count": count_timestamps(decoded),
        "recipe": {
            "video_mode": args.video_mode, "dup_factor": args.video_dup_factor,
            "max_pixels": args.max_pixels, "video_max_pixels": args.video_max_pixels,
            "model_id": args.model_id,
            "scoring_contract": "qwen3_vl_reranker_yes_no",
        },
    }
    return facts, ids, vision_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from snuai.model_config import add_model_id_argument
    d = ap.add_argument
    d("--csv"); d("--image-dir"); d("--caption-col", default="Caption")
    d("--holdout-val", action="store_true"); d("--val-frac", type=float, default=0.1)
    d("--synthetic", type=int, default=0, help="1 이상이면 합성 샘플 사용(GPU/실데이터 불필요)")
    add_model_id_argument(ap)
    d("--video-mode", action=argparse.BooleanOptionalAction, default=False)
    d("--video-dup-factor", type=int, default=1)
    d("--video-max-pixels", type=int, default=None)
    d("--max-pixels", type=int, default=602112)
    d("--out", required=True)
    d("--diff", default=None, help="이전 덤프 JSON 경로 — 구조 diff 검사")
    d("--allow-diff", default="", help="쉼표구분 필드명 — 의도된 diff는 이 목록만 허용")
    args = ap.parse_args(argv)

    if args.video_dup_factor > 1 and not args.video_mode:
        raise SystemExit("--video-dup-factor>1은 --video-mode에서만 유효")
    if args.video_dup_factor != 1 and args.video_dup_factor % 2 != 0:
        raise SystemExit("--video-dup-factor는 1 또는 짝수만 유효 "
                         "(홀수는 temporal 병합쌍이 프레임 블록과 어긋나 일부 교차 오염됨)")

    from snuai import perm
    from snuai.infer.engine import apply_pixel_budget
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_id)
    video_max_pixels = args.video_max_pixels
    if video_max_pixels is None and args.video_mode and args.video_dup_factor > 1:
        video_max_pixels = perm.N * args.video_dup_factor * args.max_pixels
    apply_pixel_budget(processor, max_pixels=args.max_pixels, video_max_pixels=video_max_pixels)

    sample = load_one_sample(args)
    facts, ids, vision_ids = audit_one(processor, sample, args)
    facts["recipe"]["video_max_pixels"] = video_max_pixels

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    out_path.with_suffix(".txt").write_text(
        collapse_vision_runs(ids, vision_ids), encoding="utf-8")

    print(json.dumps(facts, ensure_ascii=False, indent=2))
    print(f"[audit] 덤프 저장 → {out_path} (+ .txt 콜랩스 뷰)")

    if args.diff:
        prev = json.loads(Path(args.diff).read_text(encoding="utf-8"))
        allow = {s.strip() for s in args.allow_diff.split(",") if s.strip()}
        diffs = diff_facts(prev, facts, allow)
        if diffs:
            print(f"[audit] {args.diff} 대비 예상치 못한 구조 diff 발견:")
            for d_ in diffs:
                print("  -", d_)
            sys.exit(1)
        print(f"[audit] {args.diff} 대비 허용된 diff 외 구조 변화 없음 — 통과")


if __name__ == "__main__":
    main()
