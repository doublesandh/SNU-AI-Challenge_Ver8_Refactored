"""GPU 스모크 테스트 — 모델 로딩·규약·전 경로 1샘플 확인 (3090/VESSL 도착 직후 1회).

검사 항목:
  1. 4bit 로딩 + vision 비양자화 확인 (규약)
  2. reranker yes/no 토큰 확인
  3. 24-way permutation rerank / pairwise / video 각 1회 실행 + VRAM 피크

예:
  python scripts/smoke_gpu.py
  python scripts/smoke_gpu.py --four-bit
"""

from __future__ import annotations

import argparse
import time


def main():
    from snuai.model_config import add_model_id_argument

    ap = argparse.ArgumentParser()
    add_model_id_argument(ap)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--prequantized", action="store_true",
                    help="bnb-4bit 체크포인트(자체 quant config)")
    ap.add_argument("--max-pixels", type=int, default=602112)
    args = ap.parse_args()

    import torch
    from snuai.data.synthetic import make_dataset
    from snuai.infer.engine import EngineConfig, Qwen3VLRerankerEngine
    from snuai.infer.scorers import PermutationReranker, RerankerPairwiseJudge
    from snuai.train.qlora import verify_vision_not_quantized

    eng = Qwen3VLRerankerEngine(EngineConfig(
        model_id=args.model_id,
        four_bit=args.four_bit and not args.prequantized,
        max_pixels=args.max_pixels))
    print(f"[load] attn={eng.attn_used} device={eng.device}")
    if args.four_bit or args.prequantized:
        print("[quant]", verify_vision_not_quantized(eng.model))
    assert eng.yes_id != eng.no_id
    print("[tokens] reranker yes/no token ids distinct ✅")

    s = make_dataset(1, seed=0)[0]
    imgs = s.load_images()

    def timed(name, fn):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic(); out = fn(); dt = time.monotonic() - t0
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        print(f"[{name}] {dt:.2f}s | peak VRAM {vram:.2f} GiB")
        return out

    sc = PermutationReranker(eng)
    scores = timed("reranker(24 forward)", lambda: sc.scores(s.caption, imgs))
    print("        score range:", float(scores.min()), float(scores.max()))

    judge = RerankerPairwiseJudge(eng)
    timed("pairwise(2 forward)", lambda: judge.p_earlier(s.caption, imgs[0], imgs[1]))

    video_sc = PermutationReranker(eng, video_mode=True)
    timed("video_mode(24 forward)", lambda: video_sc.scores(s.caption, imgs))

    print("\n스모크 통과 — bench_3090.py로 예산 실측, exp_preproc_ab.py로 전처리 A/B 진행")


if __name__ == "__main__":
    main()
