# SNU AI Challenge — Qwen3-VL Reranker

This repository ranks four shuffled video frames into chronological order for the
SNU AI Challenge. The inference pipeline uses
[`Qwen/Qwen3-VL-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)
as a pointwise multimodal reranker.

## How it works

For every sample, the storyline is used as the query. Each of the 24 possible frame
orders is represented as a candidate multimodal document. Qwen3-VL-Reranker scores
all candidates with its native `yes`/`no` relevance head, and the highest-scoring
permutation is converted to the challenge submission format.

The refactored inference path preserves the existing canonical permutation algebra,
TTA remapping, resumable progress log, time budget guard, optional pairwise cascade,
and submission validation.

## Setup

Python 3.10+ and a CUDA-capable environment are recommended. The 8B checkpoint is
BF16 by default; use `--four-bit` only when the quality/VRAM tradeoff is acceptable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Inference

```bash
python -m snuai.infer.predict \
  --csv data/test.csv \
  --image-dir data/test \
  --model-id Qwen/Qwen3-VL-Reranker-8B \
  --tta 1 \
  --out runs/reranker
```

Useful options:

- `--four-bit`: load with bitsandbytes NF4 on CUDA.
- `--tta N`: score shuffled test-time views and map them back to canonical ranks.
- `--cascade --tau 0.15`: recheck uncertain top candidates with reranker-native
  pairwise comparisons.
- `--video-mode`: encode each proposed order through the video channel.
- `--limit N`: run a small subset before a full submission.

GPU smoke and budget checks:

```bash
python scripts/smoke_gpu.py
python scripts/bench_3090.py --csv data/train.csv --image-dir data/train --n 12
```

CPU-only end-to-end pipeline check:

```bash
python -m snuai.infer.predict --synthetic 24 --strategy dummy --tta 3 --eval --out runs/dryrun
```

## Validation

```bash
pytest -q
```

The legacy score24 SFT/DPO modules remain available for experiment reproducibility,
but their adapters target generative Qwen3-VL checkpoints and are intentionally not
accepted by the new reranker inference engine.
