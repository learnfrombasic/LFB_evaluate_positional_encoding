# Machine evaluation

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA Quadro P2000, 4096 MiB VRAM, driver 575.57.08 |
| CPU | Intel i9-8950HK, 12 threads |
| RAM | 30 GiB total, ~24 GiB available |
| Disk | 468 GB volume, 166 GB free |

## Findings

1. **VRAM is the binding constraint.** The P2000 is Pascal-generation (no
   tensor cores) with only 4 GB of VRAM. Web research done this session
   confirms full BERT-style pretraining from scratch on 4 GB VRAM is
   impractical — this matches the README's note that the project was
   "stopped due to lack of computational resources." Conclusion: treat any
   experiment here as a **short comparative feasibility study**, not a
   publication-grade pretraining run.

2. **Driver/CUDA wheel mismatch — GPU is currently unusable as pinned.**
   `requirements.txt` pins `torch==2.12.0`, which resolves (from the default
   PyPI/PyTorch index) to a `+cu130` build requiring a CUDA 13.0-capable
   driver. The installed driver (575.57.08) only supports up to CUDA 12.9,
   so `torch.cuda.is_available()` returns `False` even though the GPU is
   present and idle. Checked whether an older CUDA wheel exists for the same
   torch version (`pip index versions torch --index-url .../whl/cu124` and
   `.../whl/cu121`): neither index publishes a `2.12.0` build (they top out
   at `2.6.0+cu124` / `2.5.1+cu121`), so getting GPU support back would mean
   either (a) updating the NVIDIA driver to a CUDA-13-capable version, or
   (b) downgrading `torch` (and re-verifying compatibility with the pinned
   `transformers==5.12.0`). Both are dependency/system changes outside the
   scope of "run safe" for this session, so **experiments in this pass run
   on CPU** instead of touching the driver or destabilizing the pinned
   dependency set.
   - Remediation for a future session with more time: try `torch==2.6.0+cu124`
     in a scratch venv, run the test suite, and only adopt it project-wide if
     `transformers==5.12.0` still imports/runs cleanly against it.

3. **CPU is adequate for the reduced experiment scope.** 12 threads, no
   swap pressure, 24 GB free RAM. A 2-layer, small-hidden-size BERT on a
   short subset with a capped step count trains in a few minutes per run on
   CPU — acceptable for a same-day comparative sweep across PE types.

## Resulting experiment configuration decisions

- Sequence length reduced from 512 → 128 (the configured dataset's typical
  example length doesn't need the full 512 budget for a short comparison).
- Batch size kept at 8, no gradient accumulation needed at this scale.
- `--subset` (already supported in `src/__main__.py`) used to cap train/val
  rows instead of downloading the full `8Opt/bert-mlm-experiments-en` split.
- `training.max_steps` (new) caps each run to a bounded number of optimizer
  steps regardless of subset size, so wall-clock time per run is predictable.
- Mixed precision left enabled in config but is a no-op on CPU by design
  (see `pe-refactor-notes.md` / training pipeline changes) — it will
  activate automatically if this project later runs on a CUDA-capable box.
