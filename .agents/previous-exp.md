# Katana HPC: Current Project Rules for peak-CoT

This file is the current Katana operating guide for `peak-CoT`.

It supersedes older generic notes that were partly inherited from other projects. If this file conflicts with older advice, follow this file.

The concrete sources behind this version are:

- the current repo job scripts under `jobs/`
- the current runtime behavior in `src/generation.py`
- the read-only GPU helper bundle in `gpu_porting_bundle.py`
- the Katana-related decisions already consolidated during prior project work

Last updated: 2026-04-11.

---

## 1. Scope and Priority

This document is no longer a generic "Katana tips" memo. It is a project-specific contract for how `peak-CoT` is supposed to run on Katana today.

When there is a conflict, use this priority order:

1. Current repository code and scripts
2. This document
3. Older generic notes copied from previous projects

---

## 2. Storage Model

Katana still has the usual two-zone model, but the current project-specific placement is:

| Zone | Canonical path for this project | What belongs here |
|------|---------------------------------|-------------------|
| Home | `/home/${USER}/peak-CoT` | Git repo, source code, configs, prompts, PBS scripts, `.env` |
| Scratch | `${SCRATCH}` and in practice usually `/srv/scratch/${USER}/peak-CoT` | `.venv`, HF caches, exported data, run outputs, logs, checkpoints, weights |

### 2.1 Important correction

Older notes said `.venv` lives in home. That is not the current project convention.

For `peak-CoT`, the current scripts and checks assume:

- virtual environment: `${SCRATCH}/.venv`
- Hugging Face cache root: `${SCRATCH}/hf-home`
- run outputs: `${SCRATCH}/runs/${RUN_NAME}`

### 2.2 Practical rule

Do not write any large or regenerable artifact back to `/home/${USER}/peak-CoT`.

That includes:

- model snapshots
- dataset caches
- generated traces
- run logs
- checkpoints
- temporary large intermediates

---

## 3. Current Directory Layout on Katana

### 3.1 Home

```text
/home/${USER}/peak-CoT/
├─ src/
├─ scripts/
├─ jobs/
├─ prompts/
├─ configs/
├─ tests/
├─ .env
└─ pyproject.toml
```

### 3.2 Scratch

```text
${SCRATCH}/
├─ .venv/
├─ hf-home/
│  ├─ hub/
│  └─ datasets/
├─ data/
└─ runs/
   └─ ${RUN_NAME}/
      ├─ logs/
      ├─ checkpoints/
      └─ weights/
```

### 3.3 Project naming

For the current repo, the project directory name is fixed:

- home: `/home/${USER}/peak-CoT`
- scratch root: `/srv/scratch/${USER}/peak-CoT` or the equivalent path exported through `SCRATCH`

Do not use `/home/peak-CoT` without `${USER}`. That exact form is already known to be wrong.

---

## 4. `.env` Contract

### 4.1 Location

The `.env` file lives in:

```bash
/home/${USER}/peak-CoT/.env
```

It is git-ignored and sourced by PBS scripts.

### 4.2 Important correction

Older notes implied `.env` only needs `HF_TOKEN`.

That is no longer sufficient for this project.

### 4.3 Variables that must exist

At minimum, `.env` should export:

```bash
export SCRATCH=/srv/scratch/${USER}/peak-CoT
export HF_HOME=${SCRATCH}/hf-home
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_TOKEN="hf_..."
```

Optional but often useful:

```bash
export PROJECT_HOME=/home/${USER}/peak-CoT
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
```

### 4.4 How scripts should source `.env`

For this project, the stable pattern is:

```bash
set +u
set -a
source "${PROJECT_HOME}/.env"
set +a
set -u
```

or, when already in the repo root:

```bash
set +u
set -a
source ".env"
set +a
set -u
```

Do not treat `source .venv/bin/activate` as the standard environment bootstrap for current `peak-CoT` jobs.

---

## 5. PBS Job Rules

Katana uses PBS, not SLURM. Commands remain:

- `qsub`
- `qstat`
- `qdel`

### 5.1 Header rules

Keep the PBS header at the very top:

```bash
#!/bin/bash
#PBS -N <job_name>
#PBS -l select=1:ncpus=<N>:ngpus=<G>:mem=<M>gb
#PBS -l walltime=HH:MM:SS
#PBS -j oe
#PBS -V
```

No shell code should appear between the shebang and the last `#PBS` line.

### 5.2 Current peak-CoT template for normal jobs

For normal generation / subset / NLDD jobs, the current project pattern is:

```bash
set -euo pipefail

PROJECT_NAME="peak-CoT"
: "${PROJECT_HOME:=/home/${USER}/${PROJECT_NAME}}"
: "${RUN_NAME:=generate-$(date +%m%d_%H%M%S)}"

cd "${PROJECT_HOME}" || exit 1

if [[ ! -f "${PROJECT_HOME}/.env" ]]; then
  echo "missing .env at ${PROJECT_HOME}/.env"
  exit 1
fi

set +u
set -a
source "${PROJECT_HOME}/.env"
set +a
set -u

: "${SCRATCH:?SCRATCH must be set}"
: "${HF_HOME:?HF_HOME must be set}"
: "${HF_HUB_CACHE:?HF_HUB_CACHE must be set}"
: "${HF_DATASETS_CACHE:?HF_DATASETS_CACHE must be set}"

RUN_DIR="${SCRATCH}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/weights"
exec > >(tee -a "${RUN_DIR}/logs/<stage>.log") 2>&1
```

### 5.3 Important correction about log strategy

Older notes treated the dynamic `PBS_JOBID`-based log directory as the default modern strategy, and described `-o` as deprecated.

That is not the current `peak-CoT` rule.

Current project behavior is:

- normal jobs keep their persistent log in `${RUN_DIR}/logs/<stage>.log`
- `exec > >(tee -a ...) 2>&1` is the primary persistent logging mechanism
- `#PBS -o ...` is still allowed in this repo and is already used by some scripts
- the dynamic `PBS_JOBID` log directory is a special-case pattern, not the universal template

### 5.4 When the dynamic log directory pattern is still valid

`jobs/env_test.pbs` uses a dynamic directory like:

```bash
RUN_DATE=$(date +%m%d)
JOB_ID_SHORT=${PBS_JOBID%%.*}
JOB_ID_SHORT=${JOB_ID_SHORT: -4}
LOG_DIR="${SCRATCH}/runs/smoke-${RUN_DATE}_${JOB_ID_SHORT}"
```

That pattern is still valid when a job needs to validate environment state before a normal `RUN_DIR` flow is established.

It is not the standard template for every other job in this project.

### 5.5 Resource profile for current Stage 1 jobs

The current repo uses single-node, single-GPU jobs.

Typical values already present in the repo:

- generate / smoke / nldd: `select=1:ncpus=4:ngpus=1:mem=24gb`
- prepare subset: CPU-only, `select=1:ncpus=2:mem=8gb`
- walltime upper bound: keep jobs at or below `11:50:00`

### 5.6 Known broken pattern that must not be copied

This is wrong for current `peak-CoT`:

- `cd /home/peak-CoT`
- `source .venv/bin/activate`

If you see that pattern, treat it as stale or broken.

Use:

- `cd "/home/${USER}/peak-CoT"`
- `source ".env"` or `source "${PROJECT_HOME}/.env"`

---

## 6. Data and Model Pre-download

The older general rule is still correct: compute nodes may have no internet or unreliable outbound access.

For this project, pre-download on a login node is the safe default.

### 6.1 Model snapshots

Current code in `src/generation.py` expects the Hugging Face snapshot to live in the hub cache under scratch.

Acceptable approaches on a login node:

```bash
export SCRATCH=/srv/scratch/${USER}/peak-CoT
export HF_HOME=${SCRATCH}/hf-home
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_TOKEN="hf_..."
```

Then either:

```bash
huggingface-cli download <repo-id> --cache-dir "${HF_HUB_CACHE}" --token "${HF_TOKEN}"
```

or use Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="<repo-id>",
    cache_dir="${HF_HUB_CACHE}",
    token="${HF_TOKEN}",
)
```

### 6.2 Datasets

Preload datasets into:

```bash
${HF_DATASETS_CACHE}
```

Typical pattern:

```python
from datasets import load_dataset

load_dataset("<repo-id>", cache_dir="${HF_DATASETS_CACHE}")
```

### 6.3 Verification

Before submitting jobs, verify:

- model snapshot exists under `${HF_HUB_CACHE}`
- dataset cache exists under `${HF_DATASETS_CACHE}`
- project data exports exist under `${SCRATCH}/data`

---

## 7. GPU Runtime Rules

This section is now based on the actual helper bundle in `gpu_porting_bundle.py` plus current project decisions.

### 7.1 Important correction

Older notes treated SAE-v1 GPU behavior as examples only and left the current runtime choice underspecified.

For current `peak-CoT`, the runtime choice is now specific:

- single GPU
- no DDP
- no multi-GPU
- no `torchrun` path for Stage 1

### 7.2 What from `gpu_porting_bundle.py` is canonical to reuse

The bundle is read-only. Reuse by importing; do not edit it.

The intended reusable pieces are:

- `choose_model_dtype`
- `setup_gpu_runtime`
- `CUDATimer`
- `get_gpu_memory_stats`
- `reset_gpu_peak_memory_stats`
- `move_optimizer_state_to_device` when resuming from CPU-loaded checkpoints

### 7.3 What is reference-only, not the current project path

The bundle also contains:

- `torchrun` examples
- DDP bootstrap helpers
- `maybe_wrap_ddp`
- distributed all-reduce helpers

These are reference code, not the current Stage 1 execution path.

For Stage 1, do not enable:

- `torchrun`
- DDP wrapping
- distributed init

### 7.4 Canonical single-GPU setup

If a new Stage 1 GPU path needs runtime setup, the preferred pattern is:

```python
from gpu_porting_bundle import setup_gpu_runtime

runtime = setup_gpu_runtime(
    require_cuda=True,
    init_distributed=False,
    seed=42,
)
```

This is preferable to silently leaving the runtime policy implicit.

### 7.5 Startup logging that should be preserved

Current project startup logs already printed by `LLMGenerator` are the baseline:

```python
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
print("model_device=", model_device)
print("parameter_count=", parameter_count)
print("load_seconds=", round(load_elapsed, 3))
print("cache_hit=", not downloaded)
print("downloaded_model=", downloaded)
print("local_model_path=", local_model_path)
print("tokenizer_class=", self.tokenizer.__class__.__name__)
```

And right before real generation / forward, also keep:

```python
print("input_device=", input_ids.device)
print("batch_shape=", tuple(input_ids.shape))
```

These logs are not optional noise. They are the first-line CUDA sanity check for this project.

### 7.6 Fast diagnosis order for GPU util = 0

Use this order:

1. Did the PBS job request a GPU?
2. Does PyTorch see CUDA?
3. Is the model on CUDA?
4. Are the inputs on CUDA?
5. Does a real forward / generate call happen?
6. Are batches empty or tiny?
7. Is the code blocked before forward on tokenization, I/O, or batching?
8. Is a hook stopping or corrupting the pass?
9. Is the workload too small to show visible utilization?

Low utilization does not automatically mean "the model stayed on CPU". It can also mean the GPU is starved by upstream CPU work.

### 7.7 Forward-hook discipline

When later stages use hooks, follow these rules:

1. Always remove the hook handle in `finally`.
2. Do not raise early-stop exceptions inside the hook.
3. Log the exact target layer explicitly.
4. Prefer one forward that yields both logits and hidden states instead of multiple forwards.

The old hook examples remain useful, but these four rules are the enforceable version.

---

## 8. Explicit Corrections to Older Notes

These points intentionally overwrite earlier contradictory guidance.

### 8.1 Virtual environment location

Old claim:

- `.venv` belongs in home

Current rule:

- use `${SCRATCH}/.venv`

### 8.2 Home path template

Old claim:

- `/home/<project>`

Current rule:

- `/home/${USER}/peak-CoT`

### 8.3 `.env` contents

Old claim:

- `.env` mainly contains `HF_TOKEN`

Current rule:

- `.env` must also define `SCRATCH`, `HF_HOME`, `HF_HUB_CACHE`, and `HF_DATASETS_CACHE`

### 8.4 Standard PBS bootstrap

Old claim:

- normal jobs should `source .venv/bin/activate`

Current rule:

- normal jobs should source `.env`
- the current repo does not use `source .venv/bin/activate` as the standard PBS bootstrap

### 8.5 Log strategy

Old claim:

- dynamic `PBS_JOBID` log directories are the default modern pattern
- `#PBS -o` is deprecated

Current rule:

- standard jobs log to `${RUN_DIR}/logs/<stage>.log` via `tee`
- dynamic job-ID log dirs are special-case, mainly useful for environment checks
- `#PBS -o` is still acceptable and already used in current repo scripts

### 8.6 Runtime mode

Old claim:

- the document stayed non-committal about DDP / `torchrun`

Current rule:

- current Stage 1 on Katana is single-GPU only
- `gpu_porting_bundle.py` is reused selectively
- DDP and `torchrun` helpers are reference-only for now

---

## 9. Practical Gotchas That Still Matter

### 9.1 HF cache accidentally going to home

If `HF_HOME` is not under scratch, Hugging Face may write into the home quota and trigger quota failures.

### 9.2 Gated model auth

If `HF_TOKEN` is missing or stale, model download and refresh paths fail in ways that can look like network issues.

### 9.3 Broken absolute paths in PBS

Hardcoding `/home/peak-CoT` without `${USER}` is a known failure mode.

### 9.4 Logging before run directories exist

If a script needs to validate env state before it can trust `${RUN_DIR}`, the `PBS_JOBID`-based log directory pattern is fine. Otherwise prefer `${RUN_DIR}/logs/<stage>.log`.

### 9.5 Copying the wrong part of `gpu_porting_bundle.py`

The bundle contains both single-GPU helpers and distributed helpers. For this project, do not accidentally copy the distributed path just because it is present.

---

## 10. Minimal Peak-CoT Checklist

Before submitting a normal GPU job, confirm all of the following:

1. `PROJECT_HOME` resolves to `/home/${USER}/peak-CoT`.
2. `.env` exists and exports `SCRATCH`, `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, and `HF_TOKEN`.
3. model snapshots are already in `${HF_HUB_CACHE}`.
4. dataset cache or exported subset is already in scratch.
5. job requests exactly one GPU.
6. script writes persistent logs to `${RUN_DIR}/logs/`.
7. runtime path is single-GPU and does not enable DDP or `torchrun`.
8. startup logs include CUDA visibility, model device, and input device information.
