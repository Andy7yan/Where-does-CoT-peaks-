# Katana HPC: Transferable Engineering Experience

This document records concrete, battle-tested patterns for running ML projects on UNSW Katana. It is written for agents and engineers bootstrapping a **new** project that will reuse the same infrastructure.

**Important:** some GPU-related notes from **SAE-v1** are retained below as **examples only**. They are not canonical requirements for a new project. Their role is to provide a concrete debugging reference for model loading, forward pass, activation capture, and GPU utilisation issues.

---

## Table of Contents

1. [Katana Two-Zone Storage Model](#1-katana-two-zone-storage-model)
2. [Directory Layout & Naming Conventions](#2-directory-layout--naming-conventions)
3. [PBS Job System](#3-pbs-job-system)
4. [Environment Setup](#4-environment-setup)
5. [Data & Model Pre-download](#5-data--model-pre-download)
6. [GPU Execution Patterns and Debug Reference](#6-gpu-execution-patterns-and-debug-reference)
7. [Practical Gotchas](#7-practical-gotchas)

---

## 1. Katana Two-Zone Storage Model

Katana has two storage zones with very different characteristics. **Every file you create must go to the correct zone.**

| Zone | Path | Quota | Backed up? | What goes here |
|------|------|-------|------------|----------------|
| **Home** | `/home/<project>` | Small (~10 GB) | Yes | Git repo, source code, PBS scripts, config files, `.venv/` |
| **Scratch** | `/srv/scratch/$USER/<project>` | Large (~1 TB) | **No** | Model weights, datasets, HF cache, run outputs (logs/checkpoints/weights), large intermediate files |

### Core Rules

1. **Never write logs, checkpoints, model weights, or generated data back to home.** Home has tiny quota and is on slow networked storage.
2. **Scratch is not backed up.** Anything you need to reproduce a run (code, config) must live in home (i.e., in git). Scratch holds things that can be regenerated or re-downloaded.
3. **Clean up scratch after experiments.** Old runs accumulate fast. Delete stale `runs/` subdirectories regularly.
4. **Compute nodes have limited or no internet.** All downloads (models, datasets) must happen on a login node before submitting jobs. See [Section 5](#5-data--model-pre-download).

---

## 2. Directory Layout & Naming Conventions

### 2.1 Home directory (codebase)

The home directory on Katana is a **mirror of your local project root**. You push code locally, pull on Katana (or rsync), and the structure is identical.

### 2.2 Scratch directory (heavy resources)

All large/generated artifacts go under `/srv/scratch/$USER/<project>/`:

```text
/srv/scratch/$USER/<project>/
├── hf-home/                        # HF_HOME: Hugging Face cache root
│   ├── hub/                        # model snapshots (auto-managed by huggingface_hub)
│   ├── datasets/                   # HF datasets cache
│   └── saved-datasets/             # parsed/processed datasets saved via save_to_disk()
├── data/                           # project-specific data files (JSONL exports, etc.)
├── datasets/                       # aggregated/processed dataset bundles
└── runs/                           # one subdirectory per experiment run
    └── <run-name>/
        ├── logs/
        ├── checkpoints/
        └── weights/
```

### 2.3 Naming conventions

#### Project name
Lowercase kebab-case: `my-new-project`. Same name for home dir and scratch root.

#### Run names
Format: `<purpose>-<MMDD_HHMMSS>`

Examples:
```text
latent-gen-0402_143522
train-0403_091000
eval-0404_180000
```

- `<purpose>`: short kebab-case label.
- Timestamp: `$(date +%m%d_%H%M%S)`.
- Override with `RUN_NAME` env var for a custom name.

#### Dataset directories
Under `datasets/` in scratch. Named descriptively with version suffix.

#### HF cache
Always set `HF_HOME` to scratch: `export HF_HOME=/srv/scratch/$USER/<project>/hf-home`. This prevents HF from writing multi-GB caches to home quota.

### 2.4 What goes where — quick reference

```text
Source code, config, PBS scripts          → Home (/home/<project>/)
Model weights, datasets, caches (> MB)   → Scratch (/srv/scratch/$USER/<project>/)
Training/eval/inference run outputs       → Scratch, under runs/<run-name>/
Secrets (HF_TOKEN, API keys)             → Home, in .env (git-ignored)
Python virtual environment               → Home, in .venv/
```

---

## 3. PBS Job System

Katana uses **PBS** (Portable Batch System), **not SLURM**. Commands: `qsub`, `qstat`, `qdel`.

### 3.1 PBS script template

Every `.pbs` script follows this structure:

```bash
#!/bin/bash
#PBS -N <job_name>
#PBS -l select=1:ncpus=<N>:ngpus=<G>:mem=<M>gb
#PBS -l walltime=HH:MM:SS
#PBS -j oe
#PBS -o /srv/scratch/${USER}/<project>/runs/<run>/logs/<logfile>.log

# --- Phase 1: Environment ---
cd /home/<project>
source .venv/bin/activate
export HF_HOME=/srv/scratch/${USER}/<project>/hf-home

# --- Phase 2: Run directory ---
: "${RUN_NAME:=<purpose>-$(date +%m%d_%H%M%S)}"
RUN_DIR=/srv/scratch/${USER}/<project>/runs/${RUN_NAME}
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints" "${RUN_DIR}/weights"

# --- Phase 3: Launch (keep this as simple as possible) ---
python src/<script>.py \
    --output-dir "${RUN_DIR}" \
    --run-name "${RUN_NAME}"
```

### 3.2 Key PBS directives

| Directive | Typical value | Notes |
|-----------|--------------|-------|
| `select` | `1:ncpus=4-8:ngpus=1:mem=16-64gb` | Single node. Start minimal, scale up after validation. |
| `walltime` | `04:00:00` to `11:50:00` | Katana hard limit is 12h. Set just under (11:50:00) for long jobs. |
| `-j oe` | always set | Merges stdout+stderr into one `.oNNNNNN` file. |
| `-o` | explicit log path in scratch | If omitted, output lands in `$PBS_O_WORKDIR`. |

### 3.3 Job lifecycle commands

```bash
qsub jobs/train.pbs                           # submit
qsub -v RUN_NAME=my-experiment jobs/train.pbs # submit with override
qstat -u $USER                                # check status
qdel <JOBID>                                  # kill
```

### 3.4 PBS script design rules

1. **PBS directive block first** (`#PBS` lines), then execution block. No code between `#!/bin/bash` and the last `#PBS` line.
2. **The launch command must be trivially simple.** If you find yourself writing complex bash logic in the PBS script, move it into the Python entry point instead.
3. **Use `: "${VAR:=default}"` for configurable parameters.** This lets you override via `qsub -v VAR=value` without editing the script.
4. **Always `cd` to the codebase first** so relative imports and paths work.

---

## 4. Environment Setup

### 4.1 `.env` file

Lives at `/home/<project>/.env`, **git-ignored**. Contains:

```bash
export HF_TOKEN="hf_..."
```

HF_TOKEN is read by Python code via `os.getenv("HF_TOKEN")`.

### 4.2 Key environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `/srv/scratch/$USER/<project>/hf-home` | HF cache root (keeps it off home quota) |
| `HF_HUB_CACHE` | `${HF_HOME}/hub` | Model snapshots |
| `HF_DATASETS_CACHE` | `${HF_HOME}/datasets` | Dataset downloads |
| `HF_TOKEN` | from `.env` | Auth for gated models/datasets |

---

## 5. Data & Model Pre-download

**Compute nodes have limited or no outbound internet.** Everything must be pre-downloaded on a login node before job submission.

### 5.1 Hugging Face models

```bash
# On login node:
source .venv/bin/activate
export HF_TOKEN="hf_..."
export HF_HOME=/srv/scratch/$USER/<project>/hf-home

huggingface-cli download <repo-id> \
    --cache-dir "${HF_HOME}/hub" \
    --token "$HF_TOKEN"
```

If the model is gated, you must first accept the licence on the HF web UI.

### 5.2 Hugging Face datasets

```bash
python -c "
from datasets import load_dataset
load_dataset('<repo-id>', cache_dir='$HF_HOME/datasets')
"
```

Or use a project-specific `prepare_data.py` that exports to JSONL in scratch.

### 5.3 Verification before submitting

```bash
ls /srv/scratch/$USER/<project>/hf-home/hub/models--*   # model snapshots
ls /srv/scratch/$USER/<project>/data/                    # data files
```

---

## 6. GPU Execution Patterns and Debug Reference

This section keeps only the parts of **SAE-v1** that may help diagnose a new project's GPU issues. These are **examples**, not architectural requirements.

### 6.1 Minimal execution assumption

For GPU utilisation to rise above zero, the following chain must actually happen inside the job:

1. model is loaded successfully
2. model is moved to CUDA
3. input tensors are moved to the same CUDA device
4. a real forward pass is executed
5. the forward pass is not bypassed by empty batches, early return, or CPU-only code paths

If any one of these steps fails, the job may still look "running" while GPU utilisation remains near zero.

### 6.2 Minimal single-GPU inference pattern

```python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_or_load_model()
model.to(DEVICE)
model.eval()

batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
batch = {k: v.to(DEVICE) for k, v in batch.items()}

with torch.no_grad():
    outputs = model(**batch)
```

The three checks that matter most:

```python
next(model.parameters()).device
batch["input_ids"].device
torch.cuda.is_available()
```

If model and inputs are not on the same CUDA device, PyTorch either errors immediately or silently falls back to a code path that does not perform the intended GPU work.

### 6.3 Forward-hook example from SAE-v1

SAE-v1 captured intermediate activations from a frozen LLM using a forward hook, then sometimes aborted the rest of the model with a custom exception after the target layer was reached. This is retained here **only as a debugging example**, because a similar pattern can make a job appear active while doing very little GPU work.

Typical pattern:

```python
acts = {}

def hook_fn(module, inputs, output):
    acts["target"] = output.detach()

handle = target_module.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(**batch)

handle.remove()
```

Debug implications:

1. If the hook is attached to the wrong module, the expected tensor may never be captured.
2. If the code exits before `model(**batch)`, the hook never fires.
3. If a custom early-stop exception is thrown too early, most of the model never runs.
4. If the captured activation is immediately moved back to CPU every step, GPU may look underutilised because the pipeline becomes transfer-bound.

### 6.4 SAE-v1 example: why GPU utilisation stayed low even when CUDA was used

In SAE-v1, low GPU utilisation was not caused by "CUDA not working". The more likely issue was that the GPU repeatedly waited for upstream CPU work:

- streaming and decompressing data
- tokenisation and filtering
- activation buffering and reshaping
- host-to-device transfer

That example matters because **GPU util = 0 or very low does not automatically mean the model stayed on CPU**. It may also mean the forward pass is extremely sparse, tiny, blocked, or starved.

### 6.5 Fast diagnosis order for utilisation = 0

Check in this order:

1. **Did the job request a GPU at all?** Confirm `#PBS -l select=1:ncpus=...:ngpus=1:...`.
2. **Does PyTorch see CUDA?** Log `torch.cuda.is_available()` and `torch.cuda.device_count()` at startup.
3. **Is the model on CUDA?** Log `next(model.parameters()).device` after `model.to(device)`.
4. **Are the inputs on CUDA?** Log one representative tensor device right before forward.
5. **Is forward actually executed?** Put logging immediately before and after `model(**batch)`.
6. **Are batches empty?** Log batch size after token filtering / collation.
7. **Is the code stuck before forward?** Time data loading, tokenisation, and batch preparation separately.
8. **Is a hook or exception terminating the pass early?** Temporarily disable early-stop logic.
9. **Is the batch too small to register visible util?** Tiny batches can show near-zero average GPU utilisation even if CUDA is functioning.

### 6.6 Logging that is worth keeping

At minimum, log these once per job:

```python
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
print("model_device=", next(model.parameters()).device)
print("input_device=", batch["input_ids"].device)
print("batch_shape=", batch["input_ids"].shape)
```

If GPU util is still zero, add timing around these phases:

- model load
- dataset read
- tokenisation
- batch collation
- host-to-device copy
- forward pass

This usually separates **GPU not used** from **GPU waiting on CPU**.

### 6.7 Resource note from SAE-v1

SAE-v1 was single-node, GPU-backed, and its useful pattern here is simple: keep the GPU job structurally minimal. Do not bury model loading, batch construction, activation capture, fallback logic, and debugging branches inside one large script without timings. When utilisation is abnormal, reduce the path to:

1. load one model
2. prepare one batch
3. move batch to CUDA
4. run one forward pass
5. verify one tensor comes back from GPU

Only after this works should hooks, buffering, latent extraction, or downstream processing be reintroduced.

---

## 7. Practical Gotchas

Each one cost at least one failed job.

### 7.1 Log file duplication
Add `exec > >(tee -a "$RUN_DIR/logs/train.log") 2>&1` in the PBS script to write to both the PBS `.oNNNNNN` file and a persistent log in scratch.

### 7.2 HF cache in home
If you forget to set `HF_HOME` to scratch, HF writes multi-GB caches to `~/.cache/huggingface` inside home quota. Causes cryptic "disk quota exceeded" errors. **Always export `HF_HOME` to scratch.**

### 7.3 Gated model auth
Missing or expired `HF_TOKEN` causes 401 errors that look like network issues. Verify the token on a login node before submitting.

### 7.4 Shared caches across runs
Caches like processed datasets are shared. If you change model architecture or dataset schema, **delete the cache** so it gets recomputed. Stale caches cause silent correctness bugs.

### 7.5 PBS `-o` path must exist
If the `-o` log directory doesn't exist at job start, PBS silently drops stdout/stderr. Always `mkdir -p` the log directory **before** `qsub`, or use a fixed path you've already created.

### 7.6 Module load order
`module load ...` must come **before** `source .venv/bin/activate`. Loading the module after activation resets the Python path.

---

*Last updated: 2026-04-04.*

