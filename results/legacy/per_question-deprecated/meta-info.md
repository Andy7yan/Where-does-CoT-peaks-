# Per-Question Phase2 Input Notes

## Purpose

This note records:

- the field definitions of the Phase 2 input tables used for `t1b` / `t1b_norm`, `t1c`, and `t2b`
- why a question is counted as `qualified`
- the final qualified question counts for `medium` and `hard`

Source PQ analysis directory:

- `results/per-question-0419_143551/pq_analysis`

Important note:

- `t1b` and `t1b_norm` use the same input table: `t1b_step_surface.csv`

## Phase2 Input Fields

### `t1b_step_surface.csv`

```csv
question_id,L,step,mean_nldd,nldd_se,mean_tas_t,tas_t_se,n_clean,bin_status
```

### `t1c_kstar_ratio.csv`

```csv
question_id,difficulty_score,L,k_star,k_star_ratio,n_clean
```

### `t2b_lstar_difficulty.csv`

```csv
question_id,difficulty_score,l_star_A,l_star_S,l_star_consistent
```

## Why A Question Is Qualified

A question is counted as `qualified` only if it passes all of the stage filters below.

### Stage 1: Source Question Selection

The per-question pipeline only starts from questions that were selected from the source run as:

- `difficulty_bucket = medium` or
- `difficulty_bucket = hard`

`easy` questions are not part of the per-question branch.

### Stage 2: Per-Question Metadata Gate

For each question, PQ metadata is computed from all generated traces:

- `acc_pq = correct_count / total_traces`
- `difficulty_score = 1 - acc_pq`
- `degenerate = (acc_pq == 0) or (acc_pq == 1)`

To remain qualified, a question must satisfy:

- `degenerate = False`

In other words, the question must not be trivially always-wrong or always-correct under the PQ run.

### Stage 3: L-Curve Validity Gate

The question's exact-length L-curve is built using only bins that satisfy:

- `L >= min_nldd_length = 3`
- number of traces in that exact-length bin `>= per_question_lcurve_min_bin_size = 5`

Then the question-level L-curve is considered sufficient only if:

- it is not degenerate, and
- `valid_lcurve_bins >= per_question_min_lcurve_bins = 3`

So to remain qualified, a question must satisfy:

- `l_curve_insufficient = False`

This means the question has enough valid length bins to support stable `L*` estimation.

### Stage 4: Per-Length Sample Retention Gate

For each exact length `L`, only correct traces are considered for corruption/sample export.

Then, for a correct trace to become a retained clean sample, it must pass all corruption checks:

- corruption is built for every required `k` position
- the trace must have a complete corruption sweep for `k = 2..L`
- none of those selected corruption rows may have `corruption_failed = True`

A length bin is marked `ok` only if the number of retained complete samples satisfies:

- `n_retained >= per_question_min_retained_traces = 5`

At most:

- `per_question_max_retained_traces = 20`

are kept in that bin.

This stage determines whether a question has enough usable bins for downstream NLDD / TAS / `k*`.

### Stage 5: k* Sufficiency Gate

After PQ analysis:

- `k_star` is resolved only from bins that are actually usable after the earlier retention filters
- a question is considered sufficient for `k*` only if it has:
  `resolved_kstar_bins >= per_question_min_kstar_bins = 2`

So to remain qualified, a question must satisfy:

- `k_star_insufficient = False`

### Final Qualified Definition

A question is counted as qualified if and only if all three final flags satisfy:

- `degenerate = False`
- `l_curve_insufficient = False`
- `k_star_insufficient = False`

Operationally, these qualified questions are exactly the questions that successfully appear as usable inputs across the PQ Phase 2 views:

- `t1b_step_surface.csv`
- `t1c_kstar_ratio.csv`
- `t2b_lstar_difficulty.csv`

## Final Qualified Counts

- `qualified_medium = 149`
- `qualified_hard = 77`

