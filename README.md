# Stage 1: Minimum Viable Pipeline (MVP) & PoC

> **Important Note:** This `README.md` is an operational overview of the Stage 1 experiment. It is intentionally shorter and more readable than the full specification. For exact implementation details, schemas, thresholds, and configuration parameters, please refer to **`stage1-spec.md`**. If there is any discrepancy, `stage1-spec.md` overrides this file.

## 🎯 Objective

Stage 1 is a single-model, single-dataset minimum viable pipeline.

* **Model:** `meta-llama/Llama-3.1-8B-Instruct`
* **Dataset:** GSM8K test split
* **Goal:** jointly measure the behavioural optimum (L^*) and the mechanistic horizon (k^*) on the same pool of explicit CoT traces, then compare them in a controlled and interpretable way.

The purpose of Stage 1 is not to establish a final general theory of CoT faithfulness. Its role is to validate the full technical route end-to-end and produce a first clean comparison between:

* where accuracy peaks as CoT gets longer;
* where causal usefulness peaks within the reasoning chain;
* whether those two turning points align or diverge.

---

## 🧠 Research Question & Hypotheses

**Research Question:**
When the accuracy-optimal CoT length and the NLDD reasoning horizon are measured on the same model, dataset, and trace pool, what numerical relationship do they exhibit?

### H1: (L^* \approx k^*) — Alignment

The behavioural optimum and the mechanistic horizon coincide. Once the trace passes the optimal length, later reasoning steps no longer improve accuracy and no longer contribute causally.

### H2: (L^* < k^*) — Behavioural optimum earlier than causal horizon

Accuracy has already started to decline, but later steps still have positive causal contribution under NLDD. This suggests the model is still relying on later reasoning, but that extra reasoning is behaviourally harmful.

### H3: (L^* > k^*) — Causal horizon earlier than behavioural optimum

The strongest causal contribution occurs before the behavioural optimum. This suggests later steps may still help through mechanisms not well captured by single-step NLDD corruption, such as local repair, output stabilisation, or answer-format anchoring.

---

## 📌 What This Project Measures

This project compares two quantities from two different perspectives.

### 1. Behavioural optimum: (L^*)

(L^*) is the CoT length at which accuracy peaks.

In Stage 1, this is **not treated as one global scalar for the whole dataset**. Instead:

1. we first estimate per-question difficulty from post-hoc accuracy;
2. we split questions into difficulty bands;
3. we compute a separate accuracy-vs-length curve for each band;
4. we define a separate (L^*) for each difficulty band.

So throughout this README, (L^*) means:

[
L^*_d = \arg\max_L ; \text{accuracy}_d(L)
]

where (d) is a difficulty band.

### 2. Mechanistic horizon: (k^*)

(k^*) is the step position where the reasoning trace shows its strongest causal contribution under NLDD.

For one clean correct trace, we:

1. corrupt one reasoning step at position (k);
2. truncate all later steps;
3. recompute the correct-answer logit margin;
4. measure how much that margin drops.

This produces a per-trace NLDD profile over step positions. We then define the trace-level peak:

[
k_i^* = \arg\max_{k>1} \text{NLDD}_i(k)
]

Stage 1 uses this **per-trace peak** as the primary building block. It does **not** define the main horizon directly from a globally averaged normalised curve.

### 3. Why the main comparison is local, not global

The primary comparison is not between a global (L^*) and a global (k^*). Instead, for each difficulty band we compare:

* the behavioural optimum (L^*_d), and
* the aggregated trace-level horizon among traces whose lengths fall in a small **near-(L^*)** window.

This yields the main comparison quantity:

[
\Delta_d = k^*_{\text{near-}L^*,d} - L^*_d
]

This local comparison is the core of the project, because it asks whether the strongest causal step tends to occur at the same scale as the empirically best-performing trace length.

---

## 🔍 Shared Trace Definition

The whole project uses one unified notion of “step” and “trace length”.

* A **step** is one non-empty line after splitting the model completion by newline.
* The final answer line is recorded separately and does **not** count as a reasoning step.
* Trace length is therefore defined as `actual_num_steps`.

This same definition is used for both:

* behavioural length bucketing in the accuracy analysis;
* mechanistic corruption indexing in NLDD.

This matters because the project compares the two frameworks on the **same explicit step structure**, rather than using one notion of length for behaviour and another for mechanism.

---

## 🧪 Stage 1 Pipeline Overview

Stage 1 runs as a five-stage pipeline.

### 1. Pilot Run

A low-cost pilot validates the full pipeline before the main run.

It checks whether:

* the ICL prompt groups actually induce a broad enough step-length distribution;
* step parsing and answer extraction are reliable;
* the corruption procedure is feasible on real traces;
* NLDD values are numerically sensible on a small smoke-test subset.

### 2. Data Phase — Trace Generation

For each GSM8K question, the model generates multiple explicit CoT traces.

Length variation is induced **naturally**, not by hard length commands.

The project uses:

* multiple ICL prompt groups with different reasoning styles;
* per-group temperature settings;
* post-hoc grouping by the observed `actual_num_steps`.

Each trace is then parsed into:

* reasoning steps;
* final answer line;
* extracted numeric answer;
* correctness label;
* metadata such as token count and prompt group.

After all traces are generated, question difficulty is computed post hoc from the empirical correct rate across samples.

### 3. Coarse Analysis Stage

Before any full NLDD sweep, Stage 1 first freezes the key analysis boundaries.

This stage:

* builds difficulty bands from post-hoc question accuracy;
* computes per-difficulty `accuracy(L)` curves;
* identifies per-difficulty behavioural optima (L^*_d);
* creates per-difficulty relative length bins (short / mid / long);
* defines the near-(L^*) comparison window for each difficulty band.

This is important because the later NLDD analysis is not run over the entire trace pool indiscriminately. It is run inside a structure that has already fixed:

* which difficulty regime a trace belongs to;
* whether it is short, medium, or long relative to that regime;
* which traces count as “near the behavioural optimum”.

### 4. NLDD Measurement Stage

NLDD is run only on **correct clean traces**.

For each selected trace:

1. compute the clean logit difference;
2. corrupt each reasoning step in turn;
3. truncate all following steps;
4. recompute the corrupted logit difference;
5. convert the drop into an NLDD value.

The corruption system uses a two-tier fallback:

* **Tier 1:** numeric perturbation, preferably on the computed result;
* **Tier 2:** arithmetic operator swap when numeric perturbation is not possible.

For each selected trace, this yields:

* a full NLDD profile over step positions;
* a trace-level peak (k_i^*);
* an optional normalised ratio (r_i^* = k_i^*/L_i) used only as a secondary view.

### 5. Aggregation & Visualisation

The final stage aggregates all per-trace results.

It produces:

* per-difficulty behavioural optima (L^*_d);
* per-cell distributions of trace-level (k_i^*);
* near-(L^*) horizon summaries;
* the final (L^*) vs. (k^*) comparison for each difficulty band.

The main Stage 1 result is therefore not a single scalar, but a structured comparison across difficulty regimes.

---

## 📈 Main Outputs

Stage 1 produces six core figures.

1. **Accuracy vs. CoT Length (per difficulty)**
   Shows the behavioural inverted-U pattern and marks (L^*_d).

2. **NLDD Surface Heatmap**
   Shows mean NLDD over absolute `(L, k)` positions for each difficulty band.

3. **Per-trace (k^*) Distribution**
   Shows how trace-level horizons vary across difficulty × relative-length cells.

4. **Near-(L^*) Comparison Plot**
   Directly compares (L^*_d) with (k^*_{\text{near-}L^*,d}).

5. **Normalised NLDD Curve (secondary view)**
   Aggregates NLDD over relative position `k/L`; this is auxiliary and not the main horizon definition.

6. **Difficulty vs. Optimal Length Scatter**
   Shows how post-hoc task difficulty relates to the behavioural optimum.

---

## ✅ Scope of Stage 1

Stage 1 is intentionally narrow.

It includes:

* one model;
* one dataset;
* explicit CoT traces only;
* single-step corruption only;
* behavioural length analysis plus NLDD-based causal analysis.

It does **not** aim to provide:

* a multi-model comparison;
* a multi-dataset benchmark;
* RSA / probing / TAS analysis as part of the core Stage 1 result;
* a final explanation of why late CoT tokens help or hurt.

Those belong to later stages once the joint measurement pipeline is stable.

---

## 🚀 Why Stage 1 Matters

The two target papers motivate complementary but distinct questions.

* The optimal-length view asks: **how long should a reasoning trace be for best performance?**
* The NLDD view asks: **which reasoning steps are still causally doing work?**

Stage 1 brings these two views into one shared experimental frame.

This is the core conceptual contribution of the pipeline: instead of studying behavioural optimality and mechanistic faithfulness separately, it measures both on the same traces and asks whether they point to the same turning point.

If they align, that supports a simple story: behavioural overthinking arises because causal reasoning usefulness has already decayed.

If they diverge, that is equally informative: it suggests later CoT tokens may still serve functions other than direct stepwise causal reasoning, such as local repair, answer anchoring, or post-hoc structuring.

---

## 🔭 Stage 2 Extensions

Once Stage 1 is stable, Stage 2 can expand along three dimensions.

### Model dimension

Add stronger or different instruction-tuned models, such as:

* `Qwen2.5-7B-Instruct`
* larger reasoning-oriented models

### Dataset dimension

Extend beyond GSM8K to harder benchmarks such as:

* MATH

This will require revised parsing and answer-extraction rules.

### Analysis dimension

Add richer explanations for alignment or mismatch, including:

* step complexity proxies;
* compression-theoretic views;
* additional corruption schemes;
* stronger mechanism-oriented follow-up analysis.

---

## 🧭 One-Sentence Summary

Stage 1 asks a precise question on a tightly controlled setup:

**when explicit CoT traces get longer, does the length that maximises accuracy coincide with the step position where causal reasoning usefulness peaks?**
