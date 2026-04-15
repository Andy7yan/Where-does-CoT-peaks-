# Stage 1 Experiment Specification (v3)

> **Changelog (v2 → v3)**: This version is revised based on code audit results, formalizing implemented but undocumented design decisions into the spec. Major changes include: changing corruption to a two-tier fallback (§3.7), per-group temperature gradients (§5.1), shard directory layout (§7.2), recording implementation details (§5.6), and aligning the configuration parameter table with the actual code (§8.1).

---

## §1 Research Question and Hypotheses

### 1.1 Research Question

When measuring the accuracy-optimal CoT length ($L^*$) and the NLDD reasoning horizon ($k^*$) simultaneously on the same model and dataset, what is the numerical relationship between the two?

Background: The Optimal-length paper reports from a behavioral perspective that accuracy has an inverted-U relationship with CoT length, indicating the existence of a length $L^*$ that maximizes accuracy. The NLDD paper measures causal contribution via step-level counterfactuals from a mechanistic perspective, reporting the existence of a reasoning horizon $k^*$, beyond which a step's causal effect on the final answer approaches zero. The two papers were published independently and did not discuss each other. This experiment measures both simultaneously on the same dataset.

### 1.2 Three Mutually Exclusive Hypotheses

**H1: $L^* \approx k^*$ (Consistent)** The behavioral inflection point coincides with the mechanistic horizon. Steps beyond $L^*$ neither improve accuracy nor have any causal effect. The inverted-U shape at the behavioral level is fully explained by the faithfulness decay at the mechanistic level.

**H2: $L^* < k^*$ (Behavioral inflection precedes mechanistic horizon)** Accuracy has already decreased, but NLDD shows that subsequent steps still have a causal contribution. Implication: The model faithfully executes extra reasoning, but this reasoning is harmful (overthinking / over-correction).

**H3: $L^* > k^*$ (Behavioral inflection follows mechanistic horizon)** NLDD shows that steps no longer have a causal contribution, but accuracy is still increasing. Implication: Later steps work through indirect mechanisms not measurable by NLDD (format anchoring, attention stabilization, etc.), or the single-step corruption of NLDD lacks sensitivity.

All three hypotheses have independent research value; whichever is supported constitutes a reportable finding.

### 1.3 Interpretation Caveat

NLDD measures the causal impact of the step's text content on the final answer logit; it does not measure whether the model is "actually thinking" during the generation of tokens for that step. High NLDD means "the model's subsequent predictions relied on this text," which is not directly equivalent to "the model performed computations in this step." This distinction is especially important when compared to studies like "Think Dot by Dot"—the presence of CoT tokens is not necessarily synchronous with actual internal computation.

### 1.4 Per-Group Temperature Caveat

The ICL groups in this experiment use different sampling temperatures (§5.1), which means traces in different length bins originate from different temperature settings. This does not affect the measurement of $L^*$ and $k^*$: $L^*$ comes from the mean accuracy curve binned by `actual_num_steps`, where temperature only increases intra-bin variance rather than introducing systemic bias; the NLDD forward pass for $k^*$ is deterministic (greedy logit reading) and is unaffected by generation temperature. Temperature differences might at most affect the accuracy variance within a bin via the indirect channel of "low-temperature traces have higher quality at the same length," but it will not systematically shift the position of $L^*$. If subsequent analysis needs to control for this variable, a robustness check stratified by `prompt_id` can be performed.

---

## §2 Scope and Non-Goals

### 2.1 Experiment Positioning

Stage 1 = Minimum Viable Loop on a single model + single dataset. The goal is to verify whether the technical route of "measuring $L^*$ and $k^*$ simultaneously on the same dataset and comparing them" is feasible, producing interpretable preliminary conclusions.

### 2.2 Scope

- Model: meta-llama/Llama-3.1-8B-Instruct
- Dataset: GSM8K-Platinum test split (full 1209 questions)
- Precision: float16
- Deliverables: accuracy(L) curve and $L^*$ estimation, NLDD profile and $k^*$ estimation, direct comparison of $L^*$ vs $k^*$, 7 figures

### 2.3 Non-Goals

- No multi-model comparisons. Qwen2.5-7B-Instruct or stronger models will be introduced in Stage 2.
- No datasets other than GSM8K-Platinum. MATH (all levels) is deferred to Stage 2.
- No NLDD analysis on incorrect traces. Incorrect traces only contribute to accuracy statistics.
- No multi-step corruption. Only one step is altered at a time.
- No RSA / probing or other interpretability analyses.
- No Compression Theory Level B (H1/H2/H3 mechanistic explanation) or Level C ($\Phi$ fitting). These will unfold on Stage 2 data.
- No hard-control length instructions (e.g., "Use exactly N steps"). Length variation is achieved entirely through ICL exemplar style differences and per-group temperature gradients.
- No perplexity filtering (OQ-3 decided to suspend implementation for now).

### 2.4 Stage 2 Extension Directions

- Model dimension: Retain Llama-3.1-8B, add Qwen2.5-7B-Instruct or stronger.
- Dataset dimension: MATH (all levels), requires a new step parser (period splitting) and answer extraction (`\boxed{}` format).
- Analysis dimension: Compression Theory Levels B/C, preset stratification of question difficulty, multi-step corruption.
- Analysis dimension: Step Complexity Proxy (Compression Theory Level A): Rule + LLM judge hybrid annotation of structural features for each step (number of prior results referenced, operation types, whether new intermediates are introduced).

---

## §3 Unified Definitions

The following definitions remain consistent throughout Stage 1 and do not change depending on the measurement type (accuracy / NLDD).

### 3.1 Step Definition

A step = A non-empty text segment in the CoT text separated by a newline character.

Splitting rules, executed in order:

1. Split the raw completion text by `\n`.
2. Strip leading/trailing whitespaces for each segment.
3. Discard empty strings.
4. Discard purely punctuation or purely whitespace segments.
5. The final answer line (the line containing `####` or `The answer is`) is not counted as a step and is recorded separately as `final_answer_line`.

`actual_num_steps` = The length of the step list produced by the above process.

This definition is used simultaneously for: horizontal axis binning for accuracy curves, step indices for NLDD corruption, and statistics for natural length distributions.

### 3.2 Answer Extraction

The gold answer for GSM8K-Platinum is a numerical value. Extraction rules by priority:

1. Look for content after `####` in the completion, taking the first matched numerical value.
2. If there is no `####`, look for the numerical value after `The answer is`.
3. If still unmatched, mark `extraction_failed = True`.

Value normalization: Remove commas, dollar signs, percent signs, and cast to float. Equality condition: `abs(extracted - gold) < 1e-3`.

### 3.3 Global Calibration Constant S

To achieve cross-model comparability, NLDD uses a global normalization constant $S$, calibrated on clean reasoning traces:

$$S = \frac{1}{M} \sum_{m=1}^{M} \sigma(z_m)$$

Where $z_m$ is the final-token logit vector of the $m$-th clean trace, and $\sigma(\cdot)$ calculates the standard deviation of this vector across the entire vocabulary. $M$ = Total number of clean traces participating in calibration.

$S$ reflects the model's inherent output variability, not absolute logit magnitudes. In Stage 1, $S$ is calculated once on the full set of clean traces, and all NLDD measurements share this same $S$.

### 3.4 Logit Difference (LD)

For a given prompt, confidence is defined as the normalized margin:

$$LD = \frac{\max_{y \in Y_{correct}} \ell(y) - \max_{y' \in Y \setminus Y_{correct}} \ell(y')}{S}$$

Where $Y_{correct}$ contains all valid token IDs for the correct answer (considering tokenization variants like leading spaces). For GSM8K-Platinum multi-token answers, the first-token margin is used as a stable proxy.

This definition remains consistent under both clean and corrupt conditions.

### 3.5 NLDD Calculation

For a clean trace of length $L$, corruption position $k \in \{1, \ldots, L\}$:

$$NLDD(k) = \frac{LD_{clean} - LD_{corrupt,k}}{|LD_{clean}|} \times 100$$

- `LD_clean`: The LD value after feeding the full clean trace into the model.
- `LD_corrupt_k`: The LD value after replacing the $k$-th step with its corrupt version and truncating the $k+1$-th step and all subsequent content.
- Exclusion condition: Samples with `|LD_clean| < ε` (`ε = 1e-6`) do not participate in NLDD calculations to avoid noise amplification from near-zero baseline confidence.

`NLDD > 0` indicates the step has a causal effect (corruption lowered answer confidence). `NLDD ≈ 0` indicates weak coupling. `NLDD < 0` indicates confidence reversal (corruption actually increased the margin).

### 3.6 k* Definition

$k^*(L)$ = The step index where the NLDD value reaches its peak in the NLDD profile of a trace of length $L$.

The peak-based definition is used by default. If switching to steepest-decline or threshold-based definitions is necessary, it should be declared as a configuration parameter without changing the code logic.

### 3.7 Corruption Methods

Stage 1 uses a two-tier corruption fallback: Tier 1 arithmetic perturbation and Tier 2 operator swap.

#### Tier 1: Arithmetic Perturbation

1. Use RegEx to locate all numerical expressions in the target step.
2. **Prioritize numerical values on the right side of the equals sign** (i.e., calculation results), then operands. Select randomly within the same priority.
3. Determine if the value is an integer (Integer check: `value.is_integer()` and the original text contains no decimal point; forms like `"10.0"` are treated as non-integers):
   - Integer: Replace with `original + delta`, where `delta ∈ {-2, -1, +1, +2}`, chosen randomly. If `original = 0`, `delta ∈ {+1, +2}`.
   - Non-integer: Sample a multiplier with 50% probability each from the intervals `[FLOAT_RANGE_LOW_MIN, FLOAT_RANGE_LOW_MAX]` and `[FLOAT_RANGE_HIGH_MIN, FLOAT_RANGE_HIGH_MAX]`, replacing with `original × multiplier`. Default intervals are `[0.5, 0.9]` and `[1.1, 1.5]`.
4. If the step has no replaceable numerical values, proceed to Tier 2.
5. Quality Filtering: The token count difference between the corrupt step and the clean step must be $\leq$ `CORRUPTION_TOKEN_DELTA_MAX`. Resample if this is not met. Each candidate value can be retried up to `CORRUPTION_RETRY_LIMIT` times; if all candidates fail, proceed to Tier 2.

#### Tier 2: Operator Swap

If Tier 1 fails, attempt to swap arithmetic operators in the step (e.g., `+` ↔ `-`, `×`/`*` ↔ `÷`/`/`), subject to the same token count delta constraints.

#### Tier 3: Semantic Flip (Disabled by default)

Antonym replacement logic (e.g., `"more"` ↔ `"less"`) is preserved in the code but disabled by default. It can be enabled via the `ENABLE_TIER3_SEMANTIC_FLIP` config. Stage 1 does not use Tier 3.

#### Failure Handling

If both Tier 1 and Tier 2 fail, mark `corruption_failed = True` and skip this position.

#### Output Records

Each corruption record includes `corruption_tier` (1 or 2) and `corruption_type` (`numeric_result` / `numeric_operand` / `operator_swap` / `uncorruptible`) to stratify by tier during downstream analysis.

Exclusion condition: Geometry questions are not corrupted (GSM8K-Platinum contains almost no geometry; this rule is reserved for Stage 2 MATH).

### 3.8 Difficulty Definition

Post-hoc definition, no preset difficulty stratification:

$$difficulty(q) = 1 - accuracy(q)$$

Where $accuracy(q)$ = the correct rate for question $q$ across all samples.

Difficulty values are calculated after all traces in the Data Phase are generated and appended to the metadata for each question.

---

## §4 Pilot Run

The Pilot Run is executed before the formal Data Phase and Analysis Phase. The goal is to verify the feasibility of the end-to-end pipeline at low cost, expose deviations between design assumptions and reality, and provide a basis for parameter selection in the formal run.

### 4.1 Scope

Randomly sample 50–100 questions from the GSM8K-Platinum test split (can overlap with formal run questions; Pilot data is not included in formal analysis). Use the same ICL exemplar groups and sampling parameters as the formal run. The number of samples per question can be reduced to save costs.

The Pilot uses the same config file as the formal run, overriding generation parameters (like `pilot.samples_per_group`) via the `pilot.*` override block. No separate config file is needed.

### 4.2 Must-Verify Items

#### A. Length Guidance Effectiveness

Check if the ICL exemplars actually produced a distributional difference in `actual_num_steps`. Specific checks:

- Is there a discernible gradient in the median `actual_num_steps` across groups?
- Is the total range of combined `actual_num_steps` wide enough to support `accuracy(L)` curve fitting?
- If distributions highly overlap, ICL guidance is ineffective, and exemplars must be redesigned or supplementary methods considered.

#### B. Sample Size per Length Bin

After binning by `actual_num_steps`, check the number of total traces and correct traces per bin:

- If total traces in a bin are too low, the accuracy estimation for that bin is unreliable; the formal run may need to increase samples per group.
- If a bin has 0 correct traces, NLDD analysis cannot be performed for that bin; the formal run should narrow NLDD coverage in that length range.

#### C. Step Segmentation and Answer Extraction

- Is the `extraction_failed` ratio below the threshold?
- Is the ratio of traces with `actual_num_steps = 0` negligible after step segmentation?
- Spot-check step segmentation results to manually verify if boundaries are reasonable.

#### D. Corruption Feasibility

Run the corruption pipeline on correct traces from the Pilot data:

- Is the `corruption_failed` ratio below the threshold?
- Spot-check corrupt steps to ensure the replaced text remains grammatically coherent and the token count delta $\leq$ 2.
- Tabulate distribution by `corruption_tier` to confirm Tier 1 coverage.

#### E. NLDD End-to-End Smoke Test

Select a small number of correct traces and run the full NLDD pipeline (including S calibration, LD calculation, NLDD calculation) to confirm:

- $S > 0$ and is of a reasonable magnitude.
- `LD_clean` is mostly > 0.
- NLDD profile shapes are interpretable, not entirely zero or pure noise.

### 4.3 Decision Rules

Pilot results directly influence the formal run's parameters.

| Pilot Finding | Decision |
|---|---|
| ICL guidance effective | Formal run maintains current ICL groups and sample settings |
| ICL guidance weak | Increase samples per group, or add new exemplar groups |
| ICL guidance ineffective | Stop and redesign exemplars; do not enter formal run |
| Some bins have 0 correct traces | Formal run skips NLDD for these bins |
| `extraction_failed` too high | Check answer format instructions, fix, and rerun Pilot |
| `corruption_failed` too high | Relax corruption rules, or mark step type to be skipped |

### 4.4 Pilot Artifacts

- `pilot_traces.jsonl`: Schema identical to formal traces.
- `pilot_report.md`: Documents results for each check item and decisions made.

Pilot data is not included in the formal analysis but is retained for debugging.

---

## §5 Data Phase

The goal of the Data Phase is to generate length-varied reasoning traces and collect enough correct traces for use in the Analysis Phase.

### 5.1 Sampling Strategy

- Use `NUM_ICL_GROUPS` groups of ICL exemplars of varying complexity. Sample `SAMPLES_PER_GROUP` times per group. Total `NUM_ICL_GROUPS × SAMPLES_PER_GROUP` samples per question.
- Do not use hard-control length instructions like "Use exactly N steps". Length variation is achieved via two methods: (a) stylistic differences in ICL exemplars (brief vs detailed solutions); (b) per-group temperature gradients (low temps lean toward short chains, high temps lean toward long chains).
- Each ICL exemplar group is bound to an independent temperature, configured via `ICL_GROUP_TEMPERATURES`.
- All downstream analysis is binned by `actual_num_steps`, not by ICL group. ICL group and temperature are merely means to generate length variance, not dimensions of analysis.

### 5.2 ICL Exemplar Requirements

Each group of exemplars must meet the following:

- The solution content is correct (matches gold answer).
- Solution styles have discernible differences in step counts (verified by Pilot Run).
- The final answer is provided using the `####` format, consistent with §3.2 extraction rules.
- Exemplar questions do not overlap with the GSM8K-Platinum test split.

Specific content and quantity of exemplars are defined in the `prompts/` directory; the spec does not dictate the text.

### 5.3 Data Structures

#### Run-level Metadata

One JSON file per run (`run_meta.json`):

| Field | Type | Description |
|---|---|---|
| `run_id` | str | Unique identifier for the run |
| `model_name` | str | Model identifier |
| `dataset` | str | Dataset name and split |
| `temperature` | float \| null | Global temp (null if per-group gradients are used) |
| `icl_group_temperatures` | dict[str, float] | Actual temperatures per ICL group |
| `max_new_tokens` | int | Generation cap |
| `num_icl_groups` | int | Number of ICL groups |
| `samples_per_group` | int | Samples per group |
| `seed` | int | Global random seed |
| `prompt_ids` | list[str] | List of ICL group version IDs |
| `schema_version` | str | Trace schema version identifier |
| `timestamp` | str | Run start time |

#### Trace Table

JSONL, one trace per line, containing only per-sample fields:

| Field | Type | Description |
|---|---|---|
| `trace_id` | str | Unique identifier |
| `question_id` | str | Unique question ID |
| `question_text` | str | Raw question text |
| `gold_answer` | float | Gold standard answer |
| `prompt_id` | str | ICL group version ID |
| `raw_completion` | str | Raw model output |
| `steps` | list[str] | Step list split according to §3.1 |
| `actual_num_steps` | int | `len(steps)` |
| `final_answer_line` | str | Raw text of answer line |
| `extracted_answer` | float \| null | Numeric answer extracted per §3.2 |
| `is_correct` | bool | Match success flag |
| `extraction_failed` | bool | Extraction failure flag |
| `token_count` | int | Completion token count |
| `timestamp` | str | Generation time |

### 5.4 Per-Question Metadata

After all traces in the Data Phase are generated, calculate for each question:

| Field | Type | Description |
|---|---|---|
| `question_id` | str | Unique question ID |
| `difficulty` | float | `1 − accuracy(q)`, see §3.8 |
| `accuracy` | float | Correct rate across all samples for this question |
| `total_samples` | int | Total samples for this question |
| `correct_count` | int | Number of correct traces |
| `optimal_length` | int \| null | `actual_num_steps` corresponding to peak accuracy (min step count if tied) |
| `natural_length_distribution` | dict | `{actual_num_steps: count}` distribution |

### 5.5 Data Phase Completion Criteria

- Samples per question = `NUM_ICL_GROUPS × SAMPLES_PER_GROUP`, no omissions.
- Global `extraction_failed` ratio < `MAX_EXTRACTION_FAIL_RATE` (print warning if exceeded, do not hard fail).
- The global distribution of `actual_num_steps` covers the effective range confirmed by the Pilot Run.
- Per-question metadata is calculated and persisted.
- The `accuracy(L)` curve can be independently calculated right after the Data Phase ends, without waiting for the Analysis Phase.

### 5.6 Implementation Details (Record)

The following implementation details are not part of the experimental design but affect reproducibility, hence recorded here:

- **Prompt Injection Method**: ICL exemplars are injected in chat format—system message + each few-shot converted to (user question, assistant answer) message pairs + final user question, passed via `apply_chat_template`.
- **Tokenizer Config**: If the model lacks a `pad_token`, use `eos_token` as `pad_token`; `padding_side = "left"`. This is standard for Llama-3.1-Instruct, may need adjustment for other models.
- **Batched generation**: Multiple samples of the same prompt are batched to call `model.generate`, batch size configured via `GENERATION_BATCH_SIZE` (default 4).
- **Shard Parallelism**: Stage B can run in parallel by slicing via `--start-idx` / `--end-idx`, outputting to shard subdirectories (`<run_dir>/shards/<shard_id>/traces.jsonl`). Stage E is responsible for merging shards into a root-level `traces.jsonl`.

---

## §6 Analysis Phase

The Analysis Phase consumes the outputs of the Data Phase (traces + per-question metadata) and produces $L^*$ estimates, NLDD profiles, TAS profiles, $k^*$ estimates, and visualization results.

### 6.1 Stage 1: accuracy(L) → L*

From the full set of traces, bin by `actual_num_steps`, and calculate the accuracy mean and standard error for each bin.

Binning rules:

- Bin width = 1 step (each distinct `actual_num_steps` value is a bin).
- If a bin's sample size < `MIN_BIN_SIZE`, merge it with an adjacent bin (prioritizing the neighbor with the larger sample size). The merged bin's label is the median step count it contains.
- Merging is only used for display and $L^*$ location; original per-trace data is unaffected.
- $L^*$ = the bin label corresponding to the peak accuracy. If there are tied peaks, take the smallest step count.

This stage only requires Data Phase outputs and does not require model forward passes.

### 6.2 Stage 2: NLDD and TAS Measurement

#### 6.2.1 Question Selection

Question selection is done manually. The following strategy is for reference only and is not a hard rule:

- Sort by `difficulty` (§3.8) and sample equidistantly to ensure evenly distributed difficulty coverage.
- Prioritize questions with abundant correct traces that cover multiple `actual_num_steps` values.
- Avoid questions where correct traces are concentrated at a single length.

Selected questions are divided into two groups:

- **Full Analysis Group**: Full, step-by-step NLDD and TAS measurements on *all* of their correct traces.
- **Spot Check Analysis Group**: Perform NLDD measurement at `NUM_SPOT_CHECKS` random positions for each correct trace; TAS does not sample the corrupt side, retaining only the clean side arrays.

#### 6.2.2 S Calibration

Before any NLDD/TAS measurements, first calculate the global normalization constant $S$ (§3.3) on the full set of correct traces for this run. $S$ is calculated once and shared across all subsequent LD calculations.

#### 6.2.3 Clean Trace Selection

For each `actual_num_steps` value of every selected question:

- Filter traces where `is_correct = True`.
- If there are no correct traces, record `no_clean_trace` and skip.
- If there are multiple, select the one whose `token_count` is closest to the median of that length bin.

#### 6.2.4 NLDD Measurement Flow

For the selected clean trace (length `L = actual_num_steps`):

1. Calculate `LD_clean` (§3.4).
2. If `|LD_clean| < ε`, exclude.
3. For each corruption position `k` to be measured: Generate corrupt step (§3.7) → Concatenate truncated prefix → Calculate `LD_corrupt_k` → Calculate `NLDD(k)`.
   - Full Analysis Group: `k` iterates through `{1, ..., L}`.
   - Spot Check Analysis Group: `k` is uniformly sampled `NUM_SPOT_CHECKS` times from `{1, ..., L}`.

**Shared Forward Pass Principle:** Every forward pass simultaneously captures intermediate hidden states via forward hooks and reads final-token logits. A single pass yields raw data needed for both LD and TAS without doing it twice.

**NLDD Prompt Format:** Clean/corrupt prefixes concatenate steps using the format `"Step {i}: {text}"`, ending with `"#### "` as the answer suffix. This format influences how the model interprets the prefix; it requires review if changing models or datasets.

#### 6.2.5 TAS Measurement Flow

Trajectory extraction: For each trace (clean or corrupt), extract the hidden state at the final token of every reasoning step from the middle transformer layer (`⌊num_layers/2⌋`), yielding sequence `{h_0, h_1, ..., h_L}`, where `h_0` corresponds to the final token of the question, and `h_k` corresponds to the final token of the $k$-th reasoning step.

Record two quantities step-by-step:

- $cumulative\_displacement(k) = \|h_k - h_0\|$
- $step\_length(k) = \|h_k - h_{k-1}\|$

Derived from these two arrays:

- $running\_TAS(k) = \frac{cumulative\_displacement(k)}{\sum_{i=1}^{k} step\_length(i)}$
- $displacement\_increment(k) = cumulative\_displacement(k) - cumulative\_displacement(k-1)$
- $straightness(k) = \frac{displacement\_increment(k)}{step\_length(k)}$

Where $straightness(k) \in [0, 1]$. Approaching 1 means the new step's displacement contributes almost entirely to the total displacement; approaching 0 means the new step is detouring.

TAS inflection point definition: $k_{TAS}$ = The smallest $k$ such that for all $j \geq k$, the following holds:

$$|straightness(j) - straightness(k)| < TAS\_PLATEAU\_THRESHOLD$$

Meaning the starting position where the straightness curve enters a stable plateau.

The final TAS scalar is retained as:

$$TAS = running\_TAS(L)$$

#### 6.2.6 Data Recording Schema

##### Full Analysis Records

JSONL (`nldd_full.jsonl`), one corruption measurement point per line:

| Field | Type | Description |
|---|---|---|
| `nldd_id` | str | Unique identifier |
| `question_id` | str | Question ID |
| `clean_trace_id` | str | References trace table |
| `actual_clean_length` | int | `actual_num_steps` of the clean trace |
| `corruption_step_index` | int | `k` (1-indexed) |
| `corruption_tier` | int | Corruption tier used (1 or 2) |
| `corruption_type` | str | `numeric_result` / `numeric_operand` / `operator_swap` / `uncorruptible` |
| `corruption_failed` | bool | Failure flag |
| `ld_clean` | float | Clean LD |
| `ld_corrupt` | float | Corrupt LD |
| `nldd_value` | float | NLDD value |
| `clean_cumulative_disp` | list[float] | Step-wise `cumulative_displacement` for clean trace, length L |
| `clean_step_lengths` | list[float] | Step-wise `step_length` for clean trace, length L |
| `corrupt_cumulative_disp` | list[float] | Step-wise `cumulative_displacement` for corrupt trace, length k |
| `corrupt_step_lengths` | list[float] | Step-wise `step_length` for corrupt trace, length k |
| `timestamp` | str | Measurement time |

> Note: Clean-side arrays for the same clean trace are duplicated across different `k` rows. They can be stored separately and referenced via `clean_trace_id` in implementation; the spec only requires the data to be traceable.

##### Spot Check Analysis Records

JSONL (`nldd_spot.jsonl`), independent file:

| Field | Type | Description |
|---|---|---|
| `spot_id` | str | Unique identifier |
| `question_id` | str | Question ID |
| `clean_trace_id` | str | References trace table |
| `actual_clean_length` | int | `actual_num_steps` |
| `corruption_step_index` | int | `k` |
| `corruption_tier` | int | Corruption tier used (1 or 2) |
| `corruption_type` | str | Same as full analysis |
| `corruption_failed` | bool | Failure flag |
| `ld_clean` | float | Clean LD |
| `ld_corrupt` | float | Corrupt LD |
| `nldd_value` | float | NLDD value |
| `clean_cumulative_disp` | list[float] | Step-wise `cumulative_displacement` for clean trace, length L |
| `clean_step_lengths` | list[float] | Step-wise `step_length` for clean trace, length L |
| `timestamp` | str | Measurement time |

#### 6.2.7 k* Extraction

For each question in the full analysis group, group by `actual_clean_length`, and locate $k^*$ (§3.6, peak-based, excluding `k = 1`) in the NLDD profile for each group.

### 6.3 Visualizations

Total of 7 figures. The input data for each figure is saved as an independent CSV snapshot.

#### Figure 1: Accuracy vs CoT Length

- X-axis: `actual_num_steps`
- Y-axis: accuracy (mean ± SE)
- Annotations: $L^*$ position (vertical line + label)
- Data source: Full traces

#### Figure 2: NLDD Surface Heatmap

- X-axis: corruption position `k`
- Y-axis: `actual_clean_length = L`
- Color: mean NLDD
- Overlay: $k^*(L)$ line plot
- Data source: Full analysis records

#### Figure 3: k*(L) vs L

- X-axis: `L` (`actual_clean_length`)
- Y-axis: $k^*$
- Overlay: $L^*$ vertical line, `y = x` diagonal reference line
- Data source: Aggregated full analysis records

#### Figure 4: Mean NLDD vs Relative Position

- X-axis: `k / L`
- Y-axis: mean NLDD
- Color separation: One line per distinct `L` value
- Data source: Full analysis records

#### Figure 5: TAS vs Corruption Position

- X-axis: corruption position `k`
- Y-axis: TAS
- Facet / Color: Grouped by `actual_clean_length`
- Data source: Full analysis records

#### Figure 6: Clean TAS vs CoT Length

- X-axis: `actual_num_steps`
- Y-axis: `TAS_clean` (mean ± SE)
- Annotations: $L^*$ position (vertical line + label)
- Data source: Clean traces from both full and spot check analysis groups

#### Figure 7: TAS Inflection Distribution

- X-axis: `k_TAS / L` relative position
- Y-axis: Frequency histogram
- Overlay: $L^* / L$ and $k^* / L$ vertical lines
- Data source: Full analysis group

---

## §7 Data Flow and Dependencies

This chapter defines the execution sequence, input/output dependencies across stages, and inter-stage handover criteria. It does not dictate specific implementation details or execution environments.

### 7.1 Stage Topology

| Stage | Name | Input | Output | Prerequisites |
|---|---|---|---|---|
| A | Eval Subset Prep | GSM8K-Platinum test split | `eval_subset.jsonl`, `eval_subset_meta.json` | None |
| B | Trace Generation | eval subset, ICL exemplar, model | shard directories (with `traces.jsonl`), `run_meta.json` | A |
| C | Accuracy Aggregation | `traces.jsonl` (merged) | `accuracy_by_length.csv`, $L^*$ estimate, per-question metadata | B |
| D | NLDD & TAS Measure | `traces.jsonl`, S, manual question selection | `nldd_full.jsonl`, `nldd_spot.jsonl` | B, C |
| E | Aggregation | `nldd_full.jsonl`, `nldd_spot.jsonl`, `accuracy_by_length.csv` | `nldd_surface.csv`, `nldd_by_relative_position.csv`, `horizon_summary.csv`, `tas_inflection_summary.csv` | C, D |
| F | Plotting | All aggregated tables | 7 figures + CSV snapshots for each | E |

### 7.2 Stage Details

#### Stage A: Eval Subset Preparation

Fix an evaluation subset from the GSM8K-Platinum test split. Fixing strategy: Sort `question_text` by `sha256("{hash_seed}:{question_text}")` and take the top `SUBSET_SIZE` questions. The hash seed is written to metadata to guarantee reproducibility. `question_id` is generated by the loop index after sorting (`gsm8k_platinum_{global_index:04d}`), not sourced from the raw data.

Artifacts:

- `eval_subset.jsonl`: Each line contains `question_id`, `question_text`, `gold_answer`
- `eval_subset_meta.json`: `SUBSET_SIZE` (field name `n`), hash seed, dataset split
- `gsm8k_platinum_test.jsonl`: Export of the full test split (auxiliary artifact, optional)

#### Stage B: Trace Generation

Execute §5 Data Phase on each question in the evaluation subset.

Stage B supports shard parallelism: Each shard writes to `<run_dir>/shards/<shard_id>/traces.jsonl` + `shard_meta.json`. Slicing is controlled via `--start-idx` / `--end-idx`.

Artifacts (per shard):

- `shards/<shard_id>/traces.jsonl`
- `shards/<shard_id>/shard_meta.json`

Upon startup, Stage E is responsible for merging shards into a root-level `traces.jsonl` + `run_meta.json`.

Stage B is the primary computational cost. Proceed to Stage C immediately after completion.

#### Stage C: Accuracy Aggregation

Bin `traces.jsonl` by `actual_num_steps` (§6.1), outputting:

- `accuracy_by_length.csv`
- $L^*$ estimate
- Per-question metadata (`question_metadata.jsonl`): contains `difficulty`, `accuracy`, `optimal_length`, `natural_length_distribution`

#### Stage D: NLDD & TAS Measurement

Stage D is divided into two sub-stages:

**D1: Corruption Preparation** (No GPU forward pass required)
- Read traces, filter correct traces grouped by full/spot check.
- Generate corrupt text for each required measurement step of each selected trace.
- Output `all_steps.jsonl` (full analysis group, all steps for each trace) and `sampled_steps.jsonl` (spot check group, `NUM_SPOT_CHECKS` random positions per trace).
- Output `corruption_summary.json` (success/failure counts per tier/type).

**D2: Forward Pass + Measurement** (Requires GPU)
- S Calibration: Calculate $S$ (§3.3) on the full set of correct traces, persist it.
- Execute a forward pass for each record produced in D1, calculating LD_clean, LD_corrupt, NLDD.
- Extract hidden states via forward hooks simultaneously to calculate TAS metrics.
- Output `nldd_full.jsonl`, `nldd_spot.jsonl` (schema in §6.2.6).

#### Stage E: Aggregation

Calculate from Stage C and Stage D outputs:

- `nldd_surface.csv`: Mean NLDD aggregated by `(L, k)` grid (full analysis group).
- `nldd_by_relative_position.csv`: Aggregated after `k / L` normalization.
- `horizon_summary.csv`: $k^*$, $L^*$, and accuracy corresponding to each $L$.
- `tas_inflection_summary.csv`: $k_{TAS}$, $k_{TAS} / L$, and corresponding `actual_clean_length` for each trace in the full analysis group.

Stage E is also responsible for merging traces from shards upon startup (if not already merged).

#### Stage F: Plotting

Generate the 7 figures from §6.3 using aggregated tables. Each figure is saved alongside its input CSV snapshot. Plotting runs locally and does not require a GPU.

### 7.3 Inter-Stage Handover Criteria

| Handover Point | Check Content |
|---|---|
| A → B | `eval_subset.jsonl` exists, line count = `SUBSET_SIZE` |
| B → C | Merged `traces.jsonl` exists, line count = `SUBSET_SIZE × NUM_ICL_GROUPS × SAMPLES_PER_GROUP` |
| B → D | Same as above, and S is calibrated and persisted |
| C → D | Per-question metadata is produced |
| C, D → E | `accuracy_by_length.csv`, `nldd_full.jsonl`, `nldd_spot.jsonl` all exist |
| E → F | All aggregated CSVs exist and are non-empty |

If a handover check fails, it should throw an error and terminate, rather than failing silently.

### 7.4 Pilot Run Positioning in Topology

The Pilot Run (§4) is executed after Stage A and before the formal Stage B. The Pilot uses an independent evaluation subset, and its artifacts are stored independently, not mixed with the formal run.

Pilot outputs determine the configuration parameters for the formal run. Formal Stage B starts only after parameters are confirmed.

---

## §8 Configurations and Validation Checklist

### 8.1 Configuration Parameter Summary

All experimental parameters are centralized in a single configuration file. Pilot and formal Runs share this config; Pilot overrides generation parameters via the `pilot.*` override block.

#### Run Identification

| Parameter | Description | Current Value |
|---|---|---|
| `run_id` | Unique identifier for this run | `"peak-cot-stage1-gsm8k-platinum-llama31"` |
| `seed` | Global random seed | 42 |

#### Dataset

| Parameter | Description | Current Value |
|---|---|---|
| `dataset.name` | Dataset identifier | `"madrylab/gsm8k-platinum"` |
| `dataset.split` | Split | `"test"` |
| `dataset.subset_size` | Evaluation subset size | 1319 (full test split) |
| `dataset.hash_seed` | Hash sorting seed | 42 |

#### Model

| Parameter | Description | Current Value |
|---|---|---|
| `model.name` | Model identifier | `"meta-llama/Llama-3.1-8B-Instruct"` |
| `model.dtype` | Inference precision | `"float16"` |
| `model.hf_cache` | HuggingFace cache path | Determined by environment |

#### Generation

| Parameter | Description | Current Value |
|---|---|---|
| `generation.num_icl_groups` | Number of ICL exemplar groups | 5 |
| `generation.samples_per_group` | Samples per group | 3 |
| `generation.max_new_tokens` | Token generation cap | 512 |
| `generation.batch_size` | Generation batch size | 4 |
| `generation.icl_groups` | Configuration per ICL group (incl. temp) | See table below |

##### ICL Group Configurations

| prompt_id | temperature | Design Intent |
|---|---|---|
| `icl_minimal` | 0.0 | Greedy, shortest chain |
| `icl_short` | 0.3 | Low temp, short chain |
| `icl_medium` | 0.5 | Medium |
| `icl_detailed` | 0.7 | High temp, long chain |
| `icl_verbose` | 0.7 | High temp, detailed chain |

#### Step Segmentation & Answer Extraction

| Parameter | Description | Current Value |
|---|---|---|
| `step_segmentation.method` | Segmentation method | `"newline"` |
| `answer_extraction.markers` | Answer line marker list | `["####", "The answer is"]` |
| `answer_extraction.numeric_tolerance` | Answer equality tolerance | `1e-3` |

#### NLDD

| Parameter | Description | Current Value |
|---|---|---|
| `nldd.corruption_type` | Corruption type | `"tiered_fallback"` |
| `nldd.enable_tier3_semantic_flip` | Enable Tier 3 flag | `false` |
| `nldd.integer_perturbation_deltas` | Candidate deltas for int corruption | `[-2, -1, 1, 2]` |
| `nldd.float_perturbation_range` | Multiplier bounds for non-int | `[0.5, 0.9, 1.1, 1.5]` |
| `nldd.corruption_token_delta_max` | Max token diff between corrupt and clean | 2 |
| `nldd.corruption_retry_limit` | Max retries per candidate value | 3 |
| `nldd.ld_epsilon` | `LD_clean` exclusion threshold | `1e-6` |
| `nldd.horizon_definition` | $k^*$ locating method | `"peak"` |

#### TAS

| Parameter | Description | Current Value |
|---|---|---|
| `tas.layer` | Hidden state extraction layer | `"middle"` (⌊num_layers/2⌋) |
| `tas.plateau_threshold` | `straightness` plateau threshold | 0.05 |

#### Analysis

| Parameter | Description | Current Value |
|---|---|---|
| `analysis.min_bin_size` | Min sample size for accuracy binning | 5 |
| `analysis.num_full_analysis_questions` | Question count for full analysis group | Manually decided |
| `analysis.num_spot_checks` | Random positions per trace for spot checks | 3 |
| `analysis.max_extraction_fail_rate` | Tolerance cap for `extraction_failed` | 0.05 |

#### Pilot Override

| Parameter | Description | Current Value |
|---|---|---|
| `pilot.num_icl_groups` | Pilot ICL group count | Same as `generation.num_icl_groups` |
| `pilot.samples_per_group` | Pilot samples per group | Can be reduced |
| `pilot.max_extraction_fail_rate` | Pilot extraction threshold | Same as `analysis...` |

### 8.2 Validation Checklist

#### Post Stage A

- `eval_subset.jsonl` line count = `SUBSET_SIZE`
- Each line contains `question_id`, `question_text`, `gold_answer`, with no null values.
- `eval_subset_meta.json` records `SUBSET_SIZE`, `hash_seed`, dataset split.

#### Post Stage B

- Merged `traces.jsonl` line count = `SUBSET_SIZE × NUM_ICL_GROUPS × SAMPLES_PER_GROUP`
- Each `(question_id, prompt_id)` combination has exactly `SAMPLES_PER_GROUP` records.
- Global `extraction_failed` ratio < `MAX_EXTRACTION_FAIL_RATE` (warning if exceeded).
- The global distribution of `actual_num_steps` covers the effective range confirmed by Pilot Run.
- Ratio of traces with `actual_num_steps = 0` is negligible.
- `run_meta.json` is consistent with current config.

#### Post Stage C

- `accuracy_by_length.csv` is non-empty, each line contains bin label, sample size, mean, SE.
- $L^*$ can be located (an accuracy peak exists).
- Per-question metadata is produced, question count = `SUBSET_SIZE`.
- Difficulty distribution is reasonable (not all 0 or 1).

#### Post Stage D

- $S > 0$ and is persisted.
- `nldd_full.jsonl` covers all selected questions in the full analysis group.
- `nldd_spot.jsonl` covers all selected questions in the spot check analysis group.
- In full analysis, for every `(question_id, actual_clean_length)` combo with a clean trace, `k` goes from 1 to L without omissions (except for `corruption_failed`).
- Global `corruption_failed` ratio < 15%.
- `nldd_value` distribution is reasonable (mostly > 0, slightly $\leq$ 0).
- `ld_clean` is mostly > 0.
- `clean_cumulative_disp` and `clean_step_lengths` lengths = `actual_clean_length` in full analysis records.
- `corrupt_cumulative_disp` and `corrupt_step_lengths` lengths = `corruption_step_index` in full analysis records.

#### Post Stage E

- In the `(L, k)` grid of `nldd_surface.csv`, missing spots are marked with reasons.
- $k^*$ values in `horizon_summary.csv` are within a reasonable range ($1 < k^* \leq L$).
- $k_{TAS}$ values in `tas_inflection_summary.csv` are within a reasonable range, $k_{TAS} / L \in (0, 1]$.
- All aggregated CSVs can be properly read by the plotting script.

#### Post Stage F

- All 7 figures render properly.
- Axis labels and annotations ($L^*$, $k^*$ lines/plots) are consistent with aggregated data.
- Input CSV snapshots for each figure are saved.

### 8.3 Open Questions

| ID | Question | Tag | Current Default |
|---|---|---|---|
| OQ-1 | Does $k^*$ definition need to switch from peak-based to steepest-decline? | assumption | Keep peak-based, evaluate after Stage 1 |
| OQ-2 | Reasonable value for `TAS_PLATEAU_THRESHOLD` | TODO | TBD at 0.05, verify via Pilot TAS smoke test |
| OQ-3 | Does NLDD require perplexity filtering? | decided | No, for now |
| OQ-4 | Should the spot check analysis group cover all remaining questions, or only sample a portion? | decision pending | Manually decided |

### 8.4 Pilot Confirmed Parameters Record

The following parameters are confirmed by Pilot Run results. Record the Pilot Run ID and justification for traceability.

| Parameter | Confirmed Value | Pilot Run ID | Justification |
|---|---|---|---|
| `generation.num_icl_groups` | 5 | — | 5 groups of ICL exemplars provide sufficient length gradient |
| `generation.samples_per_group` | 3 | — | 15 samples per question is enough to cover length bins |
| `generation.icl_group_temperatures` | See §8.1 | — | Per-group varying temps combined with ICL styles produce wide distributions |
| `generation.max_new_tokens` | 512 | — | Covers the longest CoT |
| `analysis.min_bin_size` | 5 | — | Ensures SE can be calculated for each bin |
| `analysis.num_spot_checks` | 3 | — | Balances cost and coverage |
| `analysis.max_extraction_fail_rate` | 0.05 | — | Met by Pilot's actual extraction success rate |
| `tas.plateau_threshold` | 0.05 | — | Awaiting Pilot TAS smoke test verification |

> Note: Pilot Run ID and detailed justifications will be manually backfilled after the actual Pilot is completed.
