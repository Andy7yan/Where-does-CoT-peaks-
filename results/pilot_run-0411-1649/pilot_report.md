# Pilot Report

- Mode: real
- Dataset: gsm8k:test
- Pilot questions: 50
- Prompt groups: icl_detailed, icl_medium, icl_short
- Traces written: 300

## Check Results

### A. ICL Length Guidance
[FAIL] Prompt groups do not create enough distinct effective lengths.
- prompt_group_medians: {'icl_detailed': 6.5, 'icl_medium': 6.5, 'icl_short': 6.5}
- global_step_range: 9
- occupied_length_bins: 9
- missing_prompt_ids: []

### B. Per-Length Sample Volume
[WARN] At least two bins are occupied, but some bins are still sparse.
- occupied_bins: [3, 4, 5, 6, 7, 8, 9, 11, 12]
- bin_counts: {4: {'trace_count': 54, 'correct_count': 12}, 5: {'trace_count': 48, 'correct_count': 0}, 6: {'trace_count': 42, 'correct_count': 12}, 8: {'trace_count': 54, 'correct_count': 0}, 12: {'trace_count': 6, 'correct_count': 0}, 9: {'trace_count': 42, 'correct_count': 0}, 7: {'trace_count': 42, 'correct_count': 0}, 3: {'trace_count': 6, 'correct_count': 6}, 11: {'trace_count': 6, 'correct_count': 0}}
- total_correct_traces: 30
- median_occupied_bin_count: 42.0

### C. Segmentation And Extraction
[FAIL] Extraction or segmentation quality is too weak for a formal run.
- total_traces: 300
- extraction_failed_rate: 0.88
- zero_step_rate: 0.0
- pass_threshold: 0.05
- warn_threshold: 0.1

### D. Corruption Feasibility
[FAIL] Corruption quality is too unstable for a formal NLDD stage.
- correct_traces: 30
- step_attempts: 138
- corruption_failed_rate: 0.6522
- token_delta_violation_rate: 0.0
- token_delta_max: 2
- token_counter: model_tokenizer

### E. NLDD Smoke
[WARN] Deferred to Stage F by stage-boundary decision.
- tas_plateau_threshold_default: 0.05
- analysis_num_spot_checks_default: 3

## Backfill Recommendations

| Config Field | Recommended Value | Basis |
|---|---:|---|
| `dataset.subset_size` | `200` | Use 400 when check B fails; otherwise use 200. |
| `generation.num_icl_groups` | `3` | Copy the Pilot prompt-group count. |
| `generation.samples_per_group` | `2` | Double only when check B fails; otherwise copy the Pilot value. |
| `generation.temperature` | `0.0` | Copy the Pilot temperature. |
| `generation.max_new_tokens` | `160` | Use the larger of Pilot max_new_tokens and observed max token_count + 32. |
| `analysis.min_bin_size` | `5` | Use 5 when the median occupied-bin count is at least 5; else 3. |
| `analysis.max_extraction_fail_rate` | `0.05` | Copy the Pilot extraction-fail threshold. |
| `tas.plateau_threshold` | `0.05` | Provisional Stage D default because check E is deferred. |
| `analysis.num_spot_checks` | `3` | Provisional Stage D default because check E is deferred. |

## Notes

- Check E is intentionally deferred to Stage F and does not exercise `src/nldd.py` in Stage D.
- `tas.plateau_threshold` and `analysis.num_spot_checks` are provisional defaults until Stage F smoke data exists.
- The backfill table is intended to be copied into `configs/stage1.yaml` by hand after Pilot review.
