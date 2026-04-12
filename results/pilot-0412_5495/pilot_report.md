# Pilot Report

- Mode: real
- Dataset: gsm8k:test
- Pilot questions: 50
- Prompt groups: icl_minimal, icl_short, icl_medium, icl_detailed, icl_verbose
- Traces written: 3750

## Check Results

### A. ICL Length Guidance
[PASS] Prompt-group median step counts increase cleanly with enough spread.
- prompt_group_medians: {'icl_minimal': 1.0, 'icl_short': 3.0, 'icl_medium': 4.0, 'icl_detailed': 6.0, 'icl_verbose': 11.0}
- global_step_range: 29
- occupied_length_bins: 30
- missing_prompt_ids: []

### B. Per-Length Sample Volume
[WARN] At least two bins are occupied, but some bins are still sparse.
- occupied_bins: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
- bin_counts: {0: {'trace_count': 1, 'correct_count': 0}, 1: {'trace_count': 566, 'correct_count': 341}, 2: {'trace_count': 407, 'correct_count': 325}, 3: {'trace_count': 604, 'correct_count': 502}, 4: {'trace_count': 380, 'correct_count': 306}, 5: {'trace_count': 324, 'correct_count': 255}, 6: {'trace_count': 268, 'correct_count': 218}, 7: {'trace_count': 200, 'correct_count': 149}, 8: {'trace_count': 150, 'correct_count': 114}, 9: {'trace_count': 173, 'correct_count': 150}, 10: {'trace_count': 180, 'correct_count': 151}, 11: {'trace_count': 92, 'correct_count': 70}, 12: {'trace_count': 203, 'correct_count': 169}, 13: {'trace_count': 50, 'correct_count': 37}, 14: {'trace_count': 35, 'correct_count': 23}, 15: {'trace_count': 34, 'correct_count': 25}, 16: {'trace_count': 17, 'correct_count': 13}, 17: {'trace_count': 17, 'correct_count': 9}, 18: {'trace_count': 10, 'correct_count': 8}, 19: {'trace_count': 14, 'correct_count': 4}, 20: {'trace_count': 6, 'correct_count': 1}, 21: {'trace_count': 4, 'correct_count': 0}, 22: {'trace_count': 3, 'correct_count': 1}, 23: {'trace_count': 3, 'correct_count': 0}, 24: {'trace_count': 2, 'correct_count': 0}, 25: {'trace_count': 1, 'correct_count': 0}, 26: {'trace_count': 2, 'correct_count': 0}, 27: {'trace_count': 1, 'correct_count': 1}, 28: {'trace_count': 2, 'correct_count': 0}, 29: {'trace_count': 1, 'correct_count': 0}}
- total_correct_traces: 2872
- median_occupied_bin_count: 25.5

### C. Segmentation And Extraction
[PASS] Extraction and segmentation stay within Pilot thresholds.
- total_traces: 3750
- extraction_failed_rate: 0.0133
- zero_step_rate: 0.0003
- pass_threshold: 0.05
- warn_threshold: 0.1

### D. Corruption Feasibility
[WARN] Corruption is possible, but failure or token drift is still noticeable.
- correct_traces: 2872
- step_attempts: 15745
- corruption_failed_rate: 0.2688
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
| `generation.num_icl_groups` | `5` | Copy the Pilot prompt-group count. |
| `generation.samples_per_group` | `15` | Double only when check B fails; otherwise copy the Pilot value. |
| `generation.temperature` | `0.7` | Copy the Pilot temperature. |
| `generation.max_new_tokens` | `544` | Use the larger of Pilot max_new_tokens and observed max token_count + 32. |
| `analysis.min_bin_size` | `5` | Use 5 when the median occupied-bin count is at least 5; else 3. |
| `analysis.max_extraction_fail_rate` | `0.05` | Copy the Pilot extraction-fail threshold. |
| `tas.plateau_threshold` | `0.05` | Provisional Stage D default because check E is deferred. |
| `analysis.num_spot_checks` | `3` | Provisional Stage D default because check E is deferred. |

## Notes

- Check E is intentionally deferred to Stage F and does not exercise `src/nldd.py` in Stage D.
- `tas.plateau_threshold` and `analysis.num_spot_checks` are provisional defaults until Stage F smoke data exists.
- The backfill table is intended to be copied into `configs/stage1.yaml` by hand after Pilot review.
