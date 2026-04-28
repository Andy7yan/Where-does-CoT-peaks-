# TAS Null-Hypothesis Diagnostic

- CSV: `results\output-0427\tas_diagnostic.csv`
- PNG: `results\output-0427\tas_diagnostic.png`
- Trace count: 12962
- Length bins: 15
- Isotropic baseline status: `missing_step_norms`

## Fits

- `a / sqrt(L)`: R2=0.861468, AIC=-96.4036, a=0.678383
- `a + b log(L)`: R2=0.983403, AIC=-126.231, a=0.654601, b=-0.196313

## Isotropic Baseline

Strict distribution-matched isotropic simulation was not run because no hidden-state step-norm JSONL was available.
Re-run with `--recompute-step-norms` in an environment that has the configured local model cache and enough compute, or pass `--step-norms-path`.
