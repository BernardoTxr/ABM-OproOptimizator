# Benchmark workflows

The main reproduction entry point is:

```bash
python -m workflows.benchmark.reproduce_published_study --phase plan
```

It ties together the six frozen numerical baselines and the isolated two-arm
Claude Sonnet 5 follow-up used by the paper. The phases prepare and freeze the
empirical targets, stage the optimizers, resume calibration ledgers, score
incumbents on Test Books, and write a combined machine-readable summary. See
[`../../REPRODUCING.md`](../../REPRODUCING.md).

## Current modules

- `reproduce_published_study.py`: phase-oriented workflow and command planner.
- `published_summary.py`: validates complete raw ledgers and combines the two
  study roots into JSON and CSV tables.
- `run_component_aware.py` and `run_component_aware_parallel.py`: guarded v9
  calibration, with explicit method filters used for the six numerical arms.
- `opro_pair_followup.py` and `opro_pair_followup_parallel.py`: isolated plain
  and heuristic OPRO follow-up.
- `preflight_proposal_runtime.py`: provider-neutral, dual-condition live
  response-contract test that consumes no optimizer slot or simulation.
- `evaluate_component_aware.py`: Test Book evaluation for a selected method
  tuple.
- `audit_component_stage.py`, `ticker_books.py`, `usage_report.py`, and
  `score_predictions.py`: stage checks, fixed book tasks, and audit reports.
