# Published experiment configuration

These files define the simulator, data split, loss, optimizer settings, and
execution environment used in the published experiment.

Do not edit these files when running the published study. Create a new config
directory, study name, and output root for changed agents, losses, seeds,
search spaces, models, or data.

## Main inputs

- `component-aware-comparison.yaml`: five-ticker v9 comparison. The
  reproduction workflow selects its six numerical arms.
- `optimizer.yaml` and `later-evaluation.yaml`: Calibration Book and Test Book
  evaluator templates.
- `ticker-books.yaml`: five symbols, whole-date split, and ten half-hour
  windows.
- `scenario.yaml`, `agents.yaml`, `environment.yaml`, and `loss.yaml`: frozen
  simulator and grouped SMM objective.
- `heuristics.yaml`: frozen heuristic OPRO treatment.
- `component-aware-environment-lock.yaml` and
  `xeon-e2146g-execution.yaml`: dependency and reference-host execution
  contracts.

The two current OPRO arms live in
`configs/experiments/followup_foundry_sonnet5/`. They reference these files so
the evaluator, books, initial design, seeds, loss, and heuristic treatment stay
unchanged.

Use `python -m workflows.benchmark.reproduce_published_study --phase plan` to
see the complete execution plan.
