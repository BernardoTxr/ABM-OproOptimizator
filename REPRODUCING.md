# Reproducing the published experiment

This guide reruns the exact scientific protocol used for the paper's final
eight-method comparison. The reproduction workflow keeps the historical v9
numerical ledgers and the later Sonnet OPRO ledgers in separate study roots,
then combines their validated summaries.

## 1. Install the recorded environment

Use Python 3.12 on Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-experiment PYTHON=python
make install-followup-foundry PYTHON=python
```

The lock files pin the project's direct dependencies. They are not a
hash-locked transitive environment, so retain `python -m pip freeze` with a new
execution if byte-level environment provenance is required.

Run the offline gates before supplying data or credentials:

```bash
python -m workflows.benchmark.reproduce_published_study --phase check
```

This validates dependency versions and the heuristic registry, runs the
offline smoke test and test suite, and checks both study configurations with
dry runs. It performs no live model call and consumes no optimizer slot.

## 2. Add BTG-OBD-A26

Obtain the dataset and create the 90-file tree described in [DATA.md](DATA.md).
Then materialize the Calibration Book and Test Book targets and freeze their
provenance:

```bash
python -m workflows.benchmark.reproduce_published_study --phase prepare
```

The data, targets, and provenance record stay below ignored paths. A frozen
record is never overwritten silently.

## 3. Run the six numerical baselines

The portable mode runs tickers sequentially and uses three simulation workers
per ticker. It preserves every scientific input while avoiding assumptions
about CPU numbering:

```bash
python -m workflows.benchmark.reproduce_published_study \
  --phase baselines --mode portable
```

On a Linux host matching
[`configs/paper/xeon-e2146g-execution.yaml`](configs/paper/xeon-e2146g-execution.yaml),
the recorded schedule runs five ticker processes with three simulation workers
each and pins them to logical CPUs 0–10:

```bash
python -m workflows.benchmark.reproduce_published_study \
  --phase baselines --mode reference
```

Only `sobol`, `tpe`, `bads`, `gp_bo`, `mo_tpe`, and `component_gp_bo` are
selected. The two historical Qwen OPRO arms in the v9 config are not run.
Every method is first stopped after its first adaptive slot for a stage audit;
the next command resumes the same ledgers to 50 charged slots.

## 4. Configure and run the two OPRO arms

Copy the blank template and supply values for a Claude Sonnet 5 deployment you
are authorized to use:

```bash
cp .env.example .env.local
```

Set:

- `ANTHROPIC_FOUNDRY_API_KEY`
- `ANTHROPIC_FOUNDRY_BASE_URL`
- `ANTHROPIC_FOUNDRY_DEPLOYMENT`

The repository loader reads `.env.local`; an already exported variable takes
precedence. Run:

```bash
python -m workflows.benchmark.reproduce_published_study --phase opro
```

The phase makes one live maximum-history preflight for each OPRO condition,
then stages and completes plain and heuristic OPRO with the recorded three
ticker workers and two simulation workers per ticker. It can make up to 1,900
adaptive model calls across 2,500 charged OPRO slots. Live API use can incur
substantial cost.

The deployment is external to this repository. The workflow preserves the
paper's request settings and records exact prompts, responses, model/request
identities, retries, latency, token categories, and the pricing source. A new
service execution may still yield a different proposal trajectory from the
one reported in the paper.

## 5. Evaluate Test Books and regenerate the combined table

```bash
python -m workflows.benchmark.reproduce_published_study --phase evaluate
python -m workflows.benchmark.reproduce_published_study --phase summarize
```

Evaluation selects the lowest-calibration incumbent across five repetitions
for each ticker/method and scores it once on that ticker's 90 Test Books. The
summary command reads both raw study roots and refuses partial or mixed
artifacts. It checks:

- all five tickers and all eight published methods;
- repetitions 0–4 and 50 charged slots per run;
- the common 12-slot initial trajectory across both study roots;
- complete and finite best-so-far trajectories;
- the selected Test Book incumbent against its calibration ledger; and
- the OPRO completion and adaptive-call contract.

It writes machine-readable files to:

```text
results/reproduction_summary/summary.json
results/reproduction_summary/summary.csv
```

## Complete commands and resume behavior

Inspect every subprocess before a large run:

```bash
python -m workflows.benchmark.reproduce_published_study --phase plan
```

For a prepared machine, the full portable sequence is:

```bash
python -m workflows.benchmark.reproduce_published_study \
  --phase all --mode portable
```

Policy ledgers are written atomically and the underlying study runners resume
completed slots. Keep one execution mode and worker count for the life of a
study because the worker override participates in the evaluator fingerprint.
Do not copy, merge, or append policy directories between the two study roots.

The original reference run was designed for roughly a day on the recorded
Xeon host. Actual time and model cost depend on hardware, endpoint capacity,
retries, and service latency.
