# ABM OPRO Optimizer

Research code for **“Do Domain Heuristics Help LLM Optimizers Calibrate
an Agent-Based Limit Order Book?”**

This repository contains the limit-order-book simulator, empirical-book
reconstruction code, frozen experiment configurations, optimizer
implementations, and commands needed to rerun the study.

## Authors

- Bernardo Teixeira — Télécom Paris, Institut Polytechnique de Paris; BTG Alpha
  Lab
- João Pedro Alexandrino — Universidade de Fortaleza (UNIFOR)
- Renan Avila — Escola Politécnica, Universidade de São Paulo; BTG Alpha Lab

Citation metadata is available in [CITATION.cff](CITATION.cff).

## Quick start

Python 3.12 is required.

```bash
git clone https://github.com/BernardoTxr/ABM-OproOptimizator.git
cd ABM-OproOptimizator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install PYTHON=python
make smoke PYTHON=python
make test PYTHON=python
```

The smoke check runs without data, API credentials, or network access. A small
standalone simulation is also available:

```bash
lob-sim run --config configs/examples/smoke.yaml
```

## Reproduce the study

The paper's final comparison consists of six numerical optimizers from the
frozen v9 study and two Claude Sonnet 5 OPRO arms from the isolated follow-up:

| Study root | Methods |
| --- | --- |
| `icaif_smm_v9_ticker_specific_10x30m` | Sobol, TPE, BADS, GP-BO, multi-objective TPE, component-aware GP-BO |
| `icaif_smm_v9_followup_foundry_sonnet5` | plain OPRO, heuristic OPRO |

First inspect the complete execution plan. This does not read the dataset,
create results, contact an API, or consume optimizer slots.

```bash
make reproduce-plan PYTHON=python
```

Then follow [REPRODUCING.md](REPRODUCING.md) to obtain BTG-OBD-A26, configure
the model endpoint, run each phase, and regenerate a combined JSON/CSV
summary. The complete portable run is:

```bash
make install-followup-foundry PYTHON=python
cp .env.example .env.local       # fill in your own Foundry values
make reproduce PYTHON=python
```

This is a large, paid experiment. It evaluates five tickers, five repetitions,
and 50 optimizer slots per method. The reference-host schedule is available as
`make reproduce-reference`; use it only on a compatible 12-logical-CPU Linux
host.

The numerical arms are deterministic under the recorded inputs. Hosted LLM
responses may vary between runs even when the protocol and request settings
are unchanged.

## Data

The study uses the BTG-OBD-A26 dataset published by BTG Alpha Lab. It is not
redistributed here. After obtaining it from the
[BTG Alpha Lab dataset catalog](https://alphalab.btgpactual.com/datasets), put
the 90 Parquet files in this layout:

```text
BTG-OBD-A26/
  ASAI3/20251013/d10/events.parquet
  ...
  MGLU3/20260129/d10/events.parquet
```

The exact five tickers, 18 dates, split, and ten intraday windows are recorded
in [`configs/paper/ticker-books.yaml`](configs/paper/ticker-books.yaml). See
[DATA.md](DATA.md) for preparation and provenance details.

## Repository map

- `lob_sim/`: matching engine, agent archetypes, simulation, loss, optimizers,
  caches, and ledgers.
- `real_lob_inference/`: BTG-OBD event normalization and empirical-book
  reconstruction.
- `workflows/benchmark/`: calibration, evaluation, reporting, and reproduction
  workflows.
- `workflows/validation/`: environment, protocol, provenance, and offline smoke
  checks.
- `configs/paper/`: frozen published protocol inputs.
- `configs/experiments/followup_foundry_sonnet5/`: isolated two-arm OPRO
  follow-up.
- `tests/`: unit and workflow-contract tests that need no market data or API.

Configuration references resolve relative to the YAML file containing them.
Raw-data paths resolve from the repository root.

## License

This project is available under the [MIT License](LICENSE).
