# Architecture

The repository separates reusable market mechanics from research
orchestration.

```text
YAML configuration
        |
        v
AgentFactory -> SimulationRunner -> simulated market events
                                      |
BTG-OBD -> reconstruction -> empirical moments
                                      |
                                      v
                               grouped SMM score
                                      |
                                      v
proposal policy <--------------- persistent ledger
```

## Reusable packages

`lob_sim/core/` implements orders, trades, and the FIFO price-time-priority
book. `lob_sim/environment.py` routes actions and execution reports.
`lob_sim/archetypes/` contains agent policies, while
`lob_sim/agents/factory.py` maps YAML populations to concrete agents and gives
each population a deterministic random stream.

`lob_sim/simulation/` runs one scenario or a parallel ensemble.
`lob_sim/loss/` extracts stylized facts and implements the grouped simulated
method of moments objective. `lob_sim/optimization/` owns the search space,
candidate evaluation, proposal policies, caches, ledgers, runtime adapters,
and reports.

`real_lob_inference/` normalizes BTG-OBD events and reconstructs the empirical
book. `lob_sim/loss/real_facts.py` bridges reconstructed events to empirical
targets. The inference package does not depend on the simulator.

## Research workflows

`workflows/benchmark/` enforces budgets, seed schedules, environment versions,
feedback firewalls, stage audits, and per-ticker ledger isolation. The
reproduction workflow combines two isolated result roots without merging their policy
histories:

```text
v9 numerical root                         Sonnet follow-up root
sobol, tpe, bads, gp_bo,                  opro_plain,
mo_tpe, component_gp_bo                   opro_heuristic
             \                              /
              +---- published_summary -----+
```

The OPRO policy depends on the `ProposalRuntime` protocol in
`lob_sim/optimization/model_runtime.py`. OpenRouter and Anthropic-on-Foundry
are separate adapters. Runtime selection is explicit in each experiment
configuration and recorded in every adaptive proposal.

`workflows/validation/` verifies dependency versions, the heuristic registry,
empirical provenance, and offline prompt construction.

## Configuration resolution

References inside a YAML file resolve relative to that file. Raw data resolve
from the detected repository root. The frozen evaluator configuration is a
connected set:

```text
optimizer.yaml
  +-- scenario.yaml
  |     +-- agents.yaml
  |     +-- environment.yaml
  |     `-- ticker-books.yaml
  +-- ticker-books.yaml
  +-- later-evaluation.yaml
  +-- loss.yaml
  +-- heuristics.yaml
  +-- openrouter.yaml
  `-- llm-models.yaml
```

The Foundry follow-up supplies a separate runtime config and model registry but
continues to reference the same evaluator graph.

## Reproducibility boundaries

Candidate caches fingerprint simulator sources, evaluator configs, empirical
target files, raw-data identities, date splits, and runtime overrides. Stable
population IDs make an agent population's random stream independent of
unrelated population ordering.

These fingerprints protect resume safety on one execution tree. Some include
resolved paths and source-file metadata, so they are not intended as
cross-machine content checks. Numerical optimizers are deterministic under the
recorded environment and data. Hosted LLM responses may change across a new
run even when prompts and request settings are identical.

Generated outputs are written under `results/`. `published_summary.py`
regenerates numeric summaries from complete ledgers.
