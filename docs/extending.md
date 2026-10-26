# Extending the simulator

Use a new config directory and study name for every independent experiment.
The files under `configs/paper/` define the published protocol and should stay
unchanged.

## Create a new agent

1. Add a class under `lob_sim/archetypes/` that subclasses `BaseArchetype`.
2. Implement `act(obs) -> list[Action]` and `reset()`.
3. Add the new class to `lob_sim/archetypes/__init__.py`.
4. Register its YAML name and constructor arguments in
   `lob_sim/agents/factory.py`.
5. Add a population entry to your copied `agents.yaml`.
6. Add unit tests for action validity, reset behavior, seeded determinism, and
   inventory or order-lifecycle constraints.

An observation provides the timestamp, visible bid and ask levels, the agent's
execution reports, and its currently open orders. An agent returns `Action`
objects with one of `PLACE_LIMIT`, `PLACE_MARKET`, or `CANCEL`. Agents must not
mutate the book directly.

Use a stable population `id` in YAML. Instances are named `<id>_NNNN`, and that
ID participates in random-stream derivation. Parameters may be fixed scalars or
sampled specifications:

```yaml
parameters:
  activity: 0.2
  order_size:
    type: lognormal
    mean: 5.8
    std: 0.4
```

Supported sampling types are `fixed`, `uniform`, `normal`, `lognormal`, and
`exponential`.

## Modify the paper-style SMM loss

Copy `configs/paper/loss.yaml` and `configs/paper/optimizer.yaml`. Point the
copied optimizer's `loss_config` to your new loss file and give it a new
`study_name`.

The loss groups named in YAML must be implemented by
`compute_moment_vector()` in `lob_sim/loss/smm.py`. To add a moment:

1. Add its name to `AVAILABLE_MOMENTS`.
2. Compute it in `compute_moment_vector()` for both real and simulated facts.
3. Place it in exactly one group in the copied loss config.
4. Point `smm_target_cache` and `smm_window_moment_cache` at new paths so the
   empirical target and bootstrap standard errors are regenerated.
5. Add tests covering finite behavior, sign, units, and group aggregation.

The total objective is the mean of group scores, so adding ten moments to one
group does not give that group ten times the priority. Each moment is
standardized by its empirical date-cluster bootstrap standard error.

Do not reuse paper trial directories after changing the loss, simulator, data,
or config. The evaluator fingerprint will reject stale ledgers, but a distinct
study name also makes the research intent obvious.

## Change the optimizer search space

Each dimension in `optimizer.yaml` declares a type, bounds or choices, and one
or more target fields:

```yaml
- name: maker_base_depth
  type: float
  low: 5000.0
  high: 80000.0
  log: true
  targets:
    - agent: market_maker
      field: parameters.order_size
```

The `agent` value is a population ID from `agents.yaml`; `field` is a dotted
path inside that entry. `SearchSpace.validate()` is authoritative: OPRO text,
TPE suggestions, and hand-written candidates all pass through the same bounds
and shape checks.

If using heuristic OPRO, update a copied component-mechanism registry so every
control refers to a dimension in the copied search space and every primary or
collateral loss name belongs to the grouped feedback vector. Do not add signed
moment residuals or residual-trigger rules to the proposal interface.

## Add a different loss family

For a fundamentally different objective, implement it under `lob_sim/loss/`
and call it from `TrialEvaluator`. Keep these responsibilities separate:

- moment or fact extraction;
- empirical target construction;
- candidate scoring;
- validity gates;
- serialized feedback exposed through an explicit policy information firewall.

That separation prevents an optimizer from receiving information that was not
available to the comparison arms and makes cached objectives auditable.

## Verify an extension

At minimum, run:

```bash
pytest -q
lob-sim run --config path/to/scenario.yaml
lob-sim optimize --config path/to/optimizer.yaml --sampler random --repetition 0 --budget 2
```

Inspect the generated scenario metadata and trial `result.json` as well as the
aggregate score. A plausible price plot alone is not a validation test.

## Add an OPRO runtime

Do not modify `configs/paper/`. Implement the `ProposalRuntime` protocol in
`lob_sim/optimization/model_runtime.py`, register the adapter in
`runtime_factory.py`, and create a separately versioned experiment directory
with a new study name and result root. Keep provider credentials in environment
variables and record the runtime identity, request ID, retries, latency, token
categories, pricing source, prompt, raw response, and errors in the local
ledger.

Before calibration, add offline contract tests and a live dual-condition
preflight using `preflight_proposal_runtime.py`. Stage the first adaptive slot
for every method/ticker pair and verify that the shared initial observations,
evaluator identity, budget, and information firewall are unchanged. A new
provider or model is a new study; never resume it into an existing policy
ledger.
