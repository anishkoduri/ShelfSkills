# Build Your First Agent

The [starter README](../README.md) is the setup and submission guide. This page gives a practical
order for implementing `ParticipantAgent`. Keep the starter's signatures; add state only when
your learning strategy needs it.

## First Pass

1. Pick one thing to learn, such as a better extraction prompt or reusable attribute guidance.
2. Initialize it in `initialize` and retain the injected base model.
3. Use labeled training examples in `evolve` to propose a change. Return `changed=True` only when learned state changes.
4. Use `predict` for every item in the batch and preserve its `item_id` mapping.
5. Save all learned state and prove a fresh `load` restores identical prediction behavior.

For a first implementation, prompt-only learning is the simplest path. Review memory is another
option; fine-tuning is not required by the participant contract.

## Runnable Example

The starter includes a complete prompt-only example in [`examples/prompt-only`](../examples/prompt-only/).
It has its own `challenge.yaml` and `src/agent.py`, so you can validate and rehearse it without
changing the default starter agent:

```sh
datathon validate examples/prompt-only
datathon dev-run examples/prompt-only --data data/synth_v6_full \
    --output .datathon/prompt-example
```

That command uses mock models and demonstrates the lifecycle only. Mock mode does not provide a
prompt teacher, so this sample stops before revising its prompt; it still validates initialization,
prediction, save/load, and checkpoint capture. For live learning, configure your own accessible
base and teacher services in the example manifest and run with `--model-config model-policy.yaml`.
The optional [`.env.example`](../examples/prompt-only/.env.example) contains a placeholder
`DATABRICKS_CONFIG_PROFILE` only. Copy it to the repository-root `.env`, replace the profile name,
and source it in your shell; the CLI does not load `.env` automatically. The named profile must
already be configured in your Databricks CLI credentials file. Never put tokens or credentials in
`.env.example` or commit a real `.env`.
To use the example as your submission, copy its agent into
`src/agent.py`, set the root manifest's class to `PromptOnlyAgent`, and choose a `base_model` ID
allowed by the active organizer policy. Add your own participant-side validation gate before
adopting prompt changes; the sample intentionally demonstrates the contract, not a quality guarantee.

## Lifecycle

| Method | Use it for |
| --- | --- |
| `initialize(baseline)` | Start learned state and retain the injected model. |
| `evolve(context)` | Make one unattended learning update from labeled training examples. |
| `predict(batch)` | Predict every item in the batch using learned state and the injected model. |
| `save(path)` | Persist the full agent state after each successful `evolve` call. |
| `load(path)` | Rebuild learned state from a saved checkpoint directory. |

There is no `evaluate` method in the agent contract. `predict` returns predictions for one batch;
it does not grade them. `dev-run` drives learning but does not calculate Q or score checkpoints.
Local evaluation is participant-owned code around `predict`; protected evaluation is a separate
organizer operation on the frozen submission.

## The Harness Outer Loop

The harness owns the run lifecycle; your agent owns the learning strategy inside `evolve`:

1. The harness validates the folder, loads the training package, calls `initialize`, and saves the initial state as checkpoint `C000`.
2. It calls `evolve(context)`. Inside this method, your agent may select training examples, ask the metered teacher for a proposal, and compare the candidate against a participant-held validation split using the injected base model.
3. If `evolve` returns successfully and no runtime limit rejects the step, the harness calls `save` and captures the resulting state as the next checkpoint. It meters model usage and compute inside `evolve` as L. This capture happens regardless of the `changed` flag; `changed` describes whether agent state changed, while `stop` asks the harness to finish after the step.
4. The harness continues until the agent stops or a configured organizer limit is reached, then reports checkpoints, stop reason, and L. It does not calculate checkpoint Q or E during `dev-run`.
5. For an official attempt, checkpoint selection is separate from local evaluation. The agent can return an eligible `best_checkpoint_id`; a later `None` preserves that choice. If the agent never records a choice, the last checkpoint is the default, unless the organizer explicitly selects another eligible checkpoint.

This separation lets your agent own sampling, validation, candidate acceptance, and the checkpoint
it prefers, while the harness owns persistence, metering, hard limits, and final evaluation.

## Initialize and Learn

Keep the base model supplied by the harness. A prompt-only agent can revise its prompt using a
labeled training example and the metered teacher. Guard the optional teacher and empty data path:

```python
import json
from datathon.contract import BaselineContext, EvolutionContext, EvolutionResult

def initialize(self, baseline: BaselineContext) -> None:
    self.model = baseline.model
    self.prompt = "Extract only supported product attributes."
    self.revision_count = 0

def evolve(self, context: EvolutionContext) -> EvolutionResult:
    examples = [example for example in context.training_examples if example.labels]
    if not examples or context.prompt_teacher is None:
        return EvolutionResult(False, True, "No labeled examples or teacher available")

    example = examples[self.revision_count % len(examples)]
    public_attributes = [
        {
            "attribute_id": spec.attribute_id,
            "name": spec.attribute_name,
            "definition": spec.definition,
            "allowed_values": spec.allowed_values,
        }
        for spec in example.request.attribute_specs
    ]
    training_record = {
        "product_description": example.request.title,
        "attributes": public_attributes,
        "observed_labels": example.labels,
    }
    candidate = context.prompt_teacher.complete(
        "Improve the extraction prompt using this labeled training record. "
        "Return only the revised prompt.\n" + json.dumps(training_record, ensure_ascii=False)
    ).strip()
    if not candidate or candidate == self.prompt:
        return EvolutionResult(False, True, "No useful prompt change")

    self.prompt = candidate
    self.revision_count += 1
    return EvolutionResult(True, self.revision_count >= 3, "Updated extraction prompt")
```

In a real agent, choose examples deliberately, constrain the teacher's output, and persist the
revision count if it affects future learning. `changed=True`
means learned state changed; set `stop=True` when the agent should finish. `evolve` receives no
hold-out data or official Q score.

## Predict a Batch

Use the current batch item IDs for the returned results. Do not replace them with product IDs or
save them as persistent identities:

```python
from datathon.contract import BatchRules, PredictionBatch, PredictionBatchResponse

def predict(self, batch: PredictionBatch) -> PredictionBatchResponse:
    return self.model.predict(batch, BatchRules(system_prompt=self.prompt))
```

The injected model adapter maps each response back to its input item. For a memory-based agent,
pass per-product context with `BatchRules.item_context`, keyed by each current `item_id`. Store
learned guidance or reviewed examples in agent state, not batch IDs. Return one well-formed result
per input, including an empty list when you have no answer for a requested attribute.

## Save and Load

Write all learned state needed to reproduce predictions. Do not serialize the model or credentials;
the harness reattaches its model after loading.

```python
import json

def save(self, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    self.save_prediction_contract(output_path)
    state = {"prompt": self.prompt, "revision_count": self.revision_count}
    (output_path / "state.json").write_text(json.dumps(state), encoding="utf-8")

@classmethod
def load(cls, artifact_path: Path) -> "ParticipantAgent":
    cls.require_prediction_contract(artifact_path)
    state = json.loads((artifact_path / "state.json").read_text(encoding="utf-8"))
    agent = cls()
    agent.prompt = state["prompt"]
    agent.revision_count = state["revision_count"]
    return agent
```

This fragment assumes `Path` is imported and the class constructor initializes any other required
fields. Test a fresh `load` without training data before submitting.

## Choose and Restore the Final State

You do not return a replacement agent object from `evolve`. The harness calls `save` on the
current instance and records that directory as a checkpoint after each successful step. `C000`
is the initialized baseline; later successful steps create `C001`, `C002`, and so on. The `changed`
field describes whether your agent state changed; it does not control whether the harness captures
the checkpoint.

If your participant-owned validation prefers an earlier checkpoint, keep its ID in agent state and
return it in `EvolutionResult`:

```python
from datathon.contract import EvolutionContext, EvolutionResult

def evolve(self, context: EvolutionContext) -> EvolutionResult:
    return EvolutionResult(
        changed=state_changed,
        stop=should_stop,
        notes="Validation decision recorded",
        best_checkpoint_id=self.best_checkpoint_id,
    )
```

The ID must name an eligible checkpoint from this attempt; it may name the checkpoint being saved
by the current call. A later `best_checkpoint_id=None` preserves the previous choice. If you never
set one, the last eligible checkpoint is the default. The organizer may explicitly select another
eligible checkpoint when submitting.

For official evaluation, the harness loads the frozen checkpoint directory with your `load` method,
reattaches the injected base model, and calls `predict` on the protected evaluation batch. Your agent
must reconstruct its learned state from the saved files alone; it never receives hold-out labels.

## Common Patterns

- **Prompt-only:** revise a system prompt from labeled training examples; use it for the whole batch.
- **Review memory:** record corrections from training examples and retrieve relevant guidance by
  attribute specification or category profile at prediction time.
- **Validation:** reserve participant-owned products outside the learning package passed to
  `dev-run`; implement and report your own local metric. `dev-run` does not score checkpoints.
- **Usage:** teacher calls through `context.prompt_teacher` and base calls through `self.model` are
  metered. Do not create direct API clients or implement official evaluation inside the agent.

## Compare a Candidate Locally

The harness does not split the package or calculate checkpoint Q. If you want to decide whether
to keep a prompt or guidance change, use a repeatable, product-disjoint split of
`context.training_examples` inside your agent:

1. Keep the same reserved product IDs across revisions. Split by product, not by individual attribute rows.
2. Give only the training portion to the teacher or learning logic. Use the reserved labels only in your participant-owned evaluator.
3. Score the current and candidate states on the same reserved products with the exact observed-value Q formula in [the scoring reference](scoring.md).
4. Adopt a candidate only under a rule you choose, such as strictly higher Q with no malformed outputs; otherwise retain the current state.

Base-model calls made for validation from inside `evolve` are learning calls and contribute to L.
They are not official E. Calls made in a separate local script still incur your provider's charges,
but are not recorded in that experiment's ledger. Keep local scores clearly labeled as participant
estimates; only the organizer evaluates the frozen submission on hold-out data.

Use the included `model-policy.yaml` with `--model-config` for live local calls; without it,
`dev-run` uses mock models. See the [requirements reference](requirements.md#models-and-limits)
for the distinction between local routing and the official released policy.