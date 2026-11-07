# Participant Requirements

This is a short participant reference. Follow the [starter README](../README.md) for setup,
experiments, troubleshooting, and submission. The organizer's activated release is authoritative
if it differs from this checked-in starter.

## Data

- Use the train-only package in `data/<version>`. The starter includes `data/synth_v6_full`.
- The package contains `product_attribute_experiment.jsonl` and `dataset.json`. It has no hold-out rows or labels.
- Keep hold-out data, credentials, and private run artifacts out of Git. Do not copy hold-out data into your workspace or agent.
- `dev-run --data ...` gives every package record to learning; it does not reserve a validation split or score checkpoints. For local quality checks, create a separate validation subset and do not pass it to the learning run.
- Learning labels are available only in `context.training_examples`; prediction requests contain no labels. Missing labels mean unannotated, not negative.

## Agent Contract

`challenge.yaml` selects the `ParticipantAgent` and `challenge_version: "2.0"`. Keep the method signatures in `src/agent.py`.

| Method | Requirement |
| --- | --- |
| `initialize(baseline)` | Initialize learned state and keep `baseline.model` as the injected model. |
| `evolve(context)` | Learn unattended from `context.training_examples`; return `EvolutionResult`. |
| `predict(batch)` | Return one result for every input `item_id`, in any order, with requested attribute IDs mapped to lists of strings. |
| `save(path)` | Persist every learned value needed to reproduce predictions. |
| `load(path)` | Restore from the saved directory alone; do not require training data or an in-memory agent. |

Use the injected base model and `context.prompt_teacher` only. Do not create model clients, read
credentials, access hold-out data, or implement official scoring in the agent. See the
[agent API examples](agent-api.md) for practical patterns.

## Models and Limits

- `model-policy.yaml` ships with the starter. Its `models` section lists allowed model IDs and harness-credit rates; its `scoring` section carries the current provisional scoring constants. Organizers may replace this file when the policy changes.
- For live local calls, run `datathon dev-run --model-config model-policy.yaml` and set your own accessible service names in `challenge.yaml`. The policy file intentionally omits service endpoints and credentials. Without `--model-config`, `dev-run` uses mock models.
- Live calls use your workspace identity and may incur account charges. Local service names and model choices do not change the official policy; official runs use the active organizer release.
- Learning is metered as L without a harness credit cap. It does not make learning free of account or service charges.
- Step and runtime limits are organizer-controlled and may be absent from the current testing config. Ask organizers for active event limits; do not add them to your manifest.
- Each official submission uses a pushed Git commit. The active release defines how many submissions are allowed; a commit can be submitted only once.

## Before Submission

1. Run `datathon validate .` and confirm GitLab validation passes for the commit.
2. Confirm learned state survives `save` and a fresh `load`.
3. Keep credentials, hold-out data, and private run outputs out of Git. The published train-only experiment package under `data/<version>` may be committed for team use.
4. Send organizers the full pushed commit SHA. Organizers run and score the official attempt.