# Datathon team repository

Read the [challenge overview](CHALLENGE.md) for the goal, learning strategies, and scoring trade-offs.
This README covers setup, experiments, rules, and submission.

Every team starts with the same weak agent. Improve it with harness-metered learning, then
submit it. The organizer scores each submission once on hold-out data you never see. The active
organizer release sets the submission limit; ask the organizers for the event limit.

## Start Here

1. Get access to your team repository and accept the GitLab invitations. Set up your own
   Databricks workspace only if you plan to run live model experiments.
2. Install the announced harness pin and run `datathon validate .`.
3. Implement `src/agent.py`, then experiment with participant data in a new `.datathon/` output
   directory. Local `dev-run` does not report prediction quality.
4. Commit and push your agent, confirm GitLab validation passes, and send the organizer the full
   commit SHA. The organizer runs and scores that pushed revision.

Details: [connect accounts](#part-1-connect-gitlab-and-databricks),
[build and experiment](#part-2-build-and-experiment),
[submit for scoring](#part-3-submit-for-scoring), and [read your results](#results).

Optional references: [agent API in practice](docs/agent-api.md),
[participant requirements](docs/requirements.md), and [scoring](docs/scoring.md).

Rules and troubleshooting are at the end.

## How you are scored

The harness measures three numbers; your code never reports them.

| Number | Meaning |
| --- | --- |
| Q | Micro-F1 of exact observed attribute/value predictions on hold-out data |
| E | Inference efficiency (products, actual model calls, tokens, batch timing) and what your base model's tokens cost in credits |
| L | Learning cost of the checkpoint you submit, in credits |

The leaderboard also shows a provisional composite of Q, E and L; its weights and references are
printed on the board.

## Part 1. Connect GitLab and Databricks

Your code lives in GitLab; your own Databricks workspace is where you can run experiments at larger
scale. Set up both once.

### GitLab: your team repository

**1. Send the organizers your team details** (one message per team):

- A team name: lower case with dashes, for example `team-delta`. It names your repository.
- One email address per member: the address each person uses for their GitLab account. Each
  person needs a GitLab account first (free at gitlab.com), and belongs to one team only.
- Optionally, a display name for the leaderboard (the default is the team name).

**2. Get your repository.** The organizers create it from this starter and invite each member by
email to the project and, with read access, to the shared `tamu-2026-ci` and leaderboard projects.
Accept the invitation (it arrives at the address you sent), then clone the repository; the project
page's Clone button gives the URL. If no invitation arrives, check that the address matches your
GitLab account's email and tell the organizers.

**3. Create a registry token.** You need it to install the harness and download experiment data: a
GitLab personal access token with the `read_api` scope (Preferences, Access tokens). Keep it in a
shell variable and never put it in this repository.

### Databricks: your team workspace

> **Draft:** this section is still to be confirmed with the Databricks team.

Each team owns and manages its own Databricks account and workspace. Organizers do not need
access to it, its account ID, workspace URL, or credentials. Databricks may give teams credit;
the amount and how to claim it are to be announced. Account charges and free-tier quotas are
separate from the harness's measured learning credits (L).

1. **Create or use your team's account.** See Databricks'
   [Free Edition](https://docs.databricks.com/aws/en/getting-started/free-edition). Confirm that
   your account supports the compute and model services you intend to use; free-tier availability
   is not a guarantee that every event model can be served there.
2. **Open your workspace.** Use an existing workspace or create one if your account supports it.
   Configure your own compute, SQL warehouse, storage and permissions. You do not need the
   organizer's workspace name or region, and you do not invite the organizers.
3. **Keep account access private.** Share access with your teammates as needed. Use your own
   identity for experiments; never obtain or use an organizer service-principal credential.
4. **Create a Git token for Databricks.** Databricks reaches your team repository with your own
   GitLab personal access token with the `read_repository` and `write_repository` scopes, which
   allow clone, pull and push. Use a short expiry and keep it out of this repository. See GitLab's
   [personal access tokens](https://docs.gitlab.com/user/profile/personal_access_tokens/) and
   Databricks' [Connect your Git provider](https://docs.databricks.com/aws/en/repos/get-access-tokens-from-git-provider)
   (GitLab section).
5. **Link GitLab in Databricks.** In your workspace, click your username in the top bar, then
   Settings, Linked accounts, Add Git credential (a personal setting, not under admin or workspace
   settings). Choose GitLab, enter your GitLab email and paste the Git token. See
   [Configure Git integration](https://docs.databricks.com/aws/en/repos/repos-setup).
6. **Clone the repository into a Git folder.** In Workspace, Create, Git folder, paste the
   repository URL and choose GitLab. See
   [Clone a Git repository](https://docs.databricks.com/aws/en/repos/git-operations-with-repos).
   Commit and push from the Git folder; the push runs `datathon validate` in GitLab like any other.

### Databricks: connect from VS Code (optional)

Instead of working in the browser, you can use your workspace from VS Code with the
[Databricks IDE extension](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/). It runs local
Python files and notebooks on your workspace's compute, syncs local code to the workspace and
debugs with Databricks Connect. To set it up:

1. Install the extension from the VS Code marketplace and open this repository.
2. Click the Databricks icon in the sidebar, then Create configuration (or Create a new project).
3. Enter your workspace host URL and sign in with the
   [OAuth option](https://docs.databricks.com/aws/en/dev-tools/vscode-ext/authentication).
4. Choose compute supported by your account and the extension; serverless availability depends
   on your account and environment.

The extension is a convenience only. Submissions still use your pushed Git commit, and the
extension's login stays on your machine; never commit it or its tokens.

## Part 2. Build and experiment

Experiments are not official attempts and cannot be submitted. Run as many as you like.

### Install the harness

You need Python 3.11 and the pinned harness, which lives in the package registry of the GitLab
project `tamu-2026-ci` (you can read it).

Use the 0.2.14 pin below only after the organizer announces that release. A source checkout or
locally staged wheel does not mean the package has been published or official jobs activated.

```sh
export GITLAB_TOKEN="<your registry token>"
python -m venv .venv && . .venv/bin/activate
pip install uv
uv pip install --default-index https://pypi.org/simple \
  --index "https://__token__:${GITLAB_TOKEN}@gitlab.com/api/v4/projects/mlmodels%2Ftamu-2026-shared%2Ftamu-2026-ci/packages/pypi/simple" \
   --index-strategy first-index datathon==0.2.14
datathon validate .
```

`--index-strategy first-index` matters: PyPI has an unrelated package called `datathon`.
`datathon validate .` runs eight checks; GitLab runs the same checks on every push.

Start from the freshly generated team repository. Its package pin, manifest, and agent already
match the current batch contract; no earlier agent or checkpoint needs to be migrated.
Keep `inference.batch_size: 1` unless you choose another size, then implement your agent below.
Official runs use the organizer's activated release, not your local wheel or package selection.

### Upgrade an existing team repository

After the organizer announces 0.2.14, update your existing repository; do not regenerate it or
replace your agent. Change the harness line in `requirements.txt` to `datathon==0.2.14` and
update any older installation pins in your README or notebook setup. Existing repositories
do not automatically receive changes made to this starter.

With your registry token already available privately as `GITLAB_TOKEN`, reinstall in the
Python 3.11 environment used by your agent and notebook:

```sh
uv pip install --python .venv/bin/python --reinstall-package datathon \
   --default-index https://pypi.org/simple \
   --index "https://__token__:${GITLAB_TOKEN}@gitlab.com/api/v4/projects/mlmodels%2Ftamu-2026-shared%2Ftamu-2026-ci/packages/pypi/simple" \
   --index-strategy first-index "datathon[databricks]==0.2.14"
.venv/bin/python -c "from importlib.metadata import version; print(version('datathon'))"
.venv/bin/datathon validate .
```

The version check must print `0.2.14`. Restart the notebook kernel using that environment,
then run a new development experiment in a new output directory. Commit and push the pin
and setup-document changes and confirm GitLab validation passes. Keep tokens out of Git.
The pin changes installed code only; allowed model IDs and official limits still come from
the organizer's released policy. A package upgrade does not create an official submission.

### Write your agent

Edit `src/agent.py` (`ParticipantAgent`) and keep `challenge.yaml` pointing at it.
Work through the steps below. The starter passes the interface checks but does not learn and
returns empty predictions for product attributes; implementing the interface alone is not an
improvement in quality.

#### 1. Understand the inputs your agent receives

The experiment-data JSON and the agent interface are different representations of the same
products. **Do not add a `request` field to the downloaded JSON.** The harness reads each
experiment record and constructs a label-free `PredictionRequest` for your agent.

| Experiment-data field | Available to your agent as |
| --- | --- |
| `product_description` | `request.input_text` (an alias for `request.title`) |
| `product_id` | `request.product_id` |
| `attributes` specification objects | Attribute IDs in `request.attributes`; public definitions in `request.attribute_specs` |
| `labels` array | `example.labels` during learning only, not inside a prediction request |

During prediction, access each request through `item.request` in `batch.items`. During
learning, `context.training_examples` contains each supplied training `example.request`
alongside its observed labels in `example.labels`, mapping attribute IDs to observed values. The
harness does not reserve a validation split; `dev-run` uses every record in the supplied package.

Public specifications include the attribute name, definition, category profile, allowed values,
and whether the attribute is multi-valued. Missing labels are unannotated, not negative
examples; Q excludes those unannotated pairs. Q uses exact observed-value matching, not a
semantic or teacher-generated grade. See [Get experiment data](#get-experiment-data) for the
stored JSON fields and their types.

#### 2. Initialize state and keep the lifecycle intact

Keep the starter's method signatures. Decide which state your approach learns: for example,
a system prompt, per-attribute guidance, or review memory. Set its starting values in
`initialize`, and retain the injected model as `self.model = baseline.model`.

| Method | Called | You do |
| --- | --- | --- |
| `initialize(baseline)` | Once | Set up state |
| `evolve(context)` | Repeatedly while learning | Learn with metered usage; return `EvolutionResult(changed, stop, notes)` |
| `predict(batch)` | During evaluation | Return one identified prediction for every product in the batch |
| `save(path)` | After each successful `evolve` call | Persist the full agent state for the next checkpoint |
| `load(path)` | When evaluating a saved checkpoint | Rebuild the agent from that directory alone |

Do not create model clients or read endpoint credentials inside your agent. Both examples
use the model injected by the harness; your own approach must do the same.

#### 3. Implement batch prediction

Prediction uses the mandatory batch contract (`challenge_version: "2.0"`), including for one
product. `PredictionBatch.items` contains `PredictionItem(item_id, request)` entries. Return
`PredictionBatchResponse` with exactly one `PredictionResult(item_id, response)` for each input;
never substitute a product ID for the batch item ID or drop a product with missing answers.

Each per-item `PredictionResponse.attributes` maps requested attribute IDs to lists of predicted
values, for example `{"A100970": ["Puffs"]}`. Return an empty value list for an unknown answer.
Missing, duplicate or unexpected result IDs fail the batch mapping check.

For a prompt-based approach, initialize `self.prompt` in step 2 and forward the batch to
`self.model`. The model adapter shares identical attribute definitions across products and
sends the supplied batch in one measured call:

```python
from datathon.contract import BatchRules, PredictionBatch, PredictionBatchResponse

def predict(self, batch: PredictionBatch) -> PredictionBatchResponse:
   return self.model.predict(
      batch,
      BatchRules(system_prompt=self.prompt),
   )
```

To add product-specific guidance, supply `BatchRules.item_context` keyed by the current
batch's item IDs. Build that mapping from your learned state for each call; do not persist
batch IDs as learned identities. If needed, set an output limit with
`BatchRules(system_prompt, max_output_tokens=1024)`; otherwise the service default applies.

Organizer model policy can also set `max_output_tokens` for a base or teacher. A positive
integer is forwarded as the API limit; null or omission means the harness sends no limit
parameter, with no hidden numeric fallback. An explicit per-call `BatchRules` limit takes
precedence for base predictions. Service-side limits remain. This null-teacher behavior is
supported by 0.2.14; older activated releases may retain different token-limit behavior.

#### 4. Implement a learning step

Replace the starter's no-op `evolve`. Use `context.training_examples` and their observed labels
to decide what to improve. If you reserve validation products, leave them out of the data passed
to the learning run.
Use the training examples and public specifications to design the learning policy. `datathon dev-run`
trains and records learning usage; it does not calculate Q_dev or score checkpoints. To compare
prediction quality, reserve products from the participant data and evaluate them with your own local
validation code. Keep those reserved examples out of the examples passed to `evolve`. This local
metric is for development only; the organizer calculates official Q once on hold-out data.
For example, a prompt-only approach revises its prompt; a review-memory approach stores training
mistakes and derives guidance. These are design choices, not separate data sources.

For live prompt learning, call `context.prompt_teacher.complete(prompt)` through the supplied
metered teacher. Local runs supply it when enabled with `--model-config`; ordinary mock runs
do not. Use `context.learning_usage.spent` to inspect recorded learning credits. There is no credit cap. Return
`EvolutionResult(changed=True, ...)` when you update learned state, and `stop=True` when you
choose to finish. Do not implement the harness's scoring or evaluation loop inside the agent.

#### 5. Save and restore every learned change

Extend `save` and `load` together whenever you add learned state. A prompt-based agent must
persist its prompt; review memory, guidance and other state must also survive a fresh load.
`load` must rebuild the agent from the checkpoint directory alone, without training data or
an existing in-memory agent.

Retain the starter's `prediction-contract.json` marker and save/load checks. Old single-product
artifacts are incompatible; create a new learning run instead of loading them. The harness
reattaches its model after `load`; do not serialize a model client or credentials.

#### 6. Choose batching settings

The only batching setting is in your `challenge.yaml`:

```yaml
inference:
  batch_size: 1
```

It defaults to one product per evaluation call. Choose a larger positive integer when your
agent supports it. The harness delivers products in dataset order; it does not group, pack,
truncate, or automatically split requests. Your agent owns any further grouping, splitting,
retry strategy, and context/output sizing. All calls must still use the injected model so
usage can be measured. Model service limits and explicitly configured organizer runtime limits still apply.
There is no organizer batch-size cap or automatic concurrency; the model gateway does not
retry or silently fall back to single-product calls. Invalid or truncated output does not
erase incurred token usage.

Q remains micro-F1 over observed attribute values. E tokens are counted once per actual
model call, not multiplied by batch size. Batch timing and pass wall-clock are measured
separately; per-product latency is not invented by dividing a batch duration.

#### 7. Define stopping criteria

The harness records `stop_reason`. Step and runtime limits default to null (disabled);
explicit organizer limits take precedence over voluntary stopping. Local experiment
flags opt into rehearsal limits only; they do not change official settings. With no
limits, learning can incur unbounded usage until the agent stops, fails, exhausts its
training data, or the run is cancelled.

| Criterion | Controlled by | Recorded stop reason |
| --- | --- | --- |
| Learning reaches the maximum step count | Organizer | `max_steps` |
| Total runtime or a learning step reaches its time limit | Organizer | `runtime_limit` |
| `evolve` returns `EvolutionResult(..., stop=True)` | Your agent | `voluntary_stop` |

To finish voluntarily, return `stop=True` from `evolve`. Set `changed=True` as well if that
step produced learned state that should be saved. If multiple criteria apply, the recorded
reason follows the harness's priority: configured steps, configured runtime, then voluntary stop. Check
that reason in the experiment report rather than inferring it
from how many steps ran.

Your agent may return `EvolutionResult(..., best_checkpoint_id="C002")` to choose an eligible
checkpoint from its own attempt. C000 is the initial checkpoint; the harness saves one checkpoint
after each accepted learning step. The choice may name the checkpoint being saved or an earlier
one. A later `None` preserves the recorded choice; without any choice, the last checkpoint is used.
An explicit selection overrides the agent choice. Invalid choices fail, never silently fall back.
The harness does not rank checkpoints or treat participant validation as official Q. It verifies
eligibility and artifacts and reads the selected checkpoint's historical L from the ledger.
The pinned 0.2.14 contract supports this field. Official jobs must use the matching activated
release; a local package upgrade does not activate official jobs.
Learning must remain unattended; no step may wait for human input.

#### 8. Validate and compare a new experiment

Gateway failures are not converted to empty predictions or retried by the harness adapter.
The shared client logs `Gateway call failed` with safe structured details, then re-raises the
original exception and its cause chain. This applies to arbitrary exception classes, not just
timeouts or a fixed provider-code list. Failed output/usage parsing retains its original cause.
Token usage already returned by the service remains accounted for when parsing fails.

When supplied by the provider/SDK, details include HTTP status, error code, request ID, and
retry-after seconds. Raw messages, prompts, credentials, headers, and response bodies are not
copied into this structured log. Unsafe/unsupported metadata and diagnostic truncation are
explicitly marked; the raised exception itself is not truncated. Missing HTTP metadata is not
guessed. Keep exception propagation intact in your agent rather than silently accepting a failure.

Learning `failure.json` retains its existing `error` field and adds an `exception` graph.
Official worker failures raise `OfficialRunError` in the parent with `.exception_details` for
the remote graph; arbitrary exception objects are not serialized across the worker boundary.
Shared diagnostics are included in 0.2.14. Official job behavior follows the release the
organizers have activated.
They do not change SDK retry settings, limits, metering, or the one-pass scoring rule.

The harness does not score or visualize learning checkpoints. It saves and verifies their
artifacts, enforces hard limits, and meters controlled model usage and learning cost. The
participant may implement any training-data validation, prediction metrics, or visualization
inside its own agent/workflow and decide when to run it. Model-backed checks through injected
clients incur metered usage; Python-only checks do not incur model-token charges. No participant
metric replaces the organizer's one-pass final score, and hold-out data is never available to
participant code.

Run the interface checks after changing your agent:

```sh
datathon validate .
```

Then follow [Run an experiment on your laptop](#run-an-experiment-on-your-laptop) with a new
output directory. Check the recorded stop reason, actual model calls/tokens, and checkpoint
learning cost. `dev-run` does not report prediction quality; evaluate a participant-reserved
validation set with your own code. Confirm that learned state survives save/load before making
another change. Contract validation is not a quality evaluation or an official submission.

#### 9. Optional team-owned learning traces

Teams are encouraged to record **what the agent examined and changed**, but this is not
an evaluation-harness responsibility or an official performance score. The starter provides
`ParticipantAgent.log_learning_trace` as an optional example helper; the harness never calls
it, reads its records, or uses them for stopping, selection, Q/E/L, or leaderboard ranking.
It is a participant utility, not a harness API or a required agent-contract method.

Call it inside your own `evolve` only when useful, using values already produced by your
learning logic. For example, with your iteration counter, selected training examples, and
before/after state snapshots already available:

```python
from pathlib import Path

self.log_learning_trace(
   Path(".datathon/learning-trace.jsonl"),
   iteration=learning_iteration,
   examined=[example.request.product_id for example in selected_examples],
   changed=changed,
   notes="Updated state from selected training examples",
   state_before=state_before,
   state_after=state_after,
   observations={
      "predictions": None,
      "learning_credits_seen": context.learning_usage.spent,
   },
)
```

The helper appends JSONL records with a timestamp and an explicit `participant-reported`
source. It does not initialize an agent, call a model, predict, grade, or create before/after
scores. Include predictions or teacher feedback in `observations` only if your learning
step already produced them; otherwise keep them null/not measured. The observed usage
value is a snapshot during the step, not a replacement for the harness's completed-step L.
Writing the trace adds only local I/O; when done inside `evolve`, that time is part of
learning compute. Passing `None` instead of a path disables the helper without creating files.

Use the ignored `.datathon/` folder or another private location, never the checkpoint
directory passed to `save`. Traces may contain training data, labels, prompts, and feedback;
do not commit them, include them in notebook outputs, or log credentials or hold-out data.
State and observations must be JSON-serializable with finite numeric values. The helper
surfaces serialization/I/O errors to your code; teams own retention and logging behavior.
Trace files are not needed by `load` and must not affect restored predictions.

For inspection, read the JSONL records into a local notebook table or generate a local
before/after text diff. That needs no LLM. A learning trace explains a change; it does not
prove that the change improved quality on unseen products. Optional validation and the
final frozen-agent evaluation remain separate.

### Get experiment data

**Experiment data** is the participant-accessible product-attribute dataset for learning and
local experiments. Both agent examples use this same package. **Hold-out data** is separate,
organizer-only data for official scoring; it is never included in the participant package.

For a local data folder, the only required product-data files are
`product_attribute_experiment.jsonl` and `dataset.json`. Each JSONL line is one record:

| Field | Stored JSON shape |
| --- | --- |
| `run_id` | Nonempty string identifying the source run |
| `product_id` | Nonempty string identifying the product |
| `product_description` | Product description string |
| `attributes` | Array of public attribute-specification objects |
| `labels` | Array of observed-label objects, each with `attribute_id` and `values` (a list of strings) |

Each public attribute object contains `attribute_id`, `attribute_name`, `category_profile`,
`definition`, `multi_valued`, and `allowed_values` (an array of allowed-value objects).
The parser requires the attribute ID and name; specification text defaults to empty strings,
`multi_valued` to false, and `allowed_values` to an empty array when absent.
`dataset.json` supplies the dataset `name` and announced `version`; it is metadata, not another
training-data source.

The harness maps `product_description` to `request.input_text`, attribute IDs to
`request.attributes`, and public specifications to `request.attribute_specs`. During learning,
it converts the stored `labels` array into `example.labels`, a mapping from attribute ID to
observed value tuples in `context.training_examples`. Prediction requests never contain labels.
Unlabeled attributes remain in `attributes` and are absent from `example.labels`; do not turn
them into negative examples.

No additional product, attribute-definition, or label files are required. The starter includes
the published train-only package at `data/synth_v6_full`, so you can run the examples without
downloading data first. The package contains no hold-out records and may be committed in your
team repository so teammates use the same version. The package has no automatic validation
split: `dev-run` uses every record in it for learning and does not score checkpoints. For local
quality checks, create a separate validation subset and leave it out of the learning data. The
public model-policy YAML below is configuration, not data. Bundled mock-task sample data is an
optional smoke-check fallback.

When organizers announce a newer dataset version, download it from your team repository root
into `data/<version>`. `GITLAB_TOKEN` must be set as described in Part 1; never commit the token.
The versioned folder should contain exactly `product_attribute_experiment.jsonl` and `dataset.json`:

```sh
export DATATHON_DATA_VERSION="<new announced dataset version>"
export DATATHON_DATA_DIR="data/${DATATHON_DATA_VERSION}"
mkdir -p "$DATATHON_DATA_DIR"
for f in product_attribute_experiment.jsonl dataset.json; do
   curl -f -o "$DATATHON_DATA_DIR/$f" -H "PRIVATE-TOKEN: ${GITLAB_TOKEN}" "https://gitlab.com/api/v4/projects/mlmodels%2Ftamu-2026-shared%2Ftamu-2026-ci/packages/generic/experiment-data/${DATATHON_DATA_VERSION}/$f"
done
ls "$DATATHON_DATA_DIR"
```

### Run an experiment on your laptop

From the repository root, with a new output folder each time:

```sh
datathon dev-run . --output .datathon/run-1 --data "$DATATHON_DATA_DIR"
datathon inspect --db .datathon/run-1
```

For the package included in the starter, use `--data data/synth_v6_full` instead.

Without `--model-config`, this is a mock-model rehearsal even when you supply experiment data.
It checks your lifecycle and data handling, not the quality of a live base model or teacher.

Open `.datathon/run-1/trajectory.txt` to see checkpoint IDs, cumulative learning cost (`L`),
and the stop reason. The harness does not produce checkpoint Q_dev or development inference E.

For live product-attribute prompt experiments, install the pinned Databricks extra from the same
registry, using the same index options as above with `datathon[databricks]==0.2.14`. The public
`model-policy.yaml` shipped in this starter contains organizer-approved model IDs, token
rates/limits, and scoring constants. It deliberately omits service endpoints; local base and
teacher Gateway services belong in `challenge.yaml`. Do not edit the allowed IDs or prices, and
never add credentials. If the organizers publish a replacement policy, review and update this
file from their source.

Authenticate to your workspace using standard Databricks authentication, such as a configured
`DATABRICKS_CONFIG_PROFILE`. Provision or select services in your own workspace for the allowed
base model and teacher, then set their IDs and local service names under `models` in
`challenge.yaml` as shown in that file. With multiple allowed base models, select one explicitly.
For example, replace these placeholders with IDs from the public policy and services in your
own workspace:

```yaml
models:
   base:
      id: "<allowed base model ID>"
      service: "<catalog>.<schema>.<base_service>"
   teacher:
      id: "<allowed teacher model ID>"
      service: "<catalog>.<schema>.<teacher_service>"
```

   The public policy may allow multiple teachers; if you omit the teacher selection, its default
   teacher is used. A model may be allowed in both roles. Model eligibility is checked by ID,
   not by local service name; the harness does not inspect the endpoint's underlying model.
   For a Responses API route, set `api: responses` beside that role's `id` and `service`.
   For example, an allowed Qwen route can use `service: system.ai.qwen3-next-80b-a3b-instruct`.
   The default API is `chat`. Official runs ignore local service/API overrides and use the
   released organizer configuration. These options require a harness release supporting
   teacher catalogues and Responses. Changing your manifest does not upgrade the installed
   package or official jobs.

   If the endpoint rejects JSON mode, set `json_mode: false` beside the local base selection.
   This omits the API JSON-format option, not the JSON instructions or strict response parsing.
   The default is true; official runs use the released JSON-mode setting, not a local override.

Your identity must have access to both selected services. Do not assume that the organizer's
service names are accessible from your account. Live calls incur account charges; this remains
a development experiment, not an official submission.

```sh
datathon dev-run . --data data/synth_v6_full --model-config model-policy.yaml \
   --output .datathon/prompt-1
datathon inspect --db .datathon/prompt-1
```

To read experiment data already loaded into tables in your own workspace instead of a downloaded
folder, set `DATATHON_EXPERIMENT_DATA_WAREHOUSE` to your SQL warehouse ID and replace
`<catalog>.<schema>` with the schema containing those tables:

```sh
datathon dev-run . --shared-schema "<catalog>.<schema>" \
   --warehouse "$DATATHON_EXPERIMENT_DATA_WAREHOUSE" --model-config model-policy.yaml \
   --output .datathon/shared-prompt-1
```

This uses your workspace login to read `product_attribute_experiment` and its dataset manifest
directly; it writes no local data package and never reads the hold-out table. `--shared-schema`
and `--data` are alternative sources, not combined inputs. Both produce the same validated
product-attribute records and use the same learning and scoring paths. A failed shared query
stops the run; it does not silently fall back to another source.

The organizer's `shared.task` is not automatically shared across accounts. If you adapt a
prompt-agent example notebook, set `DATATHON_EXPERIMENT_DATA_SCHEMA` to your own data schema,
or to an empty string to use `DATATHON_EXPERIMENT_DATA` instead. Keep experiment tables
unchanged between inspection and learning, and keep data-bearing notebook outputs local.

The commands above use participant experiment data for the learning loop and do not request
checkpoint scoring. The harness meters teacher calls, verifies checkpoint persistence, and records
learning cost without a credit cap. To opt into local limits, add `--max-steps 3`,
`--max-runtime-seconds 900`, or `--step-timeout-seconds 120`; omitted options are disabled.
These do not change official settings. A participant may reserve its own validation set and implement a metric in participant code;
the harness does not create that split or score those checkpoints. Official hold-out data is only
read by the organizer's final evaluator.

### Run an experiment in your Databricks workspace

The supplied `workspace/experiment.py` helper runs a mock-model experiment, not the live prompt
workflow above. It writes protocol tables and artifacts only to your own experiment schema,
whose name must start with `experiment`; it creates no official submission or score.

In your Git folder, open the helper and fill in `DATATHON_EXPERIMENT_CATALOG`,
`DATATHON_EXPERIMENT_SCHEMA`, `DATATHON_EXPERIMENT_VOLUME`, `DATATHON_EXPERIMENT_WAREHOUSE`,
and `DATATHON_WHEEL_VOLUME` using resources you manage in your account. Stage the same 0.2.14
harness release and its dependencies in that wheel folder: the helper installs the available
wheel offline with `--no-index`, not the GitLab pin from `requirements.txt`. A bare harness
wheel is not enough. No secret name or organizer credential is required. Run the cells in order.

To use experiment data, upload `product_attribute_experiment.jsonl` and `dataset.json` to a Volume in
your workspace and set `DATATHON_EXPERIMENT_DATA` to its folder, for example
`/Volumes/<catalog>/<schema>/<volume>/experiment-data`; without it the helper uses sample data.
For live model-backed experiments, use the `dev-run --model-config` workflow with your own
service routing and authentication rather than assuming this helper enables live models.

## Part 3. Submit for scoring

You do not start official runs; the organizer runs the commit you name. Each submission uses one
Git commit, and the active organizer release sets the submission limit. Ask the organizers for
the event limit before planning multiple submissions.

### Official compatibility

Use only the harness-injected base model and metered teacher. Your agent must not create
model clients, depend on local credentials, or call workspace-specific APIs itself. Select
allowed model IDs in `challenge.yaml`. Local `service`, `api`, and `json_mode` settings route
development calls only; official learning and scoring preserve the selected IDs but use
the organizer's released routes and protocol settings. JSON/batch handling should work
through the injected model regardless of whether your local route uses Chat or Responses.

The official job uses its immutable harness release, not the package installed on your
laptop. Learning usage is measured as L without credit caps. Runtime/step limits default
to null and apply only when the organizer explicitly configures them. Local experiment
flags cannot override the official policy.

Official learning builds fresh checkpoints from the pushed commit; a locally learned
checkpoint is not submitted. Save every learned value and preserve the `batch-v1` marker.
The harness reattaches the model after `load`. Each input batch item ID must appear exactly
once in the returned results.

The automated official workflow uses an explicit selection, then the agent's recorded
`best_checkpoint_id`, then the last eligible checkpoint. A local development checkpoint is not
eligible for an official attempt. Small development runs and contract
checks do not prove full-dataset runtime or live permissions; those are organizer readiness
checks before the official attempt.

### Submission checklist

- [ ] `evolve` records the checkpoint you want scored, or stops with the desired last checkpoint.
- [ ] `save` writes all learned state and `load` rebuilds the agent from that directory alone.
- [ ] `datathon validate .` passes, you pushed, and the GitLab pipeline is green. The official run
  uses only pushed code.
- [ ] No tokens or hold-out data are in the repository. The train-only experiment package, if used, is under `data/<version>`.
- [ ] You sent the organizer the full commit SHA (`git rev-parse HEAD`), not a branch name.

## Results

For each scored submission you get a `team-result.json` with Q, E, L, the baseline and frontier
references and derived metrics (`delta_q`, `gap_closed`, `learning_efficiency`). The leaderboard is
in the GitLab project `tamu-2026-leaderboard` (read access came with your invitation). If
evaluation fails, you get a `FAILED` status and no score; tell the organizer.

## Rules

- **Official runs:** use only the teacher and base model the harness gives you; calls that bypass
  them are not allowed. Do not read hold-out data, grade yourself or bypass the meter.
- **Models and prices:** the organizers define allowed model IDs and credit prices. Select allowed
   IDs under `models.base.id` and `models.teacher.id` in `challenge.yaml`. For local `dev-run`, you
   may set each `service` to a Gateway service available in your workspace. These service names only
   route local experiment calls; they do not change model IDs or prices. Official runs ignore these
   service overrides and use the released model policy. A model ID not on the organizer's list is
   refused.
- **Your Databricks workspace:** provision or use model services there for local experiments. Those
   local service calls are not used for official scoring or charged to the event.
- **Data:** commit only the published train-only experiment package under `data/<version>`; never add hold-out data or credentials.
- **Safety:** local runs import your agent in-process and are not a sandbox; run only code you trust.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `datathon` has no `validate` command | Reinstall with `--index-strategy first-index`. |
| `File exists` error | Use a new `--output` folder. |
| CI fails at `agent imports` | Make `challenge.yaml` match the class in `src/agent.py`. |
| Run ends after one step with `voluntary_stop` | Expected for the no-op starter; implement `evolve`. |
| `load` fails | `save` must write all state, and `load` must rebuild from that directory alone. |
| `Incompatible artifact: batch-v1 checkpoint required` | Start a new learning run with the 2.0 manifest; keep the starter's checkpoint marker checks. |
| Batch result IDs do not match | Return every input item ID exactly once; do not use product IDs as batch IDs. |
| Invalid or truncated model output | Reduce your batch or context, or choose an adequate `max_output_tokens`; incurred calls still cost tokens. |
| `--data` is refused | Experiment data needs `product_attribute_experiment.jsonl` and `dataset.json`; optional synthetic sample CSV data uses `experiment.csv` and `dataset.json`. Keep product IDs unique and use the announced dataset version. |
| A configured model service is inaccessible | Check your own workspace login, local `models` service overrides and service permissions; do not use organizer credentials. |
| Workspace helper cannot install the harness | Put the pinned harness and all required dependencies in `DATATHON_WHEEL_VOLUME`; it installs with `--no-index`. |
| No invitation, or project not visible | Your GitLab email differs from the one the organizer listed. Tell the organizer. |
| Registry or permission error | Ask the organizer; never paste tokens anywhere in the repository. |
