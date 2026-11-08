# Brown Ballers solution

## What was submitted

- Code: `src/agent.py` and `src/learner.py`
- Saved agent state: `submission/agent-state/skill-bank.json`
- Hold-out predictions: `submission/predictions.jsonl`
- Scoring output: `submission/result.json`
- This note, plus `submission/evidence.json`

## How to rerun

From the team repository, with the shared scorer cloned beside it.

Learn from the experiment file, freeze the bank, then score. This rewrites the saved state:

```powershell
python scripts/build_submission.py
```

Score the already frozen bank with learning left off:

```powershell
python scripts/predict_frozen.py
```

`predict_frozen.py` only reads `submission/agent-state/skill-bank.json`. It does not fit weights, choose packs, or read hold-out labels. The organizer scorer is the only step that opens those labels.

## Approach

Each product attribute gets its own skill. A skill is an averaged perceptron over tokens from the product fields. Hand-written detectors for packs, grains, sweeteners, slicing, soda names, and cereal-form wording only add features. Corn syrup is not treated as a corn grain. A token or name fragment has to appear in at least two training products before it can receive a weight, so one product cannot be memorized by its spelling. The experiment labels decide the weight of each feature, which feature pack to keep, whether two of three packs must agree, and how large a margin is required before a value is emitted.

The choice uses a product-disjoint 20% of the experiment data. A reader also splits the ingredient list from the product name. A high-precision reading becomes an override only when it raises that split's exact-match micro-F1, and only for the attribute the reading describes. No foundation-model weights are updated. The saved state is the skill bank. `evolve` and the build script call the same `learn` function. Before any hold-out line is read, that function checks that a reloaded bank predicts the same values and that prediction does not change the weights.

The number in the table is that split, with weights that never saw those products. The submitted bank then relearns the same packs on all 965 experiment products and calibrates margins again on the split, using the same product reading as prediction. That second pass is not a fair held-out score, so it is not the learning evidence. Hold-out products are not in the experiment file. Hold-out labels are not read until the official scorer runs, and they are not used to choose skills. Learning took 126.4 seconds. Prediction does not learn.

## Learning evidence

Same 202 validation products, same micro-F1:

| Agent | Q |
| --- | ---: |
| Empty predictions | 0.0000 |
| Literal allowed-value copy, before learning | 0.4662 |
| Learned skills, trained without these products | 0.8498 |

Learned counts: TP 614, FP 98, FN 119.

The literal copy is a manual baseline. The perceptron weights, the pack choice, the vote, the margins, and any reader override are the automated update. Reader overrides kept this round: [{"attribute_id": "A900201", "feature": "read:sweet_plant", "value": "Plant-Based Sweetener"}].

What did not earn a place in the bank: a nearest-example memory, about 0.70 on this split; dropping name fragments entirely, which fell to 0.838; ingredient-order tokens with a finer margin grid, which fell to about 0.844. One reader override did earn a place: a plant sweetener in the ingredient list is emitted as Plant-Based Sweetener. The remaining experiment misses are ambiguous descriptions: a grain named in the label but not in the text, corn syrup versus the word sugar, fruit mixed in versus fruit flavored, and citrus versus fruit soda.

## Hold-out result

Q = 0.824412

TP 1209, FP 243, FN 272, validity 1.0, missing products 0 of 415. Dataset `synth_v6_full`, scorer `holdout-eval-q-v1`. This score is self-reported.
