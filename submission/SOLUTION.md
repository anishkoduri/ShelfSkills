# Brown Ballers solution

## What was submitted

- Final code: `src/agent.py` and `src/learner.py`
- Saved agent state: `submission/agent-state/skill-bank.json`
- Run instructions: this note
- Hold-out predictions: `submission/predictions.jsonl`, one line for every product
- Scoring-utility output: `submission/result.json`
- Learning evidence: `submission/evidence.json` and the table below

## How to rerun

From the team repository, with `tamu-2026-ci` cloned beside it. The scorer is imported from that repository and is not modified.

Learn from the experiment file, freeze the bank, write predictions, and score. This rewrites the saved state:

```powershell
python scripts/build_submission.py
```

Score the saved bank again with learning left off:

```powershell
python scripts/predict_frozen.py
```

`predict_frozen.py` loads `submission/agent-state/skill-bank.json` and does not fit weights or choose packs. Each record has its `labels` field removed before `predict_row`. The organizer scorer then reads the original file and the predictions.

## Approach

Each requested attribute has its own averaged perceptron. The features are tokens from the product name, description, and ingredients, plus hand-written detectors for packs, grains, sweeteners, slicing, soda names, and cereal form. Corn syrup and corn starch are sweeteners, not a corn grain. A token has to appear in at least two training products before it can receive a weight.

The experiment file is split by product id. A product is in the validation split when the first eight hexadecimal characters of MD5(product id), read as an integer, leave a remainder of 0 on division by 5. That is about one product in five. The other products train the weights. On the validation products the learner chooses, for each attribute, the feature pack (`short`, `long`, or `cue`), whether two of the three packs must agree, the margin required before a value is emitted, and whether an earlier epoch is more exact than the final pass. An earlier pass is kept only when validation micro-F1 rises. Passes kept this run: Soda Type kept pass 1, Soda Feature kept pass 4, Yogurt Type kept pass 1, Cereal Source kept pass 4, Flavor Content kept pass 3, Roll Form kept pass 3, Yogurt Fat Content kept pass 2, Container Format kept pass 3.

Candidate readings are hand-written. Keeping one is automatic, and a reading that fails its test is left out. A rule mined from training mistakes is kept only when validation micro-F1 rises. A product-name phrase is kept only when that attribute's validation micro-F1 rises. The diet, seed, and garlic candidates need at least nine in ten of their training rows, and are kept only when validation micro-F1 rises. The mix-in and probiotic candidates need every training row, and are kept only when validation micro-F1 rises. The fruit-piece candidate is kept only when validation micro-F1 rises. The remaining candidates are kept when the training rows that show the cue match the reading exactly, whether that means the label is present, the label is the whole set, or a named label is absent, and validation micro-F1 does not fall. The kept readings are listed below. The exact records are in `submission/evidence.json`.

- Stevia or monk fruit in the ingredients emits Plant-Based Sweetener.
- Diet in the soda name, or an artificial or plant sweetener in the ingredients, adds Diet and removes Cane & Real Sugar.
- A seeded loaf name, or flax, caraway, or chia seeds in the ingredients, adds Seeds.
- Garlic in the product name adds Garlic.
- A top-level fruit piece changes Fruit Flavored to Fruit Mixed In.
- Fruit pieces, dried fruit, or fruit on the bottom add a Fruit mix-in.
- The word probiotic adds Probiotic & Functional.
- Low or reduced fat, or 1% or 2% milk, sets Low & Reduced Fat. Whole milk and “1% or less” do not count.
- The word sourdough in the loaf name sets Sourdough.
- The word brioche in the roll name sets Brioche.
- The token oat in the bakery name adds Oats. Oatmeal is a different token.
- A fruit in the soda name sets Fruit. Spice, herbs, and ginger ale do not.
- A bottom-layer phrase, including a hyphenated on-the-bottom, sets Fruit on the Bottom.
- The shelf token zs adds Diet. The word zero by itself does not.
- The word drinkable in the product name adds Drinkable.
- The word french in the product name adds French.
- Nonfat or fat-free in the product name sets Fat Free.
- A single real grain, oat, in the ingredient head sets Oat and removes Multiple.
- A single real grain, corn, in the ingredient head sets Corn and removes Multiple. Syrup and starch do not count.
- A single real grain, wheat, in the ingredient head sets Wheat and removes Multiple.
- A single real grain, rice, in the ingredient head sets Rice and removes Multiple.
- Caffeine in the ingredient list, when the list does not say caffeine free, removes Caffeine Free.
- When the name, ingredients, and description have no granola word, Granola is removed.
- Aspartame, sucralose, acesulfame, or saccharin in the ingredients sets Artificial Sweetener, unless stevia or monk fruit is also present.
- The name token wht sets Wheat, unless the name also says rice.
- Multigrain, including the shelf token multgr, sets Multiple.
- The name phrase “thin sliced” emits Thin Sliced.
- The name phrase “tube” emits Tube.
- The name phrase “bottle” emits Bottle.
- The name phrase “coconut” emits Coconut.

`evolve` and this script call the same `learn` function. No foundation-model weights are updated. The saved state is the skill bank. Before the prediction file is opened, `learn` checks that a reloaded bank predicts the same values and that prediction does not change the weights. `predict` does not call `learn`.

The table below is the validation split, with weights that did not see those products. The saved bank then refits the same packs on all 965 experiment products and calibrates margins again on the validation split. That refit is the submitted agent. It is not the learning-evidence number. Learning took 216.3 seconds.

## Learning evidence

Same 202 validation products, same exact-match micro-F1. The learned row uses weights fit only on the other experiment products.

| Agent | What it is | Q |
| --- | --- | ---: |
| Empty predictions | Manual baseline, no values | 0.0000 |
| Literal allowed-value copy | Manual baseline, before any weights | 0.4662 |
| Learned skills | Automated weights, packs, margins, epochs, and kept readings | 0.8909 |

Learned counts: TP 649, FP 75, FN 84. Literal counts: TP 317, FP 310, FN 416.

What is automated: the perceptron weights, the pack, the vote, the margin, the epoch, and the keep-or-drop decision for each candidate reading. What is manual: the feature detectors and the wording of the candidate readings. A fixed copy of an allowed value, with no weights, is the before side of the comparison.

Compared on the experiment validation split in earlier runs, and not kept: a nearest-example memory, about 0.70; dropping name fragments, 0.838; ingredient-order tokens with a finer margin grid, about 0.844; applying every reliable name phrase at once, about 0.79. Margin-bin abstention, a category filter, token rules mined from training mistakes, and filling an abstention from similar products did not raise that split. Treating a second grain mention as a training feature for Multiple lowered an earlier result from 0.8660 to 0.8638, so that feature was not added to the perceptron. A name phrase is kept only for the attributes where the split rose.

The remaining misses on the validation products are ambiguous text. A second grain is sometimes the label Multiple and sometimes the grains named one by one. A sweetener is not always named in the ingredients. A fruit piece is sometimes mixed in and sometimes only a flavor. Culture names are not a reliable probiotic. Those cases stay with the perceptron.

## Hold-out result

Scored with the unmodified `holdout_eval` utility, `scorer_version` `holdout-eval-q-v1`.

Q = 0.854712

TP 1256, FP 202, FN 225. Validity 1.0. Missing products 0 of 415. Dataset `synth_v6_full`. Team `team-brown-ballers`. This score is self-reported.
