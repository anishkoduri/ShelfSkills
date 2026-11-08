# Brown Ballers solution

The work was to read messy grocery text and return exact attribute strings, and to show that the agent improved by updating its own skills. This note is that account: the frame, the form it took, the decisions, and what the experiment split taught. Hold-out Q is 65% of the rubric and is reported at the end. The other 35% is the before-and-after on experiment data. Performance is the hold-out number. The experiment comparison is the learning evidence.

## Approach

**Frame.** A product arrives as a short name, a description, and an ingredient list that does not always name the attribute the label cares about. Each requested attribute has a closed list of allowed strings, and a value counts only when it matches the label character for character. Q is micro-F1 over those annotated values. An extra wrong string is a false positive and still leaves the true value missing, so one bad guess is charged twice. Leaving the attribute empty costs only the miss. The skill worth learning is restraint: emit a string when the text supports that exact value, and stay quiet when it does not. Attributes with no label are ignored. Cost, latency, and the leaderboard composite are outside this score.

**Materialize.** That frame became a skill bank. `ParticipantAgent.evolve` calls `learn` on the experiment file. For each attribute, `learn` fits an averaged perceptron on tokens from the name, the description, and the ingredients. A product-disjoint slice of the same file is held back. On that slice the learner chooses which text view to trust, how large a margin to require before a value is emitted, whether an earlier pass of the weights was more exact, and which candidate readings survive. The saved bank is that result. `predict` loads the bank and does not change it. Foundation-model parameters are not trained. The feature detectors and the wording of the candidate readings were written by hand. The experiment labels decide the weights, and they decide which of those readings stay.

## Key decisions

1. **One skill per attribute.** A wheat loaf and a wheat cereal are different decisions written in the same kind of text. A shared weight vector would smear them. Each attribute keeps its own perceptron.
2. **A token must be seen twice.** Shelf copy is full of one-off spellings. A weight on a token from a single product would memorize that product. Tokens below that support never receive a weight.
3. **Let the split choose the view.** The name, the long description, and the ingredient cues disagree. Some attributes live in the name. Some live in the ingredient head. Each attribute may use the short text, the long text, a cue view, or a vote that keeps a value only when two of the three packs agree. A choice is kept when validation micro-F1 rises. An earlier training pass is kept on the same test, because more passes can start fitting noise. This run kept: Soda Type pass 1, Soda Feature pass 4, Yogurt Type pass 1, Cereal Source pass 4, Flavor Content pass 3, Roll Form pass 3, Yogurt Fat Content pass 2, Container Format pass 3.
4. **Corn syrup is a sweetener.** The word corn in this catalog is usually syrup or starch. Counting it as a grain pulled sweetened products toward Multiple. The ingredient head also stops before the "contains 2%" tail, where vitamins and trace mentions are listed and are not the source.
5. **Readings are proposals.** A person can notice that "probiotic" or "nonfat" is a real shelf phrase. The split decides whether that phrase stays. A rule mined from training mistakes, and a product-name phrase, are kept only when validation micro-F1 rises. Diet, seed, and garlic need at least nine in ten of their training rows and a rise. Mix-in and probiotic need every training row and a rise. The fruit-piece reading is kept only when validation micro-F1 rises. The other readings are kept when the training rows that show the cue match exactly, whether the label is present, the label is the whole set, or a named label is absent, and validation micro-F1 does not fall. A reading that fails is absent from the bank.
6. **Two banks, two jobs.** The fair learning number uses weights that never saw the validation products. After that choice, the same packs are refit on all 965 experiment products and the margins are calibrated again, so the submitted agent has seen every experiment label. That refit is the agent that predicts. The evidence number is the earlier one.
7. **Learning ends when the bank is written.** The hold-out file is opened after the bank is saved. Each record is a prediction input with `labels` removed before `predict_row`. `predict` does not call `learn`. The organizer scorer is a separate program, and it is the step that reads labels.

The validation split is every product id whose first eight hexadecimal characters of MD5, read as an integer, leave a remainder of 0 on division by 5. That is about one product in five. Learning took 216.3 seconds. `evolve` and `scripts/build_submission.py` call the same `learn`. Before the prediction file is written, `learn` checks that a reloaded bank predicts the same values and that prediction does not change the weights.

## What the experiment taught

Same 202 validation products, same exact-match micro-F1. The learned row uses weights fit only on the other experiment products.

| Agent | What it is | Q |
| --- | --- | ---: |
| Empty predictions | Manual baseline, no values | 0.0000 |
| Literal allowed-value copy | Manual baseline, before any weights | 0.4662 |
| Learned skills | Automatic weights, packs, margins, epochs, and kept readings | 0.8909 |

Learned counts: TP 649, FP 75, FN 84. Literal counts: TP 317, FP 310, FN 416.

The learning claim is the move from 0.4662 to 0.8909 on those same products. An empty agent scores zero because every labelled value is missed. The literal copy emits an allowed string whenever that string is written in the text. That is a serious manual baseline, and it still emits strings that are only flavor copy, a grain named inside a sweetener, or a marketing phrase. False positives fall from 310 on that copy to 75 on the learned skills, because the skills can abstain. Performance is the hold-out Q at the end. These two experiment numbers are the before and the after.

Compared on the experiment validation split in earlier runs, and not kept: a nearest-example memory, about 0.70, which copied a neighbor instead of reading the product; dropping name fragments, 0.838, so the shelf title is signal; ingredient-order tokens with a finer margin grid, about 0.844; applying every reliable name phrase at once, about 0.79. Margin-bin abstention, a category filter, token rules mined from training mistakes, and filling an abstention from similar products did not raise that split. Treating a second grain mention as a training feature for Multiple lowered an earlier result from 0.8660 to 0.8638, so that feature was not added to the perceptron.

The split kept teaching the same lesson. A reading that is often right still loses when it fires on every product, because a wrong value costs more than silence. A second grain is sometimes the single label Multiple and sometimes the grains named one by one. A sweetener is not always written as sugar or as corn syrup. A fruit piece is sometimes mixed through the cup and sometimes only a flavor. A culture name can belong to a traditional yogurt. Those cases stay with the perceptron. A hard reading does not take them, because the split already showed that a confident wrong answer is worse than a blank.

Readings the split kept:

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

The list above is what the split kept. Each line is a general reading of product text, tied to one attribute. The records behind the lines are in `submission/evidence.json`.

## Pressure test

Checked against the written criteria before this note was finished.

| Criterion | Where this package meets it |
| --- | --- |
| Performance is hold-out Q, exact-match micro-F1. Cost and the composite are not the score. | `submission/result.json`, field `q`. Validity 1.0. Missing 0 of 415. |
| The organizer scorer is used unchanged, on every product. | `holdout_eval`, `scorer_version` `holdout-eval-q-v1`, from [tamu-2026-ci](https://gitlab.com/mlmodels/tamu-2026-shared/tamu-2026-ci/). One prediction line per product. |
| Learning updates skills, not model parameters. | Perceptron weights, pack, margin, epoch, and kept readings. No foundation-model update. |
| Before and after use the same experiment products and the same metric. | The table above. Empty, literal copy, and learned skills. |
| Manual work is separate from automatic work. Limits and failed experiments are stated. | Detectors and reading wording are manual. Weights and keep-or-drop are automatic. The rejected runs are listed above. |
| The starter agent interface is the one implemented. The full harness is optional. | `ParticipantAgent` in `src/agent.py`: `initialize`, `evolve`, `predict`, `save`, `load`. |
| The bank is final before inference, and inference does not learn. | `predict` does not call `learn`. `scripts/predict_frozen.py` only loads `submission/agent-state/skill-bank.json`. |
| Hold-out records are prediction inputs. The agent does not receive labels. | The bank is saved first. `labels` is removed before `predict_row`. The scorer reads the original file. |

## What was submitted

- Final code: `src/agent.py` and `src/learner.py`
- Saved agent state: `submission/agent-state/skill-bank.json`
- Run instructions: this note
- Hold-out predictions: `submission/predictions.jsonl`, one line for every product
- Scoring-utility output: `submission/result.json`
- Learning evidence: `submission/evidence.json` and the table above

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

## Hold-out result

This is the performance number. The bank was already frozen. The unmodified `holdout_eval` utility then scored every product. `scorer_version` `holdout-eval-q-v1`.

Q = 0.854712

TP 1256, FP 202, FN 225. Validity 1.0. Missing products 0 of 415. Dataset `synth_v6_full`. Team `team-brown-ballers`. This score is self-reported.
