"""Train on experiment data, then score the hold-out once with learning disabled.

Hold-out labels are read only by the organizer scoring utility. The agent state
is already frozen before that file is opened.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT.parent / "tamu-2026-ci"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CI))

from holdout_eval.scoring import score_files  # noqa: E402

from src.learner import learn  # noqa: E402

EXPERIMENT = ROOT / "data" / "synth_v6_full" / "product_attribute_experiment.jsonl"
HOLDOUT = CI / "data" / "holdout" / "holdout_private.jsonl"
SUBMISSION = ROOT / "submission"
TEAM = "team-brown-ballers"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    rows = load_jsonl(EXPERIMENT)
    print(f"experiment products: {len(rows)}", flush=True)
    started = time.perf_counter()
    _evidence_bank, final_bank, report = learn(rows)
    report["learning_seconds"] = round(time.perf_counter() - started, 1)
    print(
        f"validation learned Q={report['learned_q']:.4f} "
        f"literal Q={report['literal_q']:.4f} empty Q={report['empty_q']:.4f} "
        f"in {report['learning_seconds']}s",
        flush=True,
    )

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    state_path = SUBMISSION / "agent-state" / "skill-bank.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"learned": True, "bank": final_bank.to_dict()}, ensure_ascii=False),
        encoding="utf-8",
    )
    (state_path.parent / "prediction-contract.json").write_text(
        json.dumps({"contract": "batch-v1"}),
        encoding="utf-8",
    )

    # Frozen from here. The next read of the hold-out is for prediction inputs only.
    predictions_path = SUBMISSION / "predictions.jsonl"
    seen: set[tuple[str, str]] = set()
    with HOLDOUT.open(encoding="utf-8") as source, predictions_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("labels", None)
            key = (str(record["run_id"]), str(record["product_id"]))
            if key in seen:
                raise RuntimeError(f"Duplicate hold-out product {key}")
            seen.add(key)
            prediction = {
                "product_id": record["product_id"],
                "run_id": record["run_id"],
                "attributes": final_bank.predict_row(record),
            }
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    result_path = SUBMISSION / "result.json"
    if result_path.exists():
        result_path.unlink()
    result = score_files(HOLDOUT, predictions_path, team_id=TEAM)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report["holdout"] = {
        "q": result["q"],
        "true_positive": result["true_positive"],
        "false_positive": result["false_positive"],
        "false_negative": result["false_negative"],
        "validity_rate": result["validity_rate"],
        "missing_products": result["missing_products"],
        "holdout_products": result["holdout_products"],
        "dataset_version": result["dataset_version"],
        "scorer_version": result["scorer_version"],
    }
    (SUBMISSION / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_explanation(report, result)
    print(json.dumps(report["holdout"], indent=2), flush=True)


READINGS = {
    "read:sweet_plant": "Stevia or monk fruit in the ingredients emits Plant-Based Sweetener.",
    "read:soda_diet_or_sweetener": "Diet in the soda name, or an artificial or plant sweetener in the ingredients, adds Diet and removes Cane & Real Sugar.",
    "read:bakery_seed": "A seeded loaf name, or flax, caraway, or chia seeds in the ingredients, adds Seeds.",
    "read:bakery_garlic": "Garlic in the product name adds Garlic.",
    "read:top_fruit_piece": "A top-level fruit piece changes Fruit Flavored to Fruit Mixed In.",
    "read:mix_fruit_piece": "Fruit pieces, dried fruit, or fruit on the bottom add a Fruit mix-in.",
    "read:yogurt_probiotic": "The word probiotic adds Probiotic & Functional.",
    "read:yogurt_lowfat": "Low or reduced fat, or 1% or 2% milk, sets Low & Reduced Fat. Whole milk and “1% or less” do not count.",
    "read:yogurt_fatfree": "Nonfat or fat-free in the product name sets Fat Free.",
    "read:yogurt_drinkable": "The word drinkable in the product name adds Drinkable.",
    "read:yogurt_french": "The word french in the product name adds French.",
    "read:loaf_sourdough": "The word sourdough in the loaf name sets Sourdough.",
    "read:roll_brioche": "The word brioche in the roll name sets Brioche.",
    "read:bakery_oat": "The token oat in the bakery name adds Oats. Oatmeal is a different token.",
    "read:soda_fruit_name": "A fruit in the soda name sets Fruit. Spice, herbs, and ginger ale do not.",
    "read:flavor_bottom": "A bottom-layer phrase, including a hyphenated on-the-bottom, sets Fruit on the Bottom.",
    "read:soda_zs": "The shelf token zs adds Diet. The word zero by itself does not.",
    "read:cereal_only_oat": "A single real grain, oat, in the ingredient head sets Oat and removes Multiple.",
    "read:cereal_only_corn": "A single real grain, corn, in the ingredient head sets Corn and removes Multiple. Syrup and starch do not count.",
    "read:cereal_only_wheat": "A single real grain, wheat, in the ingredient head sets Wheat and removes Multiple.",
    "read:cereal_only_rice": "A single real grain, rice, in the ingredient head sets Rice and removes Multiple.",
    "read:soda_listed_caffeine": "Caffeine in the ingredient list, when the list does not say caffeine free, removes Caffeine Free.",
    "read:mix_no_granola": "When the name, ingredients, and description have no granola word, Granola is removed.",
    "read:sweetener_artificial": "Aspartame, sucralose, acesulfame, or saccharin in the ingredients sets Artificial Sweetener, unless stevia or monk fruit is also present.",
    "read:cereal_name_wheat": "The name token wht sets Wheat, unless the name also says rice.",
    "read:cereal_name_multiple": "Multigrain, including the shelf token multgr, sets Multiple.",
}


def _reading_line(rule: dict) -> str:
    feature = str(rule.get("feature", ""))
    if feature in READINGS:
        return READINGS[feature]
    if feature.startswith("title:"):
        return f"The name phrase “{feature[6:]}” emits {rule.get('value', '')}."
    value = rule.get("value", "")
    drop = rule.get("drop", "")
    parts = [feature]
    if value:
        parts.append(f"emits {value}")
    if drop:
        parts.append(f"removes {drop}")
    return "; ".join(parts)


def write_explanation(report: dict, result: dict) -> None:
    readings = "\n".join(f"- {_reading_line(rule)}" for rule in report.get("rules") or [])
    names = {
        "A100190": "Soda Type",
        "A100196": "Soda Feature",
        "A100334": "Yogurt Type",
        "A101614": "Cereal Source",
        "A101643": "Flavor Content",
        "A101897": "Roll Form",
        "A102531": "Yogurt Fat Content",
        "A900101": "Container Format",
    }
    epochs = report.get("selection", {}).get("epochs") or {}
    epoch_text = ", ".join(
        f"{names.get(aid, aid)} kept pass {count}" for aid, count in sorted(epochs.items())
    ) or "none"
    text = f"""# Brown Ballers solution

The work was to read messy grocery text and return exact attribute strings, and to show that the agent improved by updating its own skills. This note is that account: the frame, the form it took, the decisions, and what the experiment split taught. Hold-out Q is 65% of the rubric and is reported at the end. The other 35% is the before-and-after on experiment data. Performance is the hold-out number. The experiment comparison is the learning evidence.

## Approach

**Frame.** A product arrives as a short name, a description, and an ingredient list that does not always name the attribute the label cares about. Each requested attribute has a closed list of allowed strings, and a value counts only when it matches the label character for character. Q is micro-F1 over those annotated values. An extra wrong string is a false positive and still leaves the true value missing, so one bad guess is charged twice. Leaving the attribute empty costs only the miss. The skill worth learning is restraint: emit a string when the text supports that exact value, and stay quiet when it does not. Attributes with no label are ignored. Cost, latency, and the leaderboard composite are outside this score.

**Materialize.** That frame became a skill bank. `ParticipantAgent.evolve` calls `learn` on the experiment file. For each attribute, `learn` fits an averaged perceptron on tokens from the name, the description, and the ingredients. A product-disjoint slice of the same file is held back. On that slice the learner chooses which text view to trust, how large a margin to require before a value is emitted, whether an earlier pass of the weights was more exact, and which candidate readings survive. The saved bank is that result. `predict` loads the bank and does not change it. Foundation-model parameters are not trained. The feature detectors and the wording of the candidate readings were written by hand. The experiment labels decide the weights, and they decide which of those readings stay.

## Key decisions

1. **One skill per attribute.** A wheat loaf and a wheat cereal are different decisions written in the same kind of text. A shared weight vector would smear them. Each attribute keeps its own perceptron.
2. **A token must be seen twice.** Shelf copy is full of one-off spellings. A weight on a token from a single product would memorize that product. Tokens below that support never receive a weight.
3. **Let the split choose the view.** The name, the long description, and the ingredient cues disagree. Some attributes live in the name. Some live in the ingredient head. Each attribute may use the short text, the long text, a cue view, or a vote that keeps a value only when two of the three packs agree. A choice is kept when validation micro-F1 rises. An earlier training pass is kept on the same test, because more passes can start fitting noise. This run kept: {epoch_text}.
4. **Corn syrup is a sweetener.** The word corn in this catalog is usually syrup or starch. Counting it as a grain pulled sweetened products toward Multiple. The ingredient head also stops before the "contains 2%" tail, where vitamins and trace mentions are listed and are not the source.
5. **Readings are proposals.** A person can notice that "probiotic" or "nonfat" is a real shelf phrase. The split decides whether that phrase stays. A rule mined from training mistakes, and a product-name phrase, are kept only when validation micro-F1 rises. Diet, seed, and garlic need at least nine in ten of their training rows and a rise. Mix-in and probiotic need every training row and a rise. The fruit-piece reading is kept only when validation micro-F1 rises. The other readings are kept when the training rows that show the cue match exactly, whether the label is present, the label is the whole set, or a named label is absent, and validation micro-F1 does not fall. A reading that fails is absent from the bank.
6. **Two banks, two jobs.** The fair learning number uses weights that never saw the validation products. After that choice, the same packs are refit on all 965 experiment products and the margins are calibrated again, so the submitted agent has seen every experiment label. That refit is the agent that predicts. The evidence number is the earlier one.
7. **Learning ends when the bank is written.** The hold-out file is opened after the bank is saved. Each record is a prediction input with `labels` removed before `predict_row`. `predict` does not call `learn`. The organizer scorer is a separate program, and it is the step that reads labels.

The validation split is every product id whose first eight hexadecimal characters of MD5, read as an integer, leave a remainder of 0 on division by 5. That is about one product in five. Learning took {report.get('learning_seconds', 'n/a')} seconds. `evolve` and `scripts/build_submission.py` call the same `learn`. Before the prediction file is written, `learn` checks that a reloaded bank predicts the same values and that prediction does not change the weights.

## What the experiment taught

Same {report['validation_products']} validation products, same exact-match micro-F1. The learned row uses weights fit only on the other experiment products.

| Agent | What it is | Q |
| --- | --- | ---: |
| Empty predictions | Manual baseline, no values | {report['empty_q']:.4f} |
| Literal allowed-value copy | Manual baseline, before any weights | {report['literal_q']:.4f} |
| Learned skills | Automatic weights, packs, margins, epochs, and kept readings | {report['learned_q']:.4f} |

Learned counts: TP {report['learned_counts']['true_positive']}, FP {report['learned_counts']['false_positive']}, FN {report['learned_counts']['false_negative']}. Literal counts: TP {report['literal_counts']['true_positive']}, FP {report['literal_counts']['false_positive']}, FN {report['literal_counts']['false_negative']}.

The learning claim is the move from {report['literal_q']:.4f} to {report['learned_q']:.4f} on those same products. An empty agent scores zero because every labelled value is missed. The literal copy emits an allowed string whenever that string is written in the text. That is a serious manual baseline, and it still emits strings that are only flavor copy, a grain named inside a sweetener, or a marketing phrase. False positives fall from {report['literal_counts']['false_positive']} on that copy to {report['learned_counts']['false_positive']} on the learned skills, because the skills can abstain. Performance is the hold-out Q at the end. These two experiment numbers are the before and the after.

Compared on the experiment validation split in earlier runs, and not kept: a nearest-example memory, about 0.70, which copied a neighbor instead of reading the product; dropping name fragments, 0.838, so the shelf title is signal; ingredient-order tokens with a finer margin grid, about 0.844; applying every reliable name phrase at once, about 0.79. Margin-bin abstention, a category filter, token rules mined from training mistakes, and filling an abstention from similar products did not raise that split. Treating a second grain mention as a training feature for Multiple lowered an earlier result from 0.8660 to 0.8638, so that feature was not added to the perceptron.

The split kept teaching the same lesson. A reading that is often right still loses when it fires on every product, because a wrong value costs more than silence. A second grain is sometimes the single label Multiple and sometimes the grains named one by one. A sweetener is not always written as sugar or as corn syrup. A fruit piece is sometimes mixed through the cup and sometimes only a flavor. A culture name can belong to a traditional yogurt. Those cases stay with the perceptron. A hard reading does not take them, because the split already showed that a confident wrong answer is worse than a blank.

Readings the split kept:

{readings}

The list above is what the split kept. Each line is a general reading of product text, tied to one attribute. The records behind the lines are in `submission/evidence.json`.

## Pressure test

Checked against the written criteria before this note was finished.

| Criterion | Where this package meets it |
| --- | --- |
| Performance is hold-out Q, exact-match micro-F1. Cost and the composite are not the score. | `submission/result.json`, field `q`. Validity {result['validity_rate']}. Missing {result['missing_products']} of {result['holdout_products']}. |
| The organizer scorer is used unchanged, on every product. | `holdout_eval`, `scorer_version` `{result['scorer_version']}`, from [tamu-2026-ci](https://gitlab.com/mlmodels/tamu-2026-shared/tamu-2026-ci/). One prediction line per product. |
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

This is the performance number. The bank was already frozen. The unmodified `holdout_eval` utility then scored every product. `scorer_version` `{result['scorer_version']}`.

Q = {result['q']:.6f}

TP {result['true_positive']}, FP {result['false_positive']}, FN {result['false_negative']}. Validity {result['validity_rate']}. Missing products {result['missing_products']} of {result['holdout_products']}. Dataset `{result['dataset_version']}`. Team `{result['team_id']}`. This score is self-reported.
"""
    (SUBMISSION / "SOLUTION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
