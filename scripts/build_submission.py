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

## Judging

**Performance (65%).** This portion is the hold-out quality score Q, exact-match micro-F1. Composite scores and resource costs are not included. Q is the `q` field in `submission/result.json`, produced by the unmodified scorer from [tamu-2026-ci](https://gitlab.com/mlmodels/tamu-2026-shared/tamu-2026-ci/).

**Learning and experimental evidence (35%).** The solution improves by updating reusable skills: the perceptron weights, the feature pack, the margin, the epoch, and which readings to keep. It does not update foundation-model parameters. The before-and-after results are on the experiment validation split, in the table below and in `submission/evidence.json`. The feature detectors and the wording of the candidate readings are manual. The weights and the keep-or-drop decision are automatic.

The starter `ParticipantAgent` interface is the one this code implements. The full harness is not used for this run. Existing model, tool, and data limits still apply: no foundation-model weight update, and learning reads the experiment package.

The bank is saved before the hold-out file is opened, and final inference does not learn. Hold-out records are prediction inputs only. Each record has `labels` removed before it reaches the agent. Training, pack choice, margins, epochs, and the keep-or-drop tests use the experiment file. The number reported below is the scorer's Q, with the counts the scorer printed.

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

The experiment file is split by product id. A product is in the validation split when the first eight hexadecimal characters of MD5(product id), read as an integer, leave a remainder of 0 on division by 5. That is about one product in five. The other products train the weights. On the validation products the learner chooses, for each attribute, the feature pack (`short`, `long`, or `cue`), whether two of the three packs must agree, the margin required before a value is emitted, and whether an earlier epoch is more exact than the final pass. An earlier pass is kept only when validation micro-F1 rises. Passes kept this run: {epoch_text}.

Candidate readings are hand-written. Keeping one is automatic, and a reading that fails its test is left out. A rule mined from training mistakes is kept only when validation micro-F1 rises. A product-name phrase is kept only when that attribute's validation micro-F1 rises. The diet, seed, and garlic candidates need at least nine in ten of their training rows, and are kept only when validation micro-F1 rises. The mix-in and probiotic candidates need every training row, and are kept only when validation micro-F1 rises. The fruit-piece candidate is kept only when validation micro-F1 rises. The remaining candidates are kept when the training rows that show the cue match the reading exactly, whether that means the label is present, the label is the whole set, or a named label is absent, and validation micro-F1 does not fall. The kept readings are listed below. The exact records are in `submission/evidence.json`.

{readings}

`evolve` and this script call the same `learn` function. No foundation-model weights are updated. The saved state is the skill bank. Before the prediction file is opened, `learn` checks that a reloaded bank predicts the same values and that prediction does not change the weights. `predict` does not call `learn`.

The table below is the validation split, with weights that did not see those products. The saved bank then refits the same packs on all 965 experiment products and calibrates margins again on the validation split. That refit is the submitted agent. It is not the learning-evidence number. Learning took {report.get('learning_seconds', 'n/a')} seconds.

## Learning evidence

Same {report['validation_products']} validation products, same exact-match micro-F1. The learned row uses weights fit only on the other experiment products.

| Agent | What it is | Q |
| --- | --- | ---: |
| Empty predictions | Manual baseline, no values | {report['empty_q']:.4f} |
| Literal allowed-value copy | Manual baseline, before any weights | {report['literal_q']:.4f} |
| Learned skills | Automated weights, packs, margins, epochs, and kept readings | {report['learned_q']:.4f} |

Learned counts: TP {report['learned_counts']['true_positive']}, FP {report['learned_counts']['false_positive']}, FN {report['learned_counts']['false_negative']}. Literal counts: TP {report['literal_counts']['true_positive']}, FP {report['literal_counts']['false_positive']}, FN {report['literal_counts']['false_negative']}.

What is automated: the perceptron weights, the pack, the vote, the margin, the epoch, and the keep-or-drop decision for each candidate reading. What is manual: the feature detectors and the wording of the candidate readings. A fixed copy of an allowed value, with no weights, is the before side of the comparison.

Compared on the experiment validation split in earlier runs, and not kept: a nearest-example memory, about 0.70; dropping name fragments, 0.838; ingredient-order tokens with a finer margin grid, about 0.844; applying every reliable name phrase at once, about 0.79. Margin-bin abstention, a category filter, token rules mined from training mistakes, and filling an abstention from similar products did not raise that split. Treating a second grain mention as a training feature for Multiple lowered an earlier result from 0.8660 to 0.8638, so that feature was not added to the perceptron. A name phrase is kept only for the attributes where the split rose.

The remaining misses on the validation products are ambiguous text. A second grain is sometimes the label Multiple and sometimes the grains named one by one. A sweetener is not always named in the ingredients. A fruit piece is sometimes mixed in and sometimes only a flavor. Culture names are not a reliable probiotic. Those cases stay with the perceptron.

## Hold-out result

Scored with the unmodified `holdout_eval` utility, `scorer_version` `{result['scorer_version']}`.

Q = {result['q']:.6f}

TP {result['true_positive']}, FP {result['false_positive']}, FN {result['false_negative']}. Validity {result['validity_rate']}. Missing products {result['missing_products']} of {result['holdout_products']}. Dataset `{result['dataset_version']}`. Team `{result['team_id']}`. This score is self-reported.
"""
    (SUBMISSION / "SOLUTION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
