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


def write_explanation(report: dict, result: dict) -> None:
    text = f"""# Brown Ballers solution

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

The number in the table is that split, with weights that never saw those products. The submitted bank then relearns the same packs on all 965 experiment products and calibrates margins again on the split, using the same product reading as prediction. That second pass is not a fair held-out score, so it is not the learning evidence. Hold-out products are not in the experiment file. Hold-out labels are not read until the official scorer runs, and they are not used to choose skills. Learning took {report.get('learning_seconds', 'n/a')} seconds. Prediction does not learn.

## Learning evidence

Same {report['validation_products']} validation products, same micro-F1:

| Agent | Q |
| --- | ---: |
| Empty predictions | {report['empty_q']:.4f} |
| Literal allowed-value copy, before learning | {report['literal_q']:.4f} |
| Learned skills, trained without these products | {report['learned_q']:.4f} |

Learned counts: TP {report['learned_counts']['true_positive']}, FP {report['learned_counts']['false_positive']}, FN {report['learned_counts']['false_negative']}.

The literal copy is a manual baseline. The perceptron weights, the pack choice, the vote, the margins, and any reader override are the automated update. Reader overrides kept this round: {json.dumps(report.get('rules') or [])}.

What did not earn a place in the bank: a nearest-example memory, about 0.70 on this split; dropping name fragments entirely, which fell to 0.838; ingredient-order tokens with a finer margin grid, which fell to about 0.844. One reader override did earn a place: a plant sweetener in the ingredient list is emitted as Plant-Based Sweetener. The remaining experiment misses are ambiguous descriptions: a grain named in the label but not in the text, corn syrup versus the word sugar, fruit mixed in versus fruit flavored, and citrus versus fruit soda.

## Hold-out result

Q = {result['q']:.6f}

TP {result['true_positive']}, FP {result['false_positive']}, FN {result['false_negative']}, validity {result['validity_rate']}, missing products {result['missing_products']} of {result['holdout_products']}. Dataset `{result['dataset_version']}`, scorer `{result['scorer_version']}`. This score is self-reported.
"""
    (SUBMISSION / "SOLUTION.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
