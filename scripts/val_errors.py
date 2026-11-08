"""Validation-split errors only. Does not open the hold-out file."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.learner import is_validation, labels_of, select_bank  # noqa: E402

ROWS = [
    json.loads(line)
    for line in (ROOT / "data" / "synth_v6_full" / "product_attribute_experiment.jsonl").open(encoding="utf-8")
    if line.strip()
]


def main() -> None:
    bank, _info = select_bank(ROWS)
    catalog = bank.catalog
    counts: dict[str, list[int]] = {aid: [0, 0, 0] for aid in catalog}
    confusions: dict[str, Counter] = {aid: Counter() for aid in catalog}
    for row in ROWS:
        if not is_validation(row["product_id"]):
            continue
        prediction = bank.predict_row(row)
        labels = labels_of(row)
        for aid, expected in labels.items():
            predicted = set(prediction.get(aid, []))
            counts[aid][0] += len(predicted & expected)
            counts[aid][1] += len(predicted - expected)
            counts[aid][2] += len(expected - predicted)
            if predicted != expected:
                confusions[aid][(tuple(sorted(expected)), tuple(sorted(predicted)))] += 1
    ranked = []
    for aid, (tp, fp, fn) in counts.items():
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom else 0.0
        ranked.append((f1, catalog[aid]["name"], aid, tp, fp, fn))
    for f1, name, aid, tp, fp, fn in sorted(ranked):
        print(f"{f1:.3f} {name} {aid} tp={tp} fp={fp} fn={fn} mode={bank.modes[aid]}")
        for pair, n in confusions[aid].most_common(4):
            print(f"    {n} expected={pair[0]} predicted={pair[1]}")


if __name__ == "__main__":
    main()
