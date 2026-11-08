"""Predict the hold-out from the frozen skill bank. Does not learn.

Hold-out labels are read only by the organizer scoring utility, after predictions
are already written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT.parent / "tamu-2026-ci"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CI))

from holdout_eval.scoring import score_files  # noqa: E402

from src.learner import SkillBank  # noqa: E402

HOLDOUT = CI / "data" / "holdout" / "holdout_private.jsonl"
STATE = ROOT / "submission" / "agent-state" / "skill-bank.json"
PREDICTIONS = ROOT / "submission" / "predictions.jsonl"
RESULT = ROOT / "submission" / "result.json"
TEAM = "team-brown-ballers"


def main() -> None:
    payload = json.loads(STATE.read_text(encoding="utf-8"))
    bank = SkillBank.from_dict(payload["bank"])
    with HOLDOUT.open(encoding="utf-8") as source, PREDICTIONS.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("labels", None)
            prediction = {
                "product_id": record["product_id"],
                "run_id": record["run_id"],
                "attributes": bank.predict_row(record),
            }
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    if RESULT.exists():
        RESULT.unlink()
    result = score_files(HOLDOUT, PREDICTIONS, team_id=TEAM)
    RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("q", "true_positive", "false_positive", "false_negative", "validity_rate", "missing_products")}, indent=2))


if __name__ == "__main__":
    main()
