"""Participant agent. Learning updates attribute skills; prediction does not learn."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from datathon.clients import MockModel
from datathon.contract import (
    BaselineContext,
    ChallengeAgent,
    EvolutionContext,
    EvolutionResult,
    Model,
    PredictionBatch,
    PredictionBatchResponse,
    PredictionResponse,
)

from src.learner import SkillBank, learn, rows_from_examples


class ParticipantAgent(ChallengeAgent):
    def __init__(self) -> None:
        self.model: Model = MockModel()
        self.bank: SkillBank | None = None
        self.learned = False

    def initialize(self, baseline: BaselineContext) -> None:
        self.model = baseline.model
        self.bank = None
        self.learned = False

    def evolve(self, context: EvolutionContext) -> EvolutionResult:
        if self.learned:
            return EvolutionResult(changed=False, stop=True, notes="Skills already learned")
        examples = [
            example for example in context.training_examples
            if getattr(example, "labels", None)
        ]
        if not examples:
            return EvolutionResult(changed=False, stop=True, notes="No labeled experiment examples")
        rows = rows_from_examples(examples)
        _evidence_bank, self.bank, _report = learn(rows)
        self.learned = True
        return EvolutionResult(
            changed=True,
            stop=True,
            notes="Learned per-attribute evidence skills from experiment labels",
        )

    @staticmethod
    def log_learning_trace(
        trace_path: Path | None,
        *,
        iteration: int,
        examined: list[str],
        changed: bool,
        notes: str = "",
        state_before: dict[str, object] | None = None,
        state_after: dict[str, object] | None = None,
        observations: dict[str, object] | None = None,
    ) -> None:
        """Append optional team-reported diagnostics, never an official evaluation."""
        if trace_path is None:
            return
        if type(iteration) is not int or iteration < 1:
            raise ValueError("Trace iteration must be a positive integer")
        record = {
            "schema_version": "participant-learning-trace-v1",
            "source": "participant-reported",
            "recorded_at": datetime.now(UTC).isoformat(),
            "iteration": iteration,
            "examined": examined,
            "changed": changed,
            "notes": notes,
            "state_before": state_before,
            "state_after": state_after,
            "observations": observations,
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)

    def predict(self, batch: PredictionBatch) -> PredictionBatchResponse:
        responses = []
        for item in batch.items:
            request = item.request
            responses.append(PredictionResponse(self._predict_request(request)))
        return PredictionBatchResponse.from_responses(batch, responses)

    def _predict_request(self, request: object) -> dict[str, list[str]]:
        description = getattr(request, "input_text", None) or getattr(request, "title", "")
        specs = list(getattr(request, "attribute_specs", []) or [])
        if not specs:
            return {attribute_id: [] for attribute_id in getattr(request, "attributes", [])}
        row = {
            "product_description": description,
            "attributes": [_spec_to_dict(spec) for spec in specs],
        }
        if self.bank is None:
            return {spec["attribute_id"]: [] for spec in row["attributes"]}
        return self.bank.predict_row(row)

    def save(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_prediction_contract(output_path)
        payload = {
            "learned": self.learned,
            "bank": None if self.bank is None else self.bank.to_dict(),
        }
        (output_path / "skill-bank.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, artifact_path: Path) -> ParticipantAgent:
        cls.require_prediction_contract(artifact_path)
        payload = json.loads((artifact_path / "skill-bank.json").read_text(encoding="utf-8"))
        agent = cls()
        agent.learned = bool(payload.get("learned"))
        bank = payload.get("bank")
        agent.bank = None if bank is None else SkillBank.from_dict(bank)
        return agent


def _spec_to_dict(spec: object) -> dict:
    values = []
    for item in getattr(spec, "allowed_values", []) or []:
        if isinstance(item, dict):
            values.append({"name": item.get("name", ""), "definition": item.get("definition", "")})
        elif hasattr(item, "name"):
            values.append({"name": item.name, "definition": getattr(item, "definition", "")})
        else:
            values.append({"name": str(item), "definition": ""})
    return {
        "attribute_id": spec.attribute_id,
        "attribute_name": getattr(spec, "attribute_name", spec.attribute_id),
        "definition": getattr(spec, "definition", ""),
        "multi_valued": bool(getattr(spec, "multi_valued", False)),
        "allowed_values": values,
    }
