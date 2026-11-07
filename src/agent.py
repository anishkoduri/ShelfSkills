"""A minimal no-op agent that passes the benchmark contract checks."""

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


class ParticipantAgent(ChallengeAgent):
    def __init__(self) -> None:
        self.rules: dict[str, str] = {}
        self.model: Model = MockModel()

    def initialize(self, baseline: BaselineContext) -> None:
        self.rules = dict(baseline.rules)
        self.model = baseline.model

    def evolve(self, context: EvolutionContext) -> EvolutionResult:
        return EvolutionResult(changed=False, stop=True, notes="Starter agent is a no-op")

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
        if any(item.request.attributes for item in batch.items):
            return PredictionBatchResponse.from_responses(batch, [
                PredictionResponse({attribute_id: [] for attribute_id in item.request.attributes})
                for item in batch.items
            ])
        return self.model.predict(batch, self.rules)

    def save(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_prediction_contract(output_path)
        (output_path / "rules.json").write_text(
            json.dumps(self.rules, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, artifact_path: Path) -> "ParticipantAgent":
        cls.require_prediction_contract(artifact_path)
        rules = json.loads((artifact_path / "rules.json").read_text(encoding="utf-8"))
        if not isinstance(rules, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in rules.items()
        ):
            raise ValueError("Agent rules must map strings to strings")
        agent = cls()
        agent.rules = rules
        return agent
