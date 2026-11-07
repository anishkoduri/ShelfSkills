from pathlib import Path

from datathon.harness import all_passed, validate_folder


def test_participant_agent_passes_contract() -> None:
    folder = Path(__file__).resolve().parents[1]
    results = validate_folder(folder)

    assert all_passed(results), [
        f"{result.status} {result.name}: {result.message}" for result in results
    ]
