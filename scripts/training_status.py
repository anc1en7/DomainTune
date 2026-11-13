"""Read local continuation progress and estimate remaining wall time."""

import json
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    run_dir = root / "outputs/adapters/phase3-r8-continuation"
    initial_dir = root / "outputs/phase3-epoch1-validation"

    def read_json(path):
        return json.loads(path.read_text("utf-8")) if path.is_file() else None

    def records(path):
        return (
            [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            if path.is_file()
            else []
        )

    initial = read_json(initial_dir / "reload-validation.json")
    run = read_json(run_dir / "run.json")
    result = {"stage": "preparing_continuation"}
    if not initial:
        path = initial_dir / "reload-predictions.jsonl"
        count = len(records(path))
        result.update(stage="validating_epoch_1", validation_completed=count, validation_total=1000)
        if count:
            result["current_validation_minutes_remaining"] = round(
                (time.time() - path.stat().st_ctime) / count * (1000 - count) / 60, 1
            )
    if run:
        updates = records(run_dir / "training.jsonl")
        completed = updates[-1]["step"] if updates else 0
        result.update(
            stage=run["status"],
            optimizer_updates_completed=completed,
            optimizer_updates_total=run["total_optimizer_steps"],
            latest_state=run.get("latest_state"),
            best_epoch=run.get("best_epoch"),
        )
        if updates:
            result["epoch"] = updates[-1]["epoch"]
            result["latest_loss"] = updates[-1]["loss"]
            result["seconds_since_training_update"] = round(
                time.time() - (run_dir / "training.jsonl").stat().st_mtime
            )
        validation_seconds = initial["generation_seconds"] if initial else 1500
        validation_remaining = 0
        validation_progress = {}
        for epoch in (2, 3):
            report = read_json(run_dir / f"validation-epoch-{epoch}.json")
            count = len(records(run_dir / f"validation-epoch-{epoch}.jsonl"))
            validation_progress[str(epoch)] = 1000 if report else count
            if not report:
                validation_remaining += validation_seconds * (1000 - count) / 1000
        result["validation_completed_by_epoch"] = validation_progress
        reload_count = len(records(run_dir / "reload-predictions.jsonl"))
        reload_report = read_json(run_dir / "reload-validation.json")
        validation_remaining += (
            0 if reload_report else validation_seconds * (1000 - reload_count) / 1000
        )
        result["reload_validation_completed"] = 1000 if reload_report else reload_count
        if len(updates) >= 2:
            recent = updates[-min(31, len(updates)) :]
            seconds_per_update = (recent[-1]["elapsed_seconds"] - recent[0]["elapsed_seconds"]) / (
                recent[-1]["step"] - recent[0]["step"]
            )
            remaining = (run["total_optimizer_steps"] - completed) * seconds_per_update
            result["estimated_minutes_remaining_including_validation"] = round(
                (remaining + validation_remaining) / 60, 1
            )
            result["seconds_per_update"] = round(seconds_per_update, 2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
