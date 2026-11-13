"""Verify Phase 3 evidence and produce small versionable reports and charts."""

import argparse
import json
import math
from pathlib import Path

from domaintune.baselines import load_frozen_data, sha256
from domaintune.evaluate import evaluation_report
from domaintune.parse import InvalidPrediction, parse_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/adapters/phase3-r8"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = json.loads((args.run / "run.json").read_text("utf-8"))
    if run["status"] != "trained" or run["probe_steps"]:
        raise ValueError("A completed full training run is required")
    data, labels, hashes = load_frozen_data(root)
    assert hashes == run["processed_sha256"]
    assert len(run["epochs"]) == run["config"]["epochs"]
    updates = [json.loads(s) for s in (args.run / "training.jsonl").read_text("utf-8").splitlines()]
    assert len(updates) == run["total_optimizer_steps"]
    assert [u["step"] for u in updates] == list(range(1, len(updates) + 1))
    assert all(math.isfinite(u["loss"]) for u in updates)
    candidates = ([run["initial_validation"]] if "initial_validation" in run else []) + run[
        "epochs"
    ]
    for epoch in candidates:
        records = [
            json.loads(s)
            for s in (args.run / f"validation-epoch-{epoch['epoch']}.jsonl")
            .read_text("utf-8")
            .splitlines()
        ]
        assert [r["sample_id"] for r in records] == [r["sample_id"] for r in data["validation"]]
        predictions = []
        for record in records:
            try:
                prediction = parse_prediction(record["output"], labels)
            except InvalidPrediction:
                prediction = None
            assert prediction == record["prediction"]
            predictions.append(prediction)
        rebuilt = evaluation_report(
            data["validation"], predictions, labels, [r["output"] for r in records]
        )
        original = json.loads(
            (args.run / f"validation-epoch-{epoch['epoch']}.json").read_text("utf-8")
        )
        assert all(rebuilt[k] == original[k] for k in rebuilt)
    selected = max(candidates, key=lambda e: e["macro_f1"])
    assert selected["epoch"] == run["best_epoch"]
    reload = json.loads((args.run / "reload-validation.json").read_text("utf-8"))
    assert reload["reproduced_training_validation"] and reload["clean_process_reload"]
    assert reload["checkpoint"] == run["selected_checkpoint"]
    assert reload["adapter_sha256"] == sha256(
        root / run["selected_checkpoint"] / "adapter_model.safetensors"
    )
    baseline = json.loads((root / "reports/baselines-llm.json").read_text("utf-8"))
    base_f1 = baseline["results"]["unchanged_llm"]["macro_f1"]
    run.update(
        status="verified",
        clean_process_reload=True,
        independently_reconstructed_validation=True,
        unchanged_macro_f1=base_f1,
        macro_f1_gain=reload["macro_f1"] - base_f1,
        adapter_sha256=reload["adapter_sha256"],
        training_log_sha256=sha256(args.run / "training.jsonl"),
    )
    for name, report in (("phase3-run.json", run), ("phase3-validation.json", reload)):
        (root / "reports" / name).write_text(json.dumps(report, indent=2) + "\n", "utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([u["step"] for u in updates], [u["loss"] for u in updates], alpha=0.6)
    axes[0].set(xlabel="Optimizer update", ylabel="Completion loss")
    axes[1].plot(
        [e["epoch"] for e in candidates],
        [e["macro_f1"] for e in candidates],
        marker="o",
        label="LoRA r8",
    )
    axes[1].axhline(base_f1, linestyle="--", label="Unchanged Qwen")
    axes[1].axhline(0.8909865946, linestyle=":", label="TF-IDF")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(root / "reports/phase3-training.png", dpi=140)
    plt.close(fig)
    summary = (
        "\n\n## Verified primary run\n\n"
        f"Selected epoch {run['best_epoch']} by validation macro-F1. "
        f"Validation accuracy {reload['accuracy']:.4f}, macro-F1 {reload['macro_f1']:.6f}, "
        f"invalid rate {reload['invalid_rate']:.4f}. Unchanged-model macro-F1 {base_f1:.6f}; "
        f"gain {run['macro_f1_gain']:.6f}. TF-IDF macro-F1 remains 0.890987.\n\n"
        "All epoch predictions were independently reparsed against frozen validation IDs; "
        "metrics, errors, and confusion counts match. Clean-process reload reproduced the "
        "selected epoch exactly. See phase3-run.json, phase3-validation.json, and "
        "phase3-training.png. Official test evaluation has not run. Phase 3 is complete; "
        "stop before Phase 4 pending the user's instruction.\n"
    )
    if "continuation" in run:
        summary += (
            "\nThis is a separate continuation experiment from the saved epoch-1 adapter. "
            "AdamW and the linear schedule were reset for two additional epochs. "
            "It is not an exact resume of the original three-epoch run. "
            "The starting epoch-1 checkpoint was also eligible for selection.\n"
        )
    report_path = root / "reports/training.md"
    text = report_path.read_text("utf-8").split("\n\n## Verified primary run")[0]
    if "continuation" in run:
        text = text.replace(
            "Status: reset-state continuation authorized on 2026-11-13; Phase 3 incomplete.",
            "Status: Phase 3 complete through a separately recorded reset-state continuation.\n"
            "The original stopped run is retained below as historical evidence.",
        )
    text = text.replace(
        "Status: implementation and 640-token feasibility verified; primary run in progress.",
        "Status: Phase 3 complete; full training and clean-process reload verified.",
    )
    report_path.write_text(text + summary, "utf-8")
    with (root / "reports/experiment_log.md").open("a", encoding="utf-8") as log:
        log.write(summary)
    handoff = root / "docs/session-handoff.md"
    text = handoff.read_text("utf-8")
    text = text.replace(
        "Phase 3 is running locally; do not launch a second training process.",
        "Phase 3 is complete. STOP before Phase 4 until the user requests it.",
    )
    if "continuation" in run:
        text = (
            "# Current status: Phase 3 complete\n\n"
            "The user authorized continuation after epoch 1 on 2026-11-13. "
            "Two additional epochs ran with reset optimizer/scheduler state as a separate "
            "experiment. See the verified report below. Stop before Phase 4.\n\n"
            "The following earlier stopping-point notes are historical.\n\n" + text
        )
    handoff.write_text(text + summary, "utf-8")
    readme = root / "README.md"
    text = readme.read_text("utf-8").replace(
        "Status: Phase 3 in progress.", "Status: Phase 3 complete."
    )
    text = text.replace(
        "Status: Phase 3 continuation authorized from epoch 1 with reset training state.",
        "Status: Phase 3 complete through a verified reset-state continuation after epoch 1.",
    )
    readme.write_text(text, "utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
