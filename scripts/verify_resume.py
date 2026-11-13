"""Verify the local interrupted-versus-uninterrupted three-update GPU probe."""

import json
from pathlib import Path

from domaintune.baselines import sha256


def main():
    import torch
    from safetensors.torch import load_file

    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs/adapters"
    full = outputs / "continuation-check-uninterrupted"
    stopped = outputs / "continuation-check-stopped"
    resumed = outputs / "continuation-check-resumed-audit"
    manifests = [json.loads((p / "run.json").read_text("utf-8")) for p in (full, stopped, resumed)]
    assert [r["status"] for r in manifests] == [
        "probe_passed",
        "stopped_with_resumable_state",
        "probe_passed",
    ]
    audit = manifests[-1]["restoration_audit"]
    assert audit["optimizer_tensors_equal"] == 288
    assert audit["scheduler_equal"] and audit["adapter_checksum_verified"]
    logs = [
        [json.loads(line) for line in (p / "training.jsonl").read_text("utf-8").splitlines()]
        for p in (full, resumed)
    ]
    assert len(logs[0]) == len(logs[1]) == 3
    for expected, actual in zip(*logs, strict=True):
        for key in ("epoch", "step", "lr"):
            assert expected[key] == actual[key], (key, expected, actual)
        assert abs(expected["loss"] - actual["loss"]) < 0.001
    tensors = [load_file(p / "epoch-2/adapter_model.safetensors") for p in (full, resumed)]
    assert tensors[0].keys() == tensors[1].keys()
    max_difference = max(
        (tensors[0][key] - tensors[1][key]).abs().max().item() for key in tensors[0]
    )
    assert max_difference < 0.0005
    report = {
        "status": "verified",
        "probe_only": True,
        "updates": 3,
        "stopped_after_update": 1,
        "matching_learning_rates": True,
        "max_loss_difference": max(abs(a["loss"] - b["loss"]) for a, b in zip(*logs, strict=True)),
        "max_adapter_tensor_difference": max_difference,
        "loss_absolute_tolerance": 0.001,
        "adapter_absolute_tolerance": 0.0005,
        "bitwise_equal_adapter_tensors": sum(
            torch.equal(tensors[0][key], tensors[1][key]) for key in tensors[0]
        ),
        "adapter_tensor_count": len(tensors[0]),
        "restoration_audit": audit,
        "bitwise_reproducible": max_difference == 0,
        "restored": [
            "adapter",
            "optimizer",
            "scheduler",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "data_position",
        ],
        "runs": [str(p.relative_to(root)) for p in (full, stopped, resumed)],
        "adapter_sha256": sha256(full / "epoch-2/adapter_model.safetensors"),
        "source_commits": [r["git_commit"] for r in manifests],
        "limitation": (
            "Mechanics probe under concurrent local validation; "
            "not a performance benchmark or quality result. "
            "Stateful restoration does not establish bitwise-identical GPU continuation."
        ),
    }
    (root / "reports/phase3-continuation-check.json").write_text(
        json.dumps(report, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
