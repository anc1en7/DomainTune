"""Clean-process adapter validation; official test evaluation is deferred to Phase 5."""

import argparse
import json
import os
from pathlib import Path

from domaintune.baselines import load_frozen_data, sha256
from domaintune.predict import evaluate_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh output directory for standalone validation of an interrupted checkpoint",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.drive.upper() != "D:" or not os.environ.get("HF_HOME", "").startswith(str(root)):
        raise RuntimeError("Dot-source scripts/environment.ps1 first")
    checkpoint = args.checkpoint.resolve()
    original_path = checkpoint.parent / f"validation-{checkpoint.name}.json"
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(root / "outputs"):
            raise ValueError("Evaluation artifacts must stay under project outputs on D:")
        destination.mkdir(parents=True, exist_ok=False)
    else:
        if not original_path.is_file():
            raise ValueError(
                "No completed validation exists; use --output for standalone validation"
            )
        destination = checkpoint.parent
        if (destination / "reload-predictions.jsonl").exists():
            raise ValueError("Reload predictions already exist; use a fresh --output directory")
    run = json.loads((checkpoint.parent / "run.json").read_text("utf-8"))
    config = run["config"]
    data, labels, hashes = load_frozen_data(root)
    if hashes != run["processed_sha256"]:
        raise ValueError("Training data provenance mismatch")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    torch.set_num_threads(4)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model = PeftModel.from_pretrained(model, checkpoint)
    report = evaluate_adapter(
        model, tokenizer, data["validation"], labels, destination / "reload-predictions.jsonl"
    )
    report.update(
        checkpoint=str(checkpoint.relative_to(root)),
        adapter_sha256=sha256(checkpoint / "adapter_model.safetensors"),
        processed_sha256=hashes,
        clean_process_reload=True,
    )
    report["reproduced_training_validation"] = False
    if original_path.is_file():
        original = json.loads(original_path.read_text("utf-8"))
        for key in ("accuracy", "macro_f1", "invalid_rate", "confusion", "errors"):
            if original[key] != report[key]:
                raise RuntimeError(f"Reload validation mismatch: {key}")
        report["reproduced_training_validation"] = True
    report["source_run_status"] = run["status"]
    (destination / "reload-validation.json").write_text(
        json.dumps(report, indent=2) + "\n", "utf-8"
    )
    print(json.dumps({k: report[k] for k in ("accuracy", "macro_f1", "invalid_rate")}), flush=True)


if __name__ == "__main__":
    main()
