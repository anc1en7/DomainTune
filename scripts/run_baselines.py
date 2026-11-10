"""Reproduce Phase 2 validation baselines using only local pinned artifacts."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import yaml

from domaintune.baselines import classical_baselines, load_frozen_data, sha256
from domaintune.evaluate import evaluation_report
from domaintune.parse import InvalidPrediction, parse_prediction
from domaintune.prompts import PROMPT_VERSION, build_messages


def llm_baseline(root, rows, labels, config):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this measured configuration")
    torch.set_num_threads(4)
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], local_files_only=True
    )
    started = time.perf_counter()
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    load_seconds = time.perf_counter() - started
    if config["prompt_version"] != PROMPT_VERSION:
        raise ValueError("Unsupported prompt version")

    def generate(row):
        encoded = tokenizer.apply_chat_template(
            build_messages(row["text"], labels),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        if (
            encoded["input_ids"].shape[1] + config["max_new_tokens"]
            > model.config.max_position_embeddings
        ):
            raise ValueError("Input exceeds context; truncation forbidden")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=config["max_new_tokens"],
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        return tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

    for row in rows[: config["warmup"]]:
        generate(row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    outputs, predictions, latencies = [], [], []
    output_dir = root / "outputs/phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "unchanged_predictions.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = generate(row)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            try:
                prediction = parse_prediction(output, labels)
            except InvalidPrediction:
                prediction = None
            outputs.append(output)
            predictions.append(prediction)
            log.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "output": output,
                        "prediction": prediction,
                        "seconds": latencies[-1],
                    }
                )
                + "\n"
            )
            log.flush()
            if (index + 1) % 50 == 0:
                print(json.dumps({"completed": index + 1, "total": len(rows)}), flush=True)
    report = evaluation_report(rows, predictions, labels, outputs)
    report.update(
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        precision="bfloat16",
        batch_size=1,
        warmup=config["warmup"],
        model_load_seconds=load_seconds,
        latency_scope="chat tokenization, device transfer, generation, decoding; synchronized",
        p50_ms=float(np.percentile(latencies, 50) * 1000),
        p95_ms=float(np.percentile(latencies, 95) * 1000),
        throughput_per_second=len(rows) / sum(latencies),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--system", choices=["classical", "llm", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.drive.upper() != "D:" or not os.environ.get("HF_HOME", "").startswith(str(root)):
        raise RuntimeError("Dot-source scripts/environment.ps1 first")
    config = yaml.safe_load(args.config.read_text("utf-8"))
    if config["selection_split"] != "validation":
        raise ValueError("Phase 2 supports validation only")
    data, labels, hashes = load_frozen_data(root)
    report = {
        "split": "validation",
        "config": config,
        "processed_sha256": hashes,
        "provenance_sha256": sha256(root / "data/manifests/provenance.json"),
        "prompt_sha256": sha256(root / "src/domaintune/prompts.py"),
        "lock_sha256": sha256(root / "uv.lock"),
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("scikit-learn", "numpy", "torch", "transformers")
        },
    }
    if args.system in ("classical", "all"):
        report["results"] = classical_baselines(data, labels, config)
    else:
        report["results"] = {}
    if args.system in ("llm", "all"):
        report["results"]["unchanged_llm"] = llm_baseline(root, data["validation"], labels, config)
    destination = root / f"reports/baselines-{args.system}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(str(destination), flush=True)


if __name__ == "__main__":
    main()
