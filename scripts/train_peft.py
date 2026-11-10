"""Train the primary LoRA experiment using frozen train/validation only."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path

import yaml

from domaintune.baselines import load_frozen_data, sha256
from domaintune.predict import evaluate_adapter
from domaintune.prompts import PROMPT_VERSION
from domaintune.train import accumulation_size, encode_completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora_r8.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/adapters/phase3-r8"))
    parser.add_argument("--probe-steps", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.drive.upper() != "D:" or not os.environ.get("HF_HOME", "").startswith(str(root)):
        raise RuntimeError("Dot-source scripts/environment.ps1 first")
    output = args.output.resolve()
    if not output.is_relative_to(root / "outputs"):
        raise ValueError("Artifacts must stay under project outputs on D:")
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text("utf-8"))
    report = {"status": "running", "config": config, "probe_steps": args.probe_steps}
    started = time.perf_counter()

    def save_report():
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
            set_seed,
        )

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Measured primary configuration requires BF16 CUDA")
        if config["micro_batch_size"] != 1 or config["selection_metric"] != "validation_macro_f1":
            raise ValueError("Unsupported microbatch or selection metric")
        torch.set_num_threads(4)
        set_seed(config["seed"])
        data, labels, hashes = load_frozen_data(root)
        report.update(
            processed_sha256=hashes,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=sha256(root / "src/domaintune/prompts.py"),
            provenance_sha256=sha256(root / "data/manifests/provenance.json"),
            lock_sha256=sha256(root / "uv.lock"),
            config_sha256=sha256(args.config),
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            python=platform.python_version(),
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            precision="bfloat16",
            seed=config["seed"],
            versions={
                n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "peft", "accelerate", "numpy")
            },
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=config["model_revision"], local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        encoded = [
            encode_completion(tokenizer, r, labels, config["max_sequence_length"])
            for r in data["train"]
        ]
        lengths = [sum(r["attention_mask"]) for r in encoded]
        report["tokenization"] = {
            "count": len(encoded),
            "min": min(lengths),
            "max": max(lengths),
            "padding": config["max_sequence_length"],
        }
        # All real rows are checked, including each of the 77 completion labels.
        report["masked_prompt_and_padding"] = all(
            x["labels"][0] == -100
            and any(t != -100 for t in x["labels"])
            and all(
                y == -100
                for y, mask in zip(x["labels"], x["attention_mask"], strict=True)
                if mask == 0
            )
            for x in encoded
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model = get_peft_model(
            model,
            LoraConfig(
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
        model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        report["trainable_parameters"], report["total_parameters"] = (
            model.get_nb_trainable_parameters()
        )
        accumulation = config["gradient_accumulation_steps"]
        total_steps = math.ceil(len(encoded) / accumulation) * config["epochs"]
        if args.probe_steps:
            total_steps = args.probe_steps
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, math.ceil(total_steps * config["warmup_ratio"]), total_steps
        )
        report.update(total_optimizer_steps=total_steps, epochs=[], best_epoch=None)
        save_report()
        step = 0
        best = -1.0
        with (output / "training.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(config["epochs"]):
                model.train()
                model.config.use_cache = False
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                epoch_started = time.perf_counter()
                order = list(range(len(encoded)))
                random.Random(config["seed"] + epoch).shuffle(order)
                optimizer.zero_grad(set_to_none=True)
                update_loss = 0.0
                losses = []
                for index, row_index in enumerate(order):
                    batch = {
                        k: torch.tensor([v], device="cuda") for k, v in encoded[row_index].items()
                    }
                    divisor = accumulation_size(index, len(order), accumulation)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    value = loss.item()
                    if not math.isfinite(value):
                        raise RuntimeError("Nonfinite training loss")
                    (loss / divisor).backward()
                    update_loss += value / divisor
                    del batch, loss
                    if (index + 1) % accumulation == 0 or index + 1 == len(order):
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        if not math.isfinite(norm.item()):
                            raise RuntimeError("Nonfinite gradient norm")
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        entry = {
                            "epoch": epoch + 1,
                            "step": step,
                            "loss": update_loss,
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                        log.write(json.dumps(entry) + "\n")
                        log.flush()
                        losses.append(update_loss)
                        update_loss = 0.0
                        if step % 10 == 0 or args.probe_steps:
                            print(json.dumps(entry), flush=True)
                        if args.probe_steps and step >= args.probe_steps:
                            break
                torch.cuda.synchronize()
                epoch_report = {
                    "epoch": epoch + 1,
                    "mean_update_loss": sum(losses) / len(losses),
                    "train_seconds": time.perf_counter() - epoch_started,
                    "optimizer_steps": len(losses),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                checkpoint = output / f"epoch-{epoch + 1}"
                model.save_pretrained(checkpoint)
                tokenizer.save_pretrained(checkpoint)
                if not args.probe_steps:
                    gc.collect()
                    torch.cuda.empty_cache()
                    validation = evaluate_adapter(
                        model,
                        tokenizer,
                        data["validation"],
                        labels,
                        output / f"validation-epoch-{epoch + 1}.jsonl",
                    )
                    (output / f"validation-epoch-{epoch + 1}.json").write_text(
                        json.dumps(validation, indent=2) + "\n", "utf-8"
                    )
                    epoch_report.update(
                        {k: validation[k] for k in ("accuracy", "macro_f1", "invalid_rate")}
                    )
                    if validation["macro_f1"] > best:
                        best = validation["macro_f1"]
                        report["best_epoch"] = epoch + 1
                        report["selected_checkpoint"] = str(checkpoint.relative_to(root))
                report["epochs"].append(epoch_report)
                print(json.dumps(epoch_report), flush=True)
                save_report()
                if args.probe_steps:
                    break
        report["status"] = "probe_passed" if args.probe_steps else "trained"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        if "torch" in locals() and torch.cuda.is_available():
            report["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        raise
    finally:
        save_report()


if __name__ == "__main__":
    main()
