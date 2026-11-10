"""Local GPU feasibility probe; uses official training data only, never test data."""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument(
        "--model-revision", help="Immutable model commit; resolves upstream if omitted"
    )
    parser.add_argument(
        "--dataset-revision", help="Immutable data commit; resolves upstream if omitted"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/phase0"))
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.examples) < 1:
        parser.error("steps, accumulation, and examples must be positive")
    root = Path(__file__).resolve().parents[1]
    if root.drive.upper() != "D:" or not os.environ.get("HF_HOME", "").startswith(str(root)):
        raise RuntimeError("Dot-source scripts/environment.ps1 before running")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(root):
        raise ValueError("Output must stay inside the project on D:")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "steps": args.steps,
        "accumulation": args.accumulation,
        "examples": args.examples,
        "micro_batch_size": 1,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
        "learning_rate": 2e-4,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "gradient_checkpointing": True,
    }
    started = time.perf_counter()
    try:
        import importlib.metadata

        import torch
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        from domaintune.parse import parse_prediction
        from domaintune.prompts import PROMPT_VERSION, build_messages

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing an unmeasured CPU training fallback")
        set_seed(42)
        torch.set_num_threads(4)
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        dataset_id = "PolyAI/banking77"
        api = HfApi()
        model_info = api.model_info(model_id, revision=args.model_revision, timeout=30)
        dataset_info = api.dataset_info(dataset_id, revision=args.dataset_revision, timeout=30)
        report.update(
            model_id=model_id,
            model_revision=model_info.sha,
            dataset_id=dataset_id,
            dataset_revision=dataset_info.sha,
            model_license=getattr(model_info.card_data, "license", None),
            dataset_license=getattr(dataset_info.card_data, "license", None),
            python=platform.python_version(),
            gpu=torch.cuda.get_device_name(),
            cuda=torch.version.cuda,
            bf16_supported=torch.cuda.is_bf16_supported(),
            versions={
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "datasets", "peft", "accelerate")
            },
            seed=42,
            prompt_version=PROMPT_VERSION,
        )
        if not report["model_license"] or not report["dataset_license"]:
            raise RuntimeError("Missing upstream license metadata; review before proceeding")
        print(
            json.dumps(
                {
                    "stage": "provenance",
                    "model_revision": model_info.sha,
                    "dataset_revision": dataset_info.sha,
                }
            ),
            flush=True,
        )
        data = load_dataset(dataset_id, revision=dataset_info.sha, split="train")
        labels = data.features["label"].names
        if len(labels) != 77 or len(set(labels)) != 77:
            raise ValueError("Expected 77 unique canonical labels")
        (args.output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=model_info.sha)
        tokenizer.pad_token = tokenizer.eos_token
        encoded = []
        lengths = []
        for row in data:
            messages = build_messages(row["text"], labels)
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            target = json.dumps({"intent": labels[row["label"]]})
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": target}], tokenize=True
            )
            if full[: len(prefix)] != prefix or len(full) <= len(prefix):
                raise ValueError("Chat completion boundary is not a strict token prefix")
            lengths.append(len(full))
            if len(encoded) < args.examples:
                encoded.append((full, [-100] * len(prefix) + full[len(prefix) :]))
        report["token_lengths"] = {
            "min": min(lengths),
            "max": max(lengths),
            "over_256": sum(n > 256 for n in lengths),
            "training_rows": len(lengths),
        }
        report["mask_verified"] = all(
            any(v == -100 for v in y) and any(v != -100 for v in y) for _, y in encoded
        )
        report["sample_hash"] = hashlib.sha256(json.dumps(encoded).encode()).hexdigest()
        # Reuse one allocation shape instead of caching many large vocabulary-logit shapes.
        padded_length = max(len(x) for x, _ in encoded)
        report["padded_smoke_sequence_length"] = padded_length
        report["smoke_label_counts"] = {
            labels[label]: sum(row["label"] == label for row in data.select(range(len(encoded))))
            for label in sorted(set(data.select(range(len(encoded)))["label"]))
        }
        print(json.dumps({"stage": "token_lengths", **report["token_lengths"]}), flush=True)
        dtype = torch.bfloat16 if report["bf16_supported"] else torch.float16
        report["precision"] = str(dtype)

        def load_base():
            return AutoModelForCausalLM.from_pretrained(
                model_id, revision=model_info.sha, torch_dtype=dtype, attn_implementation="sdpa"
            ).to("cuda")

        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = load_base()
        report["load_seconds"] = time.perf_counter() - load_start
        prompt = tokenizer.apply_chat_template(
            build_messages(data[0]["text"], labels),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        def generate(current):
            current.eval()
            with torch.inference_mode():
                output = current.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return tokenizer.decode(output[0, prompt.shape[1] :], skip_special_tokens=True)

        report["base_output"] = generate(model)
        try:
            report["base_prediction"] = str(parse_prediction(report["base_output"], labels))
        except ValueError as exc:
            report["base_parse_error"] = str(exc)
        report["inference_peak_bytes"] = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        report["parameters"] = dict(
            zip(("trainable", "total"), model.get_nb_trainable_parameters(), strict=False)
        )
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        torch.cuda.reset_peak_memory_stats()
        train_start = time.perf_counter()
        losses = []
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro in range(args.accumulation):
                x, y = encoded[(step * args.accumulation + micro) % len(encoded)]
                padding = padded_length - len(x)
                batch = {
                    "input_ids": torch.tensor(
                        [x + [tokenizer.pad_token_id] * padding], device="cuda"
                    ),
                    "labels": torch.tensor([y + [-100] * padding], device="cuda"),
                    "attention_mask": torch.tensor([[1] * len(x) + [0] * padding], device="cuda"),
                }
                with torch.autocast("cuda", dtype=dtype):
                    loss = model(**batch).loss
                if not math.isfinite(loss.item()):
                    raise RuntimeError("Nonfinite training loss")
                loss_sum += loss.item() / args.accumulation
                scaler.scale(loss / args.accumulation).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                raise RuntimeError(
                    "FP16 overflow skipped optimizer update; retry with safer settings"
                )
            losses.append(loss_sum)
            print(json.dumps({"optimizer_step": step + 1, "loss": loss_sum}), flush=True)
        torch.cuda.synchronize()
        report.update(
            losses=losses,
            train_seconds=time.perf_counter() - train_start,
            train_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            train_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        adapter = args.output / "adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        del model, optimizer, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        reloaded = PeftModel.from_pretrained(load_base(), adapter)
        report["reloaded_output"] = generate(reloaded)
        report["adapter_reload_passed"] = True
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        if "torch" in locals() and torch.cuda.is_available():
            report["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
