"""Matched greedy validation generation for trained adapters."""

import json
import time

from domaintune.evaluate import evaluation_report
from domaintune.parse import InvalidPrediction, parse_prediction
from domaintune.prompts import build_messages


class EvaluationStopped(Exception):
    """Validation interrupted deliberately; partial predictions remain evidence."""


def evaluate_adapter(model, tokenizer, rows, labels, destination, stop_requested=None):
    import torch

    model.eval()
    predictions, outputs, latencies = [], [], []
    with destination.open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows):
            if stop_requested and stop_requested():
                raise EvaluationStopped("Validation stopped at an example boundary")
            encoded = tokenizer.apply_chat_template(
                build_messages(row["text"], labels),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=48,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            output = tokenizer.decode(
                generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            try:
                prediction = parse_prediction(output, labels)
            except InvalidPrediction:
                prediction = None
            predictions.append(prediction)
            outputs.append(output)
            log.write(
                json.dumps(
                    {"sample_id": row["sample_id"], "output": output, "prediction": prediction}
                )
                + "\n"
            )
            log.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"validation_completed": index + 1}), flush=True)
    report = evaluation_report(rows, predictions, labels, outputs)
    report["generation_seconds"] = sum(latencies)
    return report
