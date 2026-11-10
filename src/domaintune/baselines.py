"""Validation-only baselines; never load the official test partition."""

import hashlib
import json
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from domaintune.evaluate import evaluation_report


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_data(root):
    provenance = json.loads((root / "data/manifests/provenance.json").read_text("utf-8"))
    labels = provenance["labels"]
    partitions = {}
    hashes = {}
    for split in ("train", "validation"):
        path = root / f"data/processed/{split}.jsonl"
        hashes[split] = sha256(path)
        if hashes[split] != provenance["processed_sha256"][split]:
            raise ValueError(f"Frozen {split} checksum mismatch")
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        manifest = (root / f"data/manifests/{split}.jsonl").read_text("utf-8").splitlines()
        if len(rows) != provenance["split_rows"][split] or len(manifest) != len(rows):
            raise ValueError("Frozen row count mismatch")
        for row, entry in zip(rows, manifest, strict=True):
            if {k: v for k, v in row.items() if k != "text"} != json.loads(entry):
                raise ValueError("Manifest mismatch")
            if labels[row["intent_id"]] != row["intent_name"]:
                raise ValueError("Label mismatch")
        partitions[split] = rows
    return partitions, labels, hashes


def classical_baselines(data, labels, config):
    train, validation = data["train"], data["validation"]
    counts = Counter(r["intent_name"] for r in train)
    majority = max(labels, key=lambda label: counts[label])
    reports = {"majority": evaluation_report(validation, [majority] * len(validation), labels)}
    reports["majority"]["selected_label"] = majority
    candidates = []
    for analyzer in config["tfidf"]["analyzers"]:
        for c in config["tfidf"]["C"]:
            started = time.perf_counter()
            pipeline = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            analyzer=analyzer,
                            ngram_range=(1, 2) if analyzer == "word" else (3, 5),
                            sublinear_tf=True,
                            min_df=2,
                        ),
                    ),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c,
                            max_iter=1000,
                            random_state=config["seed"],
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            pipeline.fit([r["text"] for r in train], [r["intent_name"] for r in train])
            predictions = pipeline.predict([r["text"] for r in validation]).tolist()
            report = evaluation_report(validation, predictions, labels)
            report.update(analyzer=analyzer, C=c, elapsed_seconds=time.perf_counter() - started)
            candidates.append(report)
            print(json.dumps({k: report[k] for k in ("analyzer", "C", "macro_f1")}), flush=True)
    best = max(candidates, key=lambda r: r["macro_f1"])
    reports["tfidf_logistic_regression"] = best
    reports["tfidf_candidates"] = [
        {k: r[k] for k in ("analyzer", "C", "accuracy", "macro_f1", "elapsed_seconds")}
        for r in candidates
    ]
    return reports
