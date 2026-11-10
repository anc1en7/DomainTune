"""Prepare pinned BANKING77 data and metadata; requires the ML extra."""

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path

import yaml

from domaintune.audit import near_duplicates
from domaintune.data import canonical_json, digest, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if root.drive.upper() != "D:" or not os.environ.get("HF_HOME", "").startswith(str(root)):
        raise RuntimeError("Dot-source scripts/environment.ps1 before running")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config["dataset_revision"]):
        raise ValueError("Dataset revision must be an immutable commit")
    if config["test_policy"] != "preserve_official_test":
        raise ValueError("Official test must be preserved")
    from datasets import load_dataset

    raw = load_dataset(config["dataset_id"], revision=config["dataset_revision"])
    # Offline datasets may fall back to the latest cached build. Fail closed if
    # its cache path does not identify the requested immutable source revision.
    for partition in raw.values():
        if not partition.cache_files or any(
            config["dataset_revision"] not in Path(f["filename"]).parts
            for f in partition.cache_files
        ):
            raise ValueError("Cannot verify pinned dataset cache revision")
    labels = raw["train"].features["label"].names
    if raw["test"].features["label"].names != labels:
        raise ValueError("Inconsistent label mapping")
    expected = json.loads((root / "reports/phase0-labels.json").read_text(encoding="utf-8"))
    if labels != expected or (len(raw["train"]), len(raw["test"])) != (10003, 3080):
        raise ValueError("Pinned dataset changed")
    splits, report = prepare(
        raw, labels, config["dataset_revision"], config["seed"], config["validation_fraction"]
    )
    manifest_dir = root / "data/manifests"
    processed = root / "data/processed"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    contents = {s: "".join(canonical_json(r) + "\n" for r in rows) for s, rows in splits.items()}
    checksums = {s: digest(content) for s, content in contents.items()}
    expected_provenance = manifest_dir / "provenance.json"
    if expected_provenance.exists():
        previous = json.loads(expected_provenance.read_text(encoding="utf-8"))
        if previous["processed_sha256"] != checksums:
            raise ValueError("Prepared split hashes differ from frozen provenance")
    for split, rows in splits.items():
        content = contents[split]
        (processed / f"{split}.jsonl").write_text(content, encoding="utf-8", newline="\n")
        manifest = [{k: v for k, v in r.items() if k != "text"} for r in rows]
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in manifest), encoding="utf-8", newline="\n"
        )
    pairs = near_duplicates(splits)
    report.update(
        config=config,
        processed_sha256=checksums,
        labels=labels,
        dataset_license="CC-BY-4.0",
        citation="Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence Encoders",
        near_duplicate_threshold=0.9,
        near_duplicate_pairs=pairs,
        versions={n: importlib.metadata.version(n) for n in ("datasets", "scikit-learn", "numpy")},
    )
    (manifest_dir / "provenance.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("split_rows", "exclusion_reasons", "duplicate_groups", "unicode")
            }
        )
    )
    print(f"Cross-split near-duplicate candidates: {len(pairs)}")


if __name__ == "__main__":
    main()
