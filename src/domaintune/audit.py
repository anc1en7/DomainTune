"""Bounded sparse character-gram near-duplicate audit, never used for selection."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from domaintune.data import normalize


def near_duplicates(splits, threshold=0.9):
    rows = [dict(r, split=s) for s in ("train", "validation", "test") for r in splits[s]]
    # Fixed audit-only transform, independent of any modeling vectorizer.
    matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32).fit_transform(
        [normalize(r["text"]) for r in rows]
    )
    pairs = []
    for start in range(0, len(rows), 128):
        scores = (matrix[start : start + 128] @ matrix.T).tocoo()
        selected = scores.data >= threshold
        for i, j, score in zip(
            scores.row[selected], scores.col[selected], scores.data[selected], strict=True
        ):
            i = int(i) + start
            j = int(j)
            if i < j and rows[i]["split"] != rows[j]["split"] and score >= threshold:
                pairs.append(
                    {
                        "left_id": rows[i]["sample_id"],
                        "right_id": rows[j]["sample_id"],
                        "left_split": rows[i]["split"],
                        "right_split": rows[j]["split"],
                        "cosine": float(score),
                        "same_label": rows[i]["intent_id"] == rows[j]["intent_id"],
                    }
                )
    return pairs
