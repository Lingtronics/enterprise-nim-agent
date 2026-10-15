"""Small deterministic lexical embeddings for a reproducible offline reference runtime.

Hashing features are a transparent baseline, not a substitute for evaluating a production
embedding model. No model weights are downloaded and Python's randomized hash is never used.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

DIMENSIONS = 384
EMBEDDING_VERSION = "sha256-word-bigram-v1"
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has how in is it of on or "
    "please should summarise summarize tell the their this to was what which with".split()
)


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # A small, explicit plural rule helps requirements/environment queries without
    # downloading an NLP tokenizer. Preserve words such as status and Kubernetes.
    return [
        word[:-1]
        if len(word) > 4
        and word.endswith("s")
        and not word.endswith("ss")
        and word not in {"status", "kubernetes"}
        else word
        for word in words
        if word not in _STOPWORDS
    ]


class HashEmbeddings:
    dimensions = DIMENSIONS
    version = EMBEDDING_VERSION

    def embed(self, text: str) -> np.ndarray:
        words = _tokens(text)
        features = Counter(f"word:{word}" for word in words)
        features.update(f"bigram:{left} {right}" for left, right in zip(words, words[1:]))
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, count in features.items():
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimensions
            sign = 1 if digest[8] & 1 else -1
            weight = 0.5 if feature.startswith("bigram:") else 1.0
            vector[index] += sign * weight * (1 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(text) for text in texts]
