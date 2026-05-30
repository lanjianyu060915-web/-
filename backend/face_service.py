"""Face feature extraction helpers.

This module intentionally contains a replaceable placeholder implementation.
InsightFace or other face-recognition models should be integrated here in a
future phase without changing the API layer.
"""

from __future__ import annotations

import hashlib

MOCK_EMBEDDING_SIZE = 128


def extract_embedding(image: bytes) -> list[float]:
    """Return a deterministic mock face embedding for uploaded image bytes.

    The current project phase must not download or load face-recognition models,
    so this function derives a stable pseudo-vector from the image content. It is
    deliberately isolated behind this function so it can be replaced by a real
    InsightFace implementation later.
    """
    digest = hashlib.sha256(image).digest()
    values: list[float] = []

    for index in range(MOCK_EMBEDDING_SIZE):
        byte = digest[index % len(digest)]
        # Map bytes from [0, 255] into a small floating-point range [-1, 1].
        values.append(round((byte / 127.5) - 1.0, 6))

    return values
