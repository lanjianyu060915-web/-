"""Face embedding helpers.

This module intentionally uses a lightweight placeholder implementation for now.
The next stage can replace ``extract_embedding`` with a real face recognition
model while keeping the API and database flow stable.
"""

from __future__ import annotations

import hashlib

EMBEDDING_DIMENSION = 128


def extract_embedding(image: bytes) -> list[float]:
    """Return a deterministic mock embedding for uploaded image bytes.

    The placeholder hashes the image content and expands it into a normalized
    vector. It does not perform real face detection or recognition.
    """
    if not image:
        raise ValueError("图片内容为空")

    digest = hashlib.sha256(image).digest()
    values = [digest[index % len(digest)] / 255 for index in range(EMBEDDING_DIMENSION)]
    magnitude = sum(value * value for value in values) ** 0.5 or 1
    return [round(value / magnitude, 8) for value in values]
