"""Face detection and feature extraction helpers based on InsightFace.

The API layer imports this module at application startup, so heavyweight and
optional ML dependencies are imported lazily. If InsightFace, ONNX Runtime,
OpenCV, model download, or model initialization is unavailable, callers receive a
clear service error instead of crashing the FastAPI process.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


class FaceServiceError(RuntimeError):
    """Base error for face-service failures that should be shown to API users."""


class FaceModelUnavailableError(FaceServiceError):
    """Raised when InsightFace or its model cannot be initialized."""


class InvalidFaceImageError(FaceServiceError):
    """Raised when uploaded bytes cannot be decoded as an image."""


class NoFaceDetectedError(FaceServiceError):
    """Raised when the model runs successfully but finds no face."""


@dataclass(frozen=True)
class FaceEmbeddingResult:
    """One detected face and its embedding."""

    embedding: list[float]
    bbox: tuple[float, float, float, float]
    area: float


_FACE_APP: Any | None = None
_FACE_APP_ERROR: str | None = None
_FACE_APP_LOCK = Lock()
_MODEL_NAME = "buffalo_l"
_DETECTION_SIZE = (640, 640)
_PROVIDERS = ["CPUExecutionProvider"]


def _get_face_app() -> Any:
    """Lazily initialize and cache the InsightFace application."""
    global _FACE_APP, _FACE_APP_ERROR

    if _FACE_APP is not None:
        return _FACE_APP
    if _FACE_APP_ERROR is not None:
        raise FaceModelUnavailableError(_FACE_APP_ERROR)

    with _FACE_APP_LOCK:
        if _FACE_APP is not None:
            return _FACE_APP
        if _FACE_APP_ERROR is not None:
            raise FaceModelUnavailableError(_FACE_APP_ERROR)

        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=_MODEL_NAME, providers=_PROVIDERS)
            app.prepare(ctx_id=-1, det_size=_DETECTION_SIZE)
        except Exception as exc:  # noqa: BLE001 - convert dependency/model errors into API-safe text.
            _FACE_APP_ERROR = (
                "人脸识别模型初始化失败：请确认 insightface、onnxruntime、opencv-python "
                "已正确安装，并且首次运行时能够下载 InsightFace 模型。"
                f" 原始错误：{exc}"
            )
            raise FaceModelUnavailableError(_FACE_APP_ERROR) from exc

        _FACE_APP = app
        return _FACE_APP


def _decode_image(image: bytes) -> Any:
    """Decode uploaded image bytes into an OpenCV BGR image array."""
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - dependency availability is runtime environment dependent.
        raise FaceModelUnavailableError(
            "图片解码依赖不可用：请安装 opencv-python 和 numpy 后重启服务。"
            f" 原始错误：{exc}"
        ) from exc

    buffer = np.frombuffer(image, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise InvalidFaceImageError("无法解析上传图片，请上传有效的 JPG、PNG 等图片文件")
    return decoded


def _face_area(face: Any) -> float:
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) < 4:
        return 0.0

    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _embedding_from_face(face: Any) -> list[float]:
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
    if embedding is None:
        raise FaceModelUnavailableError("模型未返回人脸 embedding，请检查 InsightFace 模型是否完整")

    return [float(value) for value in embedding]


def _result_from_face(face: Any) -> FaceEmbeddingResult:
    bbox_values = getattr(face, "bbox", [0.0, 0.0, 0.0, 0.0])[:4]
    bbox = tuple(float(value) for value in bbox_values)
    if len(bbox) != 4:
        bbox = (0.0, 0.0, 0.0, 0.0)

    return FaceEmbeddingResult(
        embedding=_embedding_from_face(face),
        bbox=bbox,
        area=_face_area(face),
    )


def extract_embeddings(image: bytes) -> list[FaceEmbeddingResult]:
    """Detect all faces in an image and return each face embedding.

    Raises clear FaceServiceError subclasses for invalid images, no face found,
    or unavailable model/dependencies.
    """
    decoded_image = _decode_image(image)
    app = _get_face_app()
    faces = app.get(decoded_image)
    if not faces:
        raise NoFaceDetectedError("未检测到人脸，请上传包含清晰正脸的照片")

    return [_result_from_face(face) for face in faces]


def extract_embedding(image: bytes) -> list[float]:
    """Return the embedding for the largest detected face in an image.

    This preserves the older API used by registration code while replacing the
    placeholder vector with a real InsightFace embedding. When multiple faces are
    present, the face with the largest bounding-box area is selected.
    """
    faces = extract_embeddings(image)
    largest_face = max(faces, key=lambda face: face.area)
    return largest_face.embedding
