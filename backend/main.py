"""FastAPI application for the face recognition course backend."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from database import DATABASE_PATH, init_db, iter_connection
from face_service import extract_embedding
from schemas import (
    FaceEmbedding,
    HealthResponse,
    Person,
    PersonCreate,
    RecognitionRecord,
    RecognitionResult,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Face Recognition System API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Connection = Annotated[sqlite3.Connection, Depends(iter_connection)]

RECOGNITION_THRESHOLD = 0.84
UNKNOWN_NAME = "未知"


def calculate_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for two embedding vectors."""
    if len(left) != len(right):
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", database=str(DATABASE_PATH))


@app.get("/api/persons", response_model=list[Person])
def list_persons(connection: Connection) -> list[dict]:
    cursor = connection.execute(
        """
        SELECT id, name, person_code, department, note, created_at
        FROM persons
        ORDER BY id DESC
        """
    )
    return [dict(row) for row in cursor.fetchall()]


@app.post("/api/persons", response_model=Person, status_code=status.HTTP_201_CREATED)
def create_person(payload: PersonCreate, connection: Connection) -> dict:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="姓名不能为空")

    cursor = connection.execute(
        """
        INSERT INTO persons (name, person_code, department, note)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            payload.person_code.strip() if payload.person_code else None,
            payload.department.strip() if payload.department else None,
            payload.note.strip() if payload.note else None,
        ),
    )
    connection.commit()

    created = connection.execute(
        """
        SELECT id, name, person_code, department, note, created_at
        FROM persons
        WHERE id = ?
        """,
        (cursor.lastrowid,),
    ).fetchone()
    if created is None:
        raise HTTPException(status_code=500, detail="人员创建失败")
    return dict(created)


@app.post(
    "/api/persons/{person_id}/faces",
    response_model=list[FaceEmbedding],
    status_code=status.HTTP_201_CREATED,
    summary="注册人员人脸照片",
    description="为指定人员上传一张或多张人脸照片，并保存当前阶段的占位人脸特征。",
)
def register_person_faces(
    person_id: int,
    connection: Connection,
    files: list[UploadFile] = File(...),
) -> list[dict]:
    person = connection.execute(
        "SELECT id FROM persons WHERE id = ?",
        (person_id,),
    ).fetchone()
    if person is None:
        raise HTTPException(status_code=404, detail="人员不存在")

    if not files:
        raise HTTPException(status_code=422, detail="请至少上传一张人脸照片")

    created_faces: list[dict] = []
    for uploaded_file in files:
        image_bytes = uploaded_file.file.read()
        if not image_bytes:
            file_name = uploaded_file.filename or "未命名文件"
            raise HTTPException(status_code=422, detail=f"文件 {file_name} 为空")

        embedding = extract_embedding(image_bytes)
        cursor = connection.execute(
            """
            INSERT INTO face_embeddings (person_id, file_name, content_type, embedding_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                person_id,
                uploaded_file.filename,
                uploaded_file.content_type,
                json.dumps(embedding),
            ),
        )

        created_faces.append(
            {
                "id": cursor.lastrowid,
                "person_id": person_id,
                "file_name": uploaded_file.filename,
                "content_type": uploaded_file.content_type,
                "embedding": embedding,
            }
        )

    connection.commit()

    created_ids = [face["id"] for face in created_faces]
    placeholders = ",".join("?" for _ in created_ids)
    cursor = connection.execute(
        f"""
        SELECT id, person_id, file_name, content_type, embedding_json, created_at
        FROM face_embeddings
        WHERE id IN ({placeholders})
        ORDER BY id
        """,
        created_ids,
    )
    return [
        {
            "id": row["id"],
            "person_id": row["person_id"],
            "file_name": row["file_name"],
            "content_type": row["content_type"],
            "embedding": json.loads(row["embedding_json"]),
            "created_at": row["created_at"],
        }
        for row in cursor.fetchall()
    ]


@app.post(
    "/api/recognize",
    response_model=RecognitionResult,
    summary="识别人脸照片",
    description="上传一张图片，使用当前阶段的占位人脸特征与已注册特征逐个比对。",
)
def recognize_face(
    connection: Connection,
    file: UploadFile = File(...),
) -> dict:
    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="识别图片不能为空")

    query_embedding = extract_embedding(image_bytes)
    rows = connection.execute(
        """
        SELECT
            face_embeddings.embedding_json,
            persons.id AS person_id,
            persons.name,
            persons.person_code
        FROM face_embeddings
        JOIN persons ON persons.id = face_embeddings.person_id
        ORDER BY face_embeddings.id
        """
    ).fetchall()

    best_match: dict | None = None
    best_similarity = 0.0
    for row in rows:
        stored_embedding = json.loads(row["embedding_json"])
        similarity = calculate_similarity(query_embedding, stored_embedding)
        if best_match is None or similarity > best_similarity:
            best_similarity = similarity
            best_match = dict(row)

    is_recognized = best_match is not None and best_similarity >= RECOGNITION_THRESHOLD
    result = {
        "name": best_match["name"] if is_recognized and best_match else UNKNOWN_NAME,
        "person_code": best_match["person_code"] if is_recognized and best_match else None,
        "similarity": round(best_similarity, 6) if best_match else None,
        "status": "recognized" if is_recognized else "unknown",
    }

    connection.execute(
        """
        INSERT INTO recognition_records (
            person_id, person_name, person_code, source_name, similarity, status
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            best_match["person_id"] if is_recognized and best_match else None,
            result["name"],
            result["person_code"],
            file.filename,
            result["similarity"],
            result["status"],
        ),
    )
    connection.commit()

    return result


@app.delete(
    "/api/persons/{person_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
)
def delete_person(person_id: int, connection: Connection) -> None:
    cursor = connection.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="人员不存在")


@app.get("/api/records", response_model=list[RecognitionRecord])
def list_records(connection: Connection) -> list[dict]:
    cursor = connection.execute(
        """
        SELECT id, person_id, person_name, person_code, source_name, similarity, status, created_at
        FROM recognition_records
        ORDER BY id DESC
        LIMIT 50
        """
    )
    return [dict(row) for row in cursor.fetchall()]
