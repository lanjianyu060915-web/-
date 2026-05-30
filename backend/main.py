"""FastAPI application for the face recognition course backend."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from database import DATABASE_PATH, init_db, iter_connection
from face_service import extract_embedding
from schemas import FaceRegistrationResponse, HealthResponse, Person, PersonCreate, RecognitionRecord


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
    response_model=FaceRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_person_faces(
    person_id: int,
    connection: Connection,
    files: list[UploadFile] = File(...),
) -> dict:
    person = connection.execute("SELECT id FROM persons WHERE id = ?", (person_id,)).fetchone()
    if person is None:
        raise HTTPException(status_code=404, detail="人员不存在")

    if not files:
        raise HTTPException(status_code=422, detail="请至少上传一张人脸照片")

    registered_faces = []
    for upload in files:
        if upload.content_type and not upload.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{upload.filename} 不是图片文件")

        image_bytes = upload.file.read()
        try:
            embedding = extract_embedding(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cursor = connection.execute(
            """
            INSERT INTO face_embeddings (person_id, file_name, content_type, embedding_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                person_id,
                upload.filename or "uploaded-face",
                upload.content_type,
                json.dumps(embedding),
            ),
        )
        created = connection.execute(
            """
            SELECT id, person_id, file_name, content_type, embedding_json, created_at
            FROM face_embeddings
            WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
        if created is not None:
            row = dict(created)
            registered_faces.append(
                {
                    "id": row["id"],
                    "person_id": row["person_id"],
                    "file_name": row["file_name"],
                    "content_type": row["content_type"],
                    "embedding_dimension": len(json.loads(row["embedding_json"])),
                    "created_at": row["created_at"],
                }
            )

    connection.commit()
    return {
        "person_id": person_id,
        "registered_count": len(registered_faces),
        "faces": registered_faces,
    }


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
        """
    )
    return [dict(row) for row in cursor.fetchall()]
