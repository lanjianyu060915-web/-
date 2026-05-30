"""Pydantic schemas used by the FastAPI backend."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    database: str


class PersonCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    person_code: Optional[str] = Field(default=None, max_length=50)
    department: Optional[str] = Field(default=None, max_length=100)
    note: Optional[str] = Field(default=None, max_length=255)


class Person(BaseModel):
    id: int
    name: str
    person_code: Optional[str] = None
    department: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime


class RecognitionRecord(BaseModel):
    id: int
    person_id: Optional[int] = None
    person_name: str
    person_code: Optional[str] = None
    source_name: Optional[str] = None
    similarity: Optional[float] = None
    status: str
    created_at: datetime


class RecognitionResult(BaseModel):
    name: str
    person_code: Optional[str] = None
    similarity: Optional[float] = None
    status: str


class FaceEmbedding(BaseModel):
    id: int
    person_id: int
    file_name: Optional[str] = None
    content_type: Optional[str] = None
    embedding: list[float]
    created_at: datetime
