"""SQLite helpers for the face recognition course backend."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "face_system.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection configured for dictionary-like rows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def iter_connection() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency that yields a database connection."""
    connection = get_connection()
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    """Create the initial database schema if it does not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                person_code TEXT,
                department TEXT,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS recognition_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                person_name TEXT NOT NULL,
                person_code TEXT,
                source_name TEXT,
                similarity REAL,
                status TEXT NOT NULL DEFAULT 'unknown',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                content_type TEXT,
                embedding_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );
            """
        )
