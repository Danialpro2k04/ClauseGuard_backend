import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "pending_reviews.db")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                contract_name TEXT NOT NULL,
                clause_text TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                justification TEXT,
                status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);")


def add_review(contract_name: str, clause_text: str, risk_level: str, justification: str, session_id: str = None) -> str:

    review_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO reviews (id, session_id, contract_name, clause_text, risk_level, justification, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                review_id,
                session_id,
                contract_name,
                clause_text,
                risk_level.upper(),
                justification,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return review_id


def list_reviews(status: str = None) -> list[dict]:
    with _connect() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM reviews WHERE status = ? ORDER BY created_at ASC", (status.upper(),)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM reviews ORDER BY created_at ASC").fetchall()
        return [dict(row) for row in rows]


def resolve_review(review_id: str) -> bool:

    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE reviews SET status = 'RESOLVED' WHERE id = ? AND status = 'PENDING'",
            (review_id,),
        )
        return cursor.rowcount > 0


def delete_review(review_id: str) -> bool:
    """Hard-deletes a review record by its stable id."""
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        return cursor.rowcount > 0


def clear_session_reviews(session_id: str = None) -> int:
 
    with _connect() as conn:
        if session_id:
            cursor = conn.execute("DELETE FROM reviews WHERE session_id = ?", (session_id,))
        else:
            cursor = conn.execute("DELETE FROM reviews")
        return cursor.rowcount


init_db()