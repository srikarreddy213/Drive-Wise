"""
logger.py — SQLite-based query and response logging for DriveWise.

Records every query with: question, answer, timing, eval scores, and status.
Data is used by the Analytics Dashboard (Tab 3 in the UI).
"""

import sqlite3
import json
from datetime import datetime
from src.config import LOGS_DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(LOGS_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """Create query_logs table if it doesn't exist. Call once at startup."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            car_brand           TEXT    NOT NULL,
            car_model           TEXT    NOT NULL,
            user_query          TEXT    NOT NULL,
            response            TEXT    NOT NULL,
            sources             TEXT,
            response_time_sec   REAL,
            context_relevance   REAL,
            faithfulness        REAL,
            answer_correctness  REAL,
            status              TEXT    DEFAULT 'SUCCESS'
        )
    """)
    conn.commit()
    conn.close()


def log_query(
    car_brand: str,
    car_model: str,
    user_query: str,
    response: str,
    sources: list,
    response_time_sec: float,
    context_relevance: float | None = None,
    faithfulness: float | None = None,
    answer_correctness: float | None = None,
    status: str = "SUCCESS",
) -> int:
    """Insert one query log row. Returns the new row id."""
    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO query_logs
            (timestamp, car_brand, car_model, user_query, response,
             sources, response_time_sec, context_relevance,
             faithfulness, answer_correctness, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now().isoformat(),
            car_brand, car_model, user_query, response,
            json.dumps(sources or []),
            response_time_sec,
            context_relevance, faithfulness, answer_correctness,
            status,
        ),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def update_eval_scores(
    row_id: int,
    context_relevance: float,
    faithfulness: float,
    answer_correctness: float,
) -> None:
    """Update eval scores for an existing log row (called after async eval)."""
    conn = _connect()
    conn.execute(
        """
        UPDATE query_logs
        SET context_relevance = ?,
            faithfulness      = ?,
            answer_correctness = ?
        WHERE id = ?
        """,
        (context_relevance, faithfulness, answer_correctness, row_id),
    )
    conn.commit()
    conn.close()


def get_all_logs() -> list[dict]:
    """Return all log rows as dicts (newest first), for the analytics tab."""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM query_logs ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    """Aggregate statistics for the analytics KPI cards."""
    conn = _connect()
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                                      AS total,
            AVG(response_time_sec)                                        AS avg_time,
            AVG(context_relevance)                                        AS avg_cr,
            AVG(faithfulness)                                             AS avg_faith,
            AVG(answer_correctness)                                       AS avg_ac,
            SUM(CASE WHEN status = 'NO_ANSWER_FOUND' THEN 1 ELSE 0 END)  AS failed,
            SUM(CASE WHEN status = 'SUCCESS'          THEN 1 ELSE 0 END)  AS success
        FROM query_logs
        """
    ).fetchone()
    conn.close()

    if not row or not row["total"]:
        return {
            "total_queries": 0, "avg_response_time": 0.0,
            "avg_context_relevance": 0.0, "avg_faithfulness": 0.0,
            "avg_answer_correctness": 0.0,
            "failed_count": 0, "success_count": 0,
        }
    return {
        "total_queries":          row["total"]       or 0,
        "avg_response_time":      round(row["avg_time"]   or 0.0, 2),
        "avg_context_relevance":  round(row["avg_cr"]     or 0.0, 2),
        "avg_faithfulness":       round(row["avg_faith"]  or 0.0, 2),
        "avg_answer_correctness": round(row["avg_ac"]     or 0.0, 2),
        "failed_count":           row["failed"]      or 0,
        "success_count":          row["success"]     or 0,
    }
