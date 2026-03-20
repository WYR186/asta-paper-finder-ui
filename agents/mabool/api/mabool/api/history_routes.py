"""
Session-based search history backed by SQLite.

Schema:
  sessions  — one row per conversation thread
  searches  — one row per search within a session (many → one session)

DB file: <project_root>/search_history.db  (not committed to git)
"""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mabool.utils.paths import project_root

DB_PATH = project_root() / "search_history.db"

router = APIRouter(tags=["history"])


# ── DB bootstrap ──────────────────────────────────────────────────────────────

def _init_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                name       TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS searches (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                query      TEXT    NOT NULL,
                mode       TEXT,
                before_date TEXT,
                anchor_ids  TEXT,
                s2_session_id TEXT,
                result_count  INTEGER,
                result_json   TEXT
            );
        """)
        conn.commit()


@contextmanager
def _db():
    _init_db(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Pydantic models ───────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: str


class SessionRename(BaseModel):
    name: str


class SearchSave(BaseModel):
    query: str
    mode: str | None = None
    before_date: str | None = None
    anchor_ids: list[str] | None = None
    s2_session_id: str | None = None
    result_count: int | None = None
    result_json: Any = None


# ── Session routes ────────────────────────────────────────────────────────────

@router.post("/api/sessions", status_code=201)
def create_session(req: SessionCreate) -> JSONResponse:
    with _db() as conn:
        cur = conn.execute("INSERT INTO sessions (name) VALUES (?)", (req.name,))
        row = conn.execute(
            "SELECT id, created_at, name FROM sessions WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
    return JSONResponse(dict(row), status_code=201)


@router.get("/api/sessions")
def list_sessions() -> JSONResponse:
    with _db() as conn:
        rows = conn.execute(
            """SELECT s.id, s.created_at, s.name,
                      COUNT(sr.id) AS search_count
               FROM sessions s
               LEFT JOIN searches sr ON sr.session_id = s.id
               GROUP BY s.id
               ORDER BY s.id DESC"""
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@router.patch("/api/sessions/{session_id}")
def rename_session(session_id: int, req: SessionRename) -> JSONResponse:
    with _db() as conn:
        n = conn.execute(
            "UPDATE sessions SET name = ? WHERE id = ?", (req.name, session_id)
        ).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse({"ok": True})


@router.delete("/api/sessions/{session_id}", status_code=204)
def delete_session(session_id: int) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ── Search routes (within a session) ─────────────────────────────────────────

@router.post("/api/sessions/{session_id}/searches", status_code=201)
def save_search(session_id: int, req: SearchSave) -> JSONResponse:
    with _db() as conn:
        # Atomic: verify session exists and insert in one transaction
        cur = conn.execute(
            """INSERT INTO searches
               (session_id, query, mode, before_date, anchor_ids,
                s2_session_id, result_count, result_json)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?
               WHERE EXISTS (SELECT 1 FROM sessions WHERE id = ?)""",
            (
                session_id,
                req.query,
                req.mode,
                req.before_date,
                json.dumps(req.anchor_ids) if req.anchor_ids else None,
                req.s2_session_id,
                req.result_count,
                json.dumps(req.result_json) if req.result_json is not None else None,
                session_id,
            ),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse({"id": cur.lastrowid}, status_code=201)


@router.get("/api/sessions/{session_id}/searches")
def list_searches(session_id: int) -> JSONResponse:
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, created_at, query, mode, before_date, anchor_ids,
                      s2_session_id, result_count, result_json
               FROM searches WHERE session_id = ? ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "created_at": r["created_at"],
            "query": r["query"],
            "mode": r["mode"],
            "before_date": r["before_date"],
            "anchor_ids": json.loads(r["anchor_ids"]) if r["anchor_ids"] else [],
            "s2_session_id": r["s2_session_id"],
            "result_count": r["result_count"],
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
        })
    return JSONResponse(items)


@router.delete("/api/sessions/{session_id}/searches/{search_id}", status_code=204)
def delete_search(session_id: int, search_id: int) -> None:
    with _db() as conn:
        conn.execute(
            "DELETE FROM searches WHERE id = ? AND session_id = ?",
            (search_id, session_id),
        )
