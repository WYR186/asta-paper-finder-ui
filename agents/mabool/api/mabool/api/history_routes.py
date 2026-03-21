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
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                query            TEXT    NOT NULL,
                mode             TEXT,
                before_date      TEXT,
                anchor_ids       TEXT,
                s2_session_id    TEXT,
                result_count     INTEGER,
                result_json      TEXT,
                parent_search_id INTEGER REFERENCES searches(id) ON DELETE SET NULL
            );
            CREATE TABLE IF NOT EXISTS bookmarks (
                corpus_id  TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                title      TEXT,
                authors    TEXT,
                year       INTEGER,
                venue      TEXT,
                url        TEXT,
                abstract   TEXT,
                tags       TEXT DEFAULT '[]',
                note       TEXT DEFAULT '',
                status     TEXT DEFAULT 'unread'
            );
            CREATE TABLE IF NOT EXISTS paper_notes (
                corpus_id  TEXT PRIMARY KEY,
                note       TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        conn.commit()

    # Migrate: add parent_search_id if it doesn't exist yet (existing DBs)
    with sqlite3.connect(path) as conn:
        try:
            conn.execute("ALTER TABLE searches ADD COLUMN parent_search_id INTEGER REFERENCES searches(id) ON DELETE SET NULL")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Migrate: add status column to bookmarks (existing DBs)
    with sqlite3.connect(path) as conn:
        try:
            conn.execute("ALTER TABLE bookmarks ADD COLUMN status TEXT DEFAULT 'unread'")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists


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
    parent_search_id: int | None = None


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
                s2_session_id, result_count, result_json, parent_search_id)
               SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                req.parent_search_id,
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
                      s2_session_id, result_count, result_json, parent_search_id
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
            "parent_search_id": r["parent_search_id"],
        })
    return JSONResponse(items)


@router.delete("/api/sessions/{session_id}/searches/{search_id}", status_code=204)
def delete_search(session_id: int, search_id: int) -> None:
    with _db() as conn:
        conn.execute(
            "DELETE FROM searches WHERE id = ? AND session_id = ?",
            (search_id, session_id),
        )


# ── Bookmark models ───────────────────────────────────────────────────────────

class BookmarkCreate(BaseModel):
    corpus_id: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str | None = None


_VALID_STATUSES = {"unread", "reading", "read", "to_review"}

class BookmarkPatch(BaseModel):
    tags: list[str] | None = None
    note: str | None = None
    status: str | None = None  # "unread" | "reading" | "read" | "to_review"


class NoteUpsert(BaseModel):
    note: str


# ── Bookmark routes ───────────────────────────────────────────────────────────

@router.post("/api/bookmarks", status_code=201)
def add_bookmark(req: BookmarkCreate) -> JSONResponse:
    with _db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO bookmarks
               (corpus_id, title, authors, year, venue, url, abstract)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                req.corpus_id,
                req.title,
                json.dumps(req.authors) if req.authors is not None else None,
                req.year,
                req.venue,
                req.url,
                req.abstract,
            ),
        )
    return JSONResponse({"ok": True}, status_code=201)


@router.get("/api/bookmarks/ids")
def list_bookmark_ids() -> JSONResponse:
    with _db() as conn:
        rows = conn.execute("SELECT corpus_id FROM bookmarks").fetchall()
    return JSONResponse({"ids": [r["corpus_id"] for r in rows]})


@router.get("/api/bookmarks")
def list_bookmarks() -> JSONResponse:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM bookmarks ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        item = dict(r)
        if item.get("authors"):
            try:
                item["authors"] = json.loads(item["authors"])
            except Exception:
                pass
        if item.get("tags"):
            try:
                item["tags"] = json.loads(item["tags"])
            except Exception:
                item["tags"] = []
        result.append(item)
    return JSONResponse(result)


@router.delete("/api/bookmarks/{corpus_id}", status_code=204)
def delete_bookmark(corpus_id: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM bookmarks WHERE corpus_id = ?", (corpus_id,))


@router.patch("/api/bookmarks/{corpus_id}")
def patch_bookmark(corpus_id: str, req: BookmarkPatch) -> JSONResponse:
    with _db() as conn:
        if req.tags is not None:
            conn.execute(
                "UPDATE bookmarks SET tags = ? WHERE corpus_id = ?",
                (json.dumps(req.tags), corpus_id),
            )
        if req.note is not None:
            conn.execute(
                "UPDATE bookmarks SET note = ? WHERE corpus_id = ?",
                (req.note, corpus_id),
            )
        if req.status is not None:
            if req.status not in _VALID_STATUSES:
                raise HTTPException(status_code=422, detail=f"Invalid status. Must be one of: {', '.join(_VALID_STATUSES)}")
            conn.execute(
                "UPDATE bookmarks SET status = ? WHERE corpus_id = ?",
                (req.status, corpus_id),
            )
        row = conn.execute(
            "SELECT * FROM bookmarks WHERE corpus_id = ?", (corpus_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    item = dict(row)
    if item.get("authors"):
        try:
            item["authors"] = json.loads(item["authors"])
        except Exception:
            pass
    if item.get("tags"):
        try:
            item["tags"] = json.loads(item["tags"])
        except Exception:
            item["tags"] = []
    return JSONResponse(item)


# ── Note routes ───────────────────────────────────────────────────────────────

@router.put("/api/notes/{corpus_id}")
def upsert_note(corpus_id: str, req: NoteUpsert) -> JSONResponse:
    with _db() as conn:
        conn.execute(
            """INSERT INTO paper_notes (corpus_id, note, updated_at) VALUES (?, ?, datetime('now'))
               ON CONFLICT(corpus_id) DO UPDATE SET note = excluded.note, updated_at = datetime('now')""",
            (corpus_id, req.note),
        )
    return JSONResponse({"ok": True})


@router.get("/api/notes")
def list_notes() -> JSONResponse:
    with _db() as conn:
        rows = conn.execute("SELECT corpus_id, note FROM paper_notes").fetchall()
    return JSONResponse({r["corpus_id"]: r["note"] for r in rows})


# ── Qdrant index stats ────────────────────────────────────────────────────────

@router.get("/api/qdrant/stats")
def get_qdrant_stats() -> JSONResponse:
    from mabool.agents.local_dense.qdrant_index import get_index_stats  # noqa: PLC0415
    return JSONResponse(get_index_stats())


# ── Stats route ───────────────────────────────────────────────────────────────

@router.get("/api/stats")
def get_stats() -> JSONResponse:
    with _db() as conn:
        result_rows = conn.execute(
            "SELECT result_json FROM searches WHERE result_json IS NOT NULL"
        ).fetchall()
        total_searches = conn.execute("SELECT COUNT(*) FROM searches").fetchone()[0]

    tokens_by_model: dict[str, dict[str, int]] = {}
    for row in result_rows:
        try:
            data = json.loads(row["result_json"])
            breakdown = data.get("token_breakdown_by_model", {})
            for model, usage in breakdown.items():
                if model not in tokens_by_model:
                    tokens_by_model[model] = {"total": 0, "prompt": 0, "completion": 0, "reasoning": 0}
                tokens_by_model[model]["total"] += usage.get("total", 0)
                tokens_by_model[model]["prompt"] += usage.get("prompt", 0)
                tokens_by_model[model]["completion"] += usage.get("completion", 0)
                tokens_by_model[model]["reasoning"] += usage.get("reasoning", 0)
        except Exception:
            pass

    grand_total = sum(v["total"] for v in tokens_by_model.values())
    return JSONResponse({
        "total_searches": total_searches,
        "tokens_by_model": tokens_by_model,
        "grand_total_tokens": grand_total,
    })
