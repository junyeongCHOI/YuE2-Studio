"""SQLite-backed library: one row per take, holding its request and its outcome."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    status        TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    style         TEXT NOT NULL DEFAULT '',
    lyrics        TEXT NOT NULL DEFAULT '',
    cot           TEXT NOT NULL DEFAULT 'full',
    seed          INTEGER,
    cfg_scale     REAL,
    max_tokens    INTEGER,
    ode_steps     INTEGER NOT NULL DEFAULT 32,
    preset        TEXT,
    mode          TEXT NOT NULL DEFAULT 'compose',
    source_job    TEXT,
    source_kind   TEXT NOT NULL DEFAULT 'prompt',
    source_name   TEXT,
    abc           TEXT,
    notes         TEXT NOT NULL DEFAULT '',
    favorite      INTEGER NOT NULL DEFAULT 0,
    elapsed       REAL,
    audio_seconds REAL,
    truncated     TEXT,
    error         TEXT,
    timing        TEXT,
    history       TEXT,
    mastering     TEXT,
    request       TEXT
);
CREATE INDEX IF NOT EXISTS jobs_created   ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS jobs_status    ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_favorite  ON jobs(favorite);
"""

COLUMNS = ["id", "created_at", "updated_at", "status", "title", "style", "lyrics", "cot",
           "seed", "cfg_scale", "max_tokens", "ode_steps", "preset", "mode", "source_job",
           "source_kind", "source_name", "abc", "notes", "favorite", "elapsed",
           "audio_seconds", "truncated", "error", "timing", "history", "mastering", "request"]
JSON_COLUMNS = {"truncated", "timing", "history", "mastering", "request"}
EDITABLE = {"title", "style", "lyrics", "notes", "favorite"}
# Explicit values for the NOT NULL columns: a missing key must not become NULL.
NOT_NULL_DEFAULTS = {"status": "queued", "title": "", "style": "", "lyrics": "", "cot": "full",
                     "ode_steps": 32, "mode": "compose", "source_kind": "prompt",
                     "notes": "", "favorite": 0}


def _dump(value):
    return None if value is None else json.dumps(value, ensure_ascii=False, default=float)


def _like_escape(term):
    """Make % and _ literal inside a LIKE pattern (paired with ESCAPE '\\')."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _filters(search=None, status=None, favorite=None):
    """Shared WHERE clause so list() and count() can never disagree."""
    clauses, parameters = [], {}
    if search:
        like = " OR ".join(f"{column} LIKE :search ESCAPE '\\'"
                           for column in ("title", "style", "lyrics", "notes"))
        clauses.append(f"({like})")
        parameters["search"] = f"%{_like_escape(search)}%"
    if status:
        clauses.append("status = :status")
        parameters["status"] = status
    if favorite:
        clauses.append("favorite = 1")
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), parameters


def _load(value):
    if value is None:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


class Store:
    """Thread-safe SQLite wrapper. One connection guarded by a lock."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self.lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.executescript(SCHEMA)
            self.connection.commit()

    def close(self):
        with self.lock:
            self.connection.close()

    # -- writes ----------------------------------------------------------

    def upsert(self, record):
        row = {name: record.get(name) for name in COLUMNS}
        for name, fallback in NOT_NULL_DEFAULTS.items():
            if row[name] is None:
                row[name] = fallback
        row["updated_at"] = time.time()
        if row["created_at"] is None:
            row["created_at"] = row["updated_at"]
        for name in JSON_COLUMNS:
            if not isinstance(row[name], (str, type(None))):
                row[name] = _dump(row[name])
        row["favorite"] = int(bool(row.get("favorite")))
        placeholders = ", ".join(f":{name}" for name in COLUMNS)
        assignments = ", ".join(f"{name}=excluded.{name}" for name in COLUMNS if name != "id")
        with self.lock:
            self.connection.execute(
                f"INSERT INTO jobs ({', '.join(COLUMNS)}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}", row)
            self.connection.commit()

    def update(self, job_id, **fields):
        """Edit the user-owned columns. Unknown columns are rejected."""
        unknown = set(fields) - EDITABLE
        if unknown:
            raise ValueError(f"not editable: {sorted(unknown)}")
        if not fields:
            return self.get(job_id)
        if "favorite" in fields:
            fields["favorite"] = int(bool(fields["favorite"]))
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{name}=:{name}" for name in fields)
        with self.lock:
            cursor = self.connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=:id", {**fields, "id": job_id})
            self.connection.commit()
        return self.get(job_id) if cursor.rowcount else None

    def delete(self, job_id):
        with self.lock:
            cursor = self.connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            self.connection.commit()
        return cursor.rowcount > 0

    # -- reads -----------------------------------------------------------

    def get(self, job_id):
        with self.lock:
            row = self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._record(row)

    def list(self, search=None, status=None, favorite=None, limit=200, offset=0):
        where, parameters = _filters(search, status, favorite)
        parameters.update(limit=int(limit), offset=int(offset))
        with self.lock:
            rows = self.connection.execute(
                f"SELECT * FROM jobs {where} ORDER BY created_at DESC LIMIT :limit OFFSET :offset",
                parameters).fetchall()
        return [self._record(row) for row in rows]

    def count(self, search=None, status=None, favorite=None):
        """Row count under the same filters list() applies."""
        where, parameters = _filters(search, status, favorite)
        with self.lock:
            return self.connection.execute(
                f"SELECT COUNT(*) FROM jobs {where}", parameters).fetchone()[0]

    @staticmethod
    def _record(row):
        if row is None:
            return None
        record = dict(row)
        for name in JSON_COLUMNS:
            record[name] = _load(record[name])
        record["favorite"] = bool(record["favorite"])
        return record


def import_json_library(store, library):
    """One-time migration of the job.json files written before SQLite."""
    imported = 0
    for path in sorted(Path(library).glob("*/job.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if store.get(data.get("id")) is not None:
            continue
        request = data.get("request", {}) or {}
        result = data.get("result") or {}
        store.upsert({
            "id": data["id"], "created_at": data.get("created_at", path.stat().st_mtime),
            "status": "interrupted" if data.get("status") in {"queued", "running"} else data.get("status", "complete"),
            "title": request.get("title", ""), "style": request.get("style", ""),
            "lyrics": request.get("lyrics", ""), "cot": request.get("cot", "full"),
            "seed": request.get("seed"), "cfg_scale": request.get("cfg_scale"),
            "max_tokens": request.get("max_tokens"), "ode_steps": request.get("ode_steps", 32),
            "preset": request.get("preset"), "mode": request.get("mode", "compose"),
            "source_job": request.get("source_job"), "source_kind": request.get("source_kind", "prompt"),
            "abc": request.get("abc"), "elapsed": data.get("elapsed"),
            "audio_seconds": result.get("audio_seconds"), "truncated": result.get("truncated"),
            "error": data.get("error"), "timing": result.get("timing"),
            "history": data.get("history"), "mastering": data.get("mastering"),
            "request": request,
        })
        imported += 1
    return imported
