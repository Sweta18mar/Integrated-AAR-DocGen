"""Job state and vector index.

JobStore maps to the DynamoDB job table; VectorStore maps to S3 Vectors with
one namespace per tenant and a filter by job.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, tenant TEXT, workspace TEXT, title TEXT, status TEXT,
  instructions TEXT, outline TEXT, vendors TEXT DEFAULT '[]', error TEXT DEFAULT '',
  created REAL, updated REAL);
CREATE TABLE IF NOT EXISTS files (
  job_id TEXT, name TEXT, kind TEXT, path TEXT, chars INTEGER DEFAULT 0, redactions TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS events (job_id TEXT, ts REAL, stage TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS drafts (
  job_id TEXT, version INTEGER, content TEXT, feedback TEXT, created REAL,
  PRIMARY KEY (job_id, version));
CREATE TABLE IF NOT EXISTS exports (job_id TEXT, version INTEGER, format TEXT, path TEXT);
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY, tenant TEXT, job_id TEXT, source TEXT, text TEXT, vector BLOB);
CREATE INDEX IF NOT EXISTS chunks_tenant ON chunks (tenant, job_id);
"""


class Store:
    """SQLite-backed store, safe to share across Streamlit threads."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.data_dir / "platform.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            rows = self.db.execute(sql, args).fetchall()
            self.db.commit()
            return rows

    def job_dir(self, job_id: str) -> Path:
        path = self.data_dir / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------ jobs
    def create_job(self, tenant: str, workspace: str, title: str, instructions: str, outline: list[dict]) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        now = time.time()
        self._exec(
            "INSERT INTO jobs (id, tenant, workspace, title, status, instructions, outline, created, updated)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, tenant, workspace, title, "CREATED", instructions, json.dumps(outline), now, now),
        )
        return job_id

    def job(self, job_id: str) -> dict:
        rows = self._exec("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not rows:
            raise KeyError(f"No job {job_id}")
        job = dict(rows[0])
        job["outline"] = json.loads(job["outline"] or "[]")
        job["vendors"] = json.loads(job["vendors"] or "[]")
        return job

    def jobs(self, tenant: str) -> list[dict]:
        return [dict(r) for r in self._exec(
            "SELECT id, title, workspace, status, updated FROM jobs WHERE tenant=? ORDER BY created DESC", (tenant,))]

    def update_job(self, job_id: str, **fields) -> None:
        for key in ("outline", "vendors"):
            if key in fields:
                fields[key] = json.dumps(fields[key])
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    # ------------------------------------------------------------ files
    def add_file(self, job_id: str, name: str, kind: str, path: Path) -> None:
        self._exec("INSERT INTO files (job_id, name, kind, path) VALUES (?,?,?,?)", (job_id, name, kind, str(path)))

    def files(self, job_id: str) -> list[dict]:
        rows = self._exec("SELECT * FROM files WHERE job_id=?", (job_id,))
        return [{**dict(r), "redactions": json.loads(r["redactions"] or "{}")} for r in rows]

    def update_file(self, job_id: str, name: str, **fields) -> None:
        if "redactions" in fields:
            fields["redactions"] = json.dumps(fields["redactions"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self._exec(f"UPDATE files SET {cols} WHERE job_id=? AND name=?", (*fields.values(), job_id, name))

    # ------------------------------------------------------------ events
    def log(self, job_id: str, stage: str, message: str) -> None:
        self._exec("INSERT INTO events VALUES (?,?,?,?)", (job_id, time.time(), stage, message))

    def events(self, job_id: str) -> list[dict]:
        return [dict(r) for r in self._exec("SELECT * FROM events WHERE job_id=? ORDER BY ts", (job_id,))]

    # ------------------------------------------------------------ drafts and exports
    def save_draft(self, job_id: str, content: dict, feedback: str = "") -> int:
        with self._lock:
            version = (self._exec("SELECT MAX(version) v FROM drafts WHERE job_id=?", (job_id,))[0]["v"] or 0) + 1
            self._exec("INSERT INTO drafts VALUES (?,?,?,?,?)",
                       (job_id, version, json.dumps(content), feedback, time.time()))
        return version

    def drafts(self, job_id: str) -> list[dict]:
        rows = self._exec("SELECT * FROM drafts WHERE job_id=? ORDER BY version", (job_id,))
        return [{**dict(r), "content": json.loads(r["content"])} for r in rows]

    def add_export(self, job_id: str, version: int, fmt: str, path: Path) -> None:
        self._exec("INSERT INTO exports VALUES (?,?,?,?)", (job_id, version, fmt, str(path)))

    def exports(self, job_id: str) -> list[dict]:
        return [dict(r) for r in self._exec("SELECT * FROM exports WHERE job_id=? ORDER BY version, format", (job_id,))]


class VectorStore:
    """Chunk vectors in the same SQLite file, namespaced by tenant and filtered by job."""

    def __init__(self, store: Store):
        self.store = store

    def add(self, tenant: str, job_id: str, items: list[dict], vectors: list[list[float]]) -> None:
        with self.store._lock:
            self.store.db.executemany(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                [(it["id"], tenant, job_id, it["source"], it["text"], np.asarray(v, dtype=np.float32).tobytes())
                 for it, v in zip(items, vectors)],
            )
            self.store.db.commit()

    def count(self, tenant: str, job_id: str) -> int:
        return self.store._exec("SELECT COUNT(*) n FROM chunks WHERE tenant=? AND job_id=?", (tenant, job_id))[0]["n"]

    def chunks(self, tenant: str, job_id: str) -> list[dict]:
        return [dict(r) for r in self.store._exec(
            "SELECT id, source, text FROM chunks WHERE tenant=? AND job_id=? ORDER BY rowid", (tenant, job_id))]

    def search(self, tenant: str, job_id: str, vector: list[float], k: int = 5) -> list[dict]:
        """Top-k by cosine similarity. Always filtered to one tenant and one job."""
        rows = self.store._exec(
            "SELECT id, source, text, vector FROM chunks WHERE tenant=? AND job_id=?", (tenant, job_id))
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        query = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query) or 1.0)
        scores = matrix @ query / np.where(norms == 0, 1.0, norms)
        order = np.argsort(-scores)[:k]
        return [{"id": rows[i]["id"], "source": rows[i]["source"], "text": rows[i]["text"],
                 "score": float(scores[i])} for i in order]
