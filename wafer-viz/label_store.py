"""
SQLite-backed store for the in-house defect taxonomy and per-wafer labels.

Two label fields are kept structurally separate on every wafer row:

- predicted_* : written only by the model (initial inference, retrain's
  batch pass). Every write to these columns is guarded by
  `WHERE verified = 0`, so an automated process can never move or
  overwrite a label a human has confirmed.
- verified_*  : written only by `verify_label`, which is called from a
  human action in the UI (the "Verify" button in Classify, or "Confirm"
  in Defect Reevaluation). This is the only path that can set `verified=1`.
"""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS defect_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES defect_types(id),
    created_at TEXT NOT NULL
);

-- Plain UNIQUE(name, parent_id) doesn't work here: SQL treats every NULL as
-- distinct from every other NULL, so it silently allows unlimited duplicate
-- top-level types (parent_id IS NULL). Partial indexes split top-level and
-- child uniqueness so NULL is never part of an indexed equality check.
CREATE UNIQUE INDEX IF NOT EXISTS idx_defect_types_top_level
    ON defect_types(name) WHERE parent_id IS NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_defect_types_child
    ON defect_types(name, parent_id) WHERE parent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS wafer_labels (
    wafer_id INTEGER PRIMARY KEY,
    predicted_type_id INTEGER REFERENCES defect_types(id),
    predicted_confidence REAL,
    model_version TEXT,
    verified_type_id INTEGER REFERENCES defect_types(id),
    verified INTEGER NOT NULL DEFAULT 0,
    verified_by TEXT,
    verified_at TEXT,
    updated_at TEXT NOT NULL
);
"""


def connect(db_path: str = "data/labels.db") -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def seed_taxonomy(conn: sqlite3.Connection, umbrella_classes: list[str]) -> None:
    now = datetime.datetime.utcnow().isoformat()
    for name in umbrella_classes:
        conn.execute(
            "INSERT OR IGNORE INTO defect_types (name, parent_id, created_at) "
            "VALUES (?, NULL, ?)",
            (name, now),
        )
    conn.commit()


def get_or_create_type(conn: sqlite3.Connection, name: str, parent_id: int | None) -> int:
    cur = conn.execute(
        "SELECT id FROM defect_types WHERE name = ? AND parent_id IS ?",
        (name, parent_id),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    now = datetime.datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT INTO defect_types (name, parent_id, created_at) VALUES (?, ?, ?)",
        (name, parent_id, now),
    )
    conn.commit()
    return cur.lastrowid


def list_taxonomy(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT id, name, parent_id FROM defect_types ORDER BY parent_id IS NOT NULL, name")
    return [{"id": r[0], "name": r[1], "parent_id": r[2]} for r in cur.fetchall()]


def type_name(conn: sqlite3.Connection, type_id: int | None) -> str | None:
    if type_id is None:
        return None
    cur = conn.execute("SELECT name FROM defect_types WHERE id = ?", (type_id,))
    row = cur.fetchone()
    return row[0] if row else None


def upsert_prediction(
    conn: sqlite3.Connection,
    wafer_id: int,
    type_id: int,
    confidence: float,
    model_version: str,
) -> None:
    """Write a model prediction. No-ops on wafers a human has already verified."""
    now = datetime.datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO wafer_labels (wafer_id, predicted_type_id, predicted_confidence,
                                   model_version, verified, updated_at)
        VALUES (?, ?, ?, ?, 0, ?)
        ON CONFLICT(wafer_id) DO UPDATE SET
            predicted_type_id = excluded.predicted_type_id,
            predicted_confidence = excluded.predicted_confidence,
            model_version = excluded.model_version,
            updated_at = excluded.updated_at
        WHERE wafer_labels.verified = 0
        """,
        (wafer_id, type_id, confidence, model_version, now),
    )
    conn.commit()


def upsert_predictions_bulk(conn: sqlite3.Connection, records) -> None:
    """Bulk version of upsert_prediction: records is an iterable of
    (wafer_id, type_id, confidence, model_version) tuples. Single commit,
    still guarded by WHERE verified = 0 per row."""
    now = datetime.datetime.utcnow().isoformat()
    conn.executemany(
        """
        INSERT INTO wafer_labels (wafer_id, predicted_type_id, predicted_confidence,
                                   model_version, verified, updated_at)
        VALUES (?, ?, ?, ?, 0, ?)
        ON CONFLICT(wafer_id) DO UPDATE SET
            predicted_type_id = excluded.predicted_type_id,
            predicted_confidence = excluded.predicted_confidence,
            model_version = excluded.model_version,
            updated_at = excluded.updated_at
        WHERE wafer_labels.verified = 0
        """,
        [(wid, tid, conf, model_version, now) for wid, tid, conf, model_version in records],
    )
    conn.commit()


def verify_label(
    conn: sqlite3.Connection,
    wafer_id: int,
    type_id: int,
    verified_by: str = "",
) -> None:
    """Human-confirmed ground truth. The only function allowed to set verified=1."""
    now = datetime.datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO wafer_labels (wafer_id, verified_type_id, verified, verified_by,
                                   verified_at, updated_at)
        VALUES (?, ?, 1, ?, ?, ?)
        ON CONFLICT(wafer_id) DO UPDATE SET
            verified_type_id = excluded.verified_type_id,
            verified = 1,
            verified_by = excluded.verified_by,
            verified_at = excluded.verified_at,
            updated_at = excluded.updated_at
        """,
        (wafer_id, type_id, verified_by, now, now),
    )
    conn.commit()


def get_label(conn: sqlite3.Connection, wafer_id: int) -> dict | None:
    cur = conn.execute(
        "SELECT wafer_id, predicted_type_id, predicted_confidence, model_version, "
        "verified_type_id, verified, verified_by, verified_at FROM wafer_labels "
        "WHERE wafer_id = ?",
        (wafer_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    keys = [
        "wafer_id", "predicted_type_id", "predicted_confidence", "model_version",
        "verified_type_id", "verified", "verified_by", "verified_at",
    ]
    return dict(zip(keys, row))


def verified_examples_for_type(conn: sqlite3.Connection, type_id: int) -> list[int]:
    cur = conn.execute(
        "SELECT wafer_id FROM wafer_labels WHERE verified = 1 AND verified_type_id = ?",
        (type_id,),
    )
    return [r[0] for r in cur.fetchall()]


def dedupe_taxonomy(conn: sqlite3.Connection) -> int:
    """One-off migration: collapse duplicate (name, parent_id) rows created
    before the partial-unique-index fix. Keeps the lowest id per group as
    canonical, remaps any wafer_labels/defect_types references to it, then
    deletes the rest. Returns the number of duplicate rows removed."""
    rows = conn.execute("SELECT id, name, parent_id FROM defect_types ORDER BY id").fetchall()
    groups: dict[tuple, list[int]] = {}
    for id_, name, parent_id in rows:
        groups.setdefault((name, parent_id), []).append(id_)

    remap = {}
    for ids in groups.values():
        canonical = min(ids)
        for dup in ids:
            if dup != canonical:
                remap[dup] = canonical

    if not remap:
        return 0

    for dup, canonical in remap.items():
        conn.execute("UPDATE wafer_labels SET predicted_type_id = ? WHERE predicted_type_id = ?", (canonical, dup))
        conn.execute("UPDATE wafer_labels SET verified_type_id = ? WHERE verified_type_id = ?", (canonical, dup))
        conn.execute("UPDATE defect_types SET parent_id = ? WHERE parent_id = ?", (canonical, dup))
    conn.execute(f"DELETE FROM defect_types WHERE id IN ({','.join('?' * len(remap))})", list(remap.keys()))
    conn.commit()
    return len(remap)


def verified_count(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM wafer_labels WHERE verified = 1")
    return cur.fetchone()[0]
