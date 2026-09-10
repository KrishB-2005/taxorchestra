"""Persistence for resolved field catalogs.

Resolving all 229 fields on the 1040 is the expensive part of the system: it is
the only stage that makes a model call per field. The result is a pure function
of the blank PDF, so it is cached against the form's SHA-256 — a new tax year
ships a new PDF, gets a new digest, and re-resolves automatically. There is no
stale-cache failure mode where last year's field numbering silently fills this
year's form.

Three backends behind one interface: `sqlite` (default, single file),
`memory` (tests), `dynamodb` (multi-instance deployments).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Protocol

CACHE_SCHEMA_VERSION = 1


class MappingCache(Protocol):
    name: str

    def get(self, key: str) -> dict[str, Any] | None: ...

    def put(self, key: str, value: dict[str, Any]) -> None: ...


class MemoryCache:
    """Process-local. Used by tests and by `--no-cache` runs."""

    name = "memory"

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._data.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = value


class SQLiteCache:
    """Default backend: one file, no service to run."""

    name = "sqlite"

    def __init__(self, path: str | Path | None = None) -> None:
        raw = path or os.getenv(
            "TAXORCHESTRA_CACHE_PATH", ".taxorchestra/mapping.cache.sqlite"
        )
        self.path = Path(raw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS field_catalog (
                    cache_key      TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    payload        TEXT NOT NULL,
                    created_at     REAL NOT NULL
                )
                """
            )

    def get(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM field_catalog WHERE cache_key = ? AND schema_version = ?",
                (key, CACHE_SCHEMA_VERSION),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO field_catalog (cache_key, schema_version, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    schema_version = excluded.schema_version,
                    created_at = excluded.created_at
                """,
                (key, CACHE_SCHEMA_VERSION, json.dumps(value), time.time()),
            )


class DynamoDBCache:
    """Shared backend for multiple workers.

    Same contract as SQLiteCache; the point of the abstraction is that nothing
    upstream changes when a deployment outgrows a local file.
    """

    name = "dynamodb"

    def __init__(self, table_name: str | None = None) -> None:
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - install-time
            raise RuntimeError(
                "dynamodb cache needs: pip install 'taxorchestra[aws]'"
            ) from exc
        self.table_name = table_name or os.getenv(
            "TAXORCHESTRA_DYNAMO_TABLE", "taxorchestra-field-mappings"
        )
        self._table = boto3.resource(
            "dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1")
        ).Table(self.table_name)

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._table.get_item(
            Key={"cache_key": f"v{CACHE_SCHEMA_VERSION}#{key}"}
        ).get("Item")
        return json.loads(item["payload"]) if item else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._table.put_item(
            Item={
                "cache_key": f"v{CACHE_SCHEMA_VERSION}#{key}",
                "payload": json.dumps(value),
                "created_at": int(time.time()),
            }
        )


_BACKENDS: dict[str, type] = {
    "memory": MemoryCache,
    "sqlite": SQLiteCache,
    "dynamodb": DynamoDBCache,
}


def build_cache(backend: str | None = None) -> MappingCache:
    name = (backend or os.getenv("TAXORCHESTRA_CACHE", "sqlite")).lower()
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise ValueError(
            f"unknown cache backend {name!r}; expected one of {', '.join(sorted(_BACKENDS))}"
        ) from None
