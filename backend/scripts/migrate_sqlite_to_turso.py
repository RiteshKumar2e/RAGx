"""Copy a local SQLite database into Turso.

Switching ``TURSO_DATABASE_URL`` on changes *where* RAGX reads and writes; it
does not move anything. Without this step the schema is created empty on Turso
and previously ingested documents look like they have vanished.

Usage (from ``backend/``)::

    python scripts/migrate_sqlite_to_turso.py            # dry run: report only
    python scripts/migrate_sqlite_to_turso.py --apply    # perform the copy

Only tables that exist on both sides are copied, and rows are written with
``INSERT OR REPLACE`` keyed on the primary key, so re-running is safe: a second
run overwrites the same rows rather than duplicating them.

Vectors live in Qdrant and the graph in Neo4j/NetworkX; neither is touched here.
This moves the relational metadata only.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

BATCH = 200


def _open_turso():
    import libsql  # noqa: PLC0415

    settings = get_settings()
    if not settings.turso_database_url:
        raise SystemExit("TURSO_DATABASE_URL is not set — nothing to migrate to.")
    host = settings.turso_database_url.split("://", 1)[-1].rstrip("/")
    return libsql.connect(f"https://{host}", auth_token=settings.turso_auth_token)


def _table_names(cursor) -> list[str]:
    rows = cursor.execute(
        "select name from sqlite_master where type='table' "
        "and name not like 'sqlite_%' order by name"
    ).fetchall()
    return [r[0] for r in rows]


def _ordered_for_foreign_keys(local: sqlite3.Connection, tables: list[str]) -> list[str]:
    """Return tables ordered so parents are inserted before their children.

    Inserting a child row before its parent violates the foreign keys the schema
    declares. The dependency graph is read from the database itself rather than
    hard-coded, so adding a model later does not silently break this script.
    """
    deps: dict[str, set[str]] = {}
    for table in tables:
        parents = {
            row[2]
            for row in local.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            if row[2] in tables and row[2] != table
        }
        deps[table] = parents

    ordered: list[str] = []
    remaining = dict(deps)
    while remaining:
        ready = sorted(t for t, parents in remaining.items() if not (parents - set(ordered)))
        if not ready:
            # A cycle (or a self-reference chain) -- fall back to the remaining
            # order rather than dropping tables from the migration.
            ordered.extend(sorted(remaining))
            break
        ordered.extend(ready)
        for table in ready:
            remaining.pop(table)
    return ordered


def _columns(cursor, table: str) -> list[str]:
    return [row[1] for row in cursor.execute(f'PRAGMA table_info("{table}")').fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="perform the copy (default is a dry run)"
    )
    parser.add_argument(
        "--source",
        default=str(Path(__file__).resolve().parent.parent / "data" / "ragx.db"),
        help="path to the local SQLite file",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"No local database at {source}")

    local = sqlite3.connect(source)
    local.row_factory = sqlite3.Row
    remote = _open_turso()

    local_tables = set(_table_names(local))
    remote_tables = set(_table_names(remote))
    shared = sorted(local_tables & remote_tables)
    skipped = sorted(local_tables - remote_tables)

    if skipped:
        print(f"skipping (not on Turso): {', '.join(skipped)}")

    plan = _ordered_for_foreign_keys(local, shared)
    mode = "COPY" if args.apply else "DRY RUN"
    print(f"\n{mode}: {source}  ->  Turso\n")
    print(f"{'table':28} {'rows':>7}  {'status'}")
    print("-" * 60)

    total = 0
    for table in plan:
        local_cols = _columns(local, table)
        remote_cols = set(_columns(remote, table))
        cols = [c for c in local_cols if c in remote_cols]
        if not cols:
            print(f"{table:28} {'-':>7}  no shared columns, skipped")
            continue

        rows = local.execute(
            f'select {", ".join(chr(34) + c + chr(34) for c in cols)} from "{table}"'
        ).fetchall()
        if not rows:
            print(f"{table:28} {0:>7}  empty")
            continue

        if not args.apply:
            print(f"{table:28} {len(rows):>7}  would copy")
            total += len(rows)
            continue

        placeholders = ", ".join("?" for _ in cols)
        column_list = ", ".join(f'"{c}"' for c in cols)
        sql = f'INSERT OR REPLACE INTO "{table}" ({column_list}) VALUES ({placeholders})'
        payload = [tuple(row[c] for c in cols) for row in rows]
        for start in range(0, len(payload), BATCH):
            remote.executemany(sql, payload[start : start + BATCH])
        remote.commit()
        print(f"{table:28} {len(rows):>7}  copied")
        total += len(rows)

    print("-" * 60)
    print(f"{'total':28} {total:>7}  {'rows' if args.apply else 'rows pending'}")

    if not args.apply:
        print("\nNothing was written. Re-run with --apply to perform the copy.")

    local.close()
    remote.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
