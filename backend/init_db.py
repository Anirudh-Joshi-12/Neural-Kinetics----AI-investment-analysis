"""Database initialization with migration support.

Usage:
    python init_db.py              # Run pending migrations (safe, non-destructive)
    python init_db.py --reset      # DROP + recreate everything (dev only)
    python init_db.py --status     # Show migration status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from app.core.config import settings
from app.core.database import Base, engine


MIGRATIONS_DIR = ROOT / "migrations"


def _load_migration_files() -> list[Path]:
    """Return sorted list of migration .sql files."""
    if not MIGRATIONS_DIR.exists():
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _get_applied_migrations() -> set[str]:
    """Return set of already-applied migration filenames."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS _schema_migrations (
                filename VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
            )
        """))
        conn.commit()
        result = conn.execute(text("SELECT filename FROM _schema_migrations"))
        return {row[0] for row in result.fetchall()}


def _apply_migration(filepath: Path, verbose: bool = True) -> None:
    """Apply a single migration SQL file."""
    sql = filepath.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in sql.split(';'):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    if verbose:
        print(f"[init_db] applied {filepath.name}")


def run_migrations(verbose: bool = True) -> None:
    """Run all pending migrations in order."""
    # 1. Ensure pgvector extension exists.
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    if verbose:
        print(f"[init_db] vector extension enabled on {settings.database_url.split('@')[-1]}")

    # 2. Import models so SQLAlchemy knows about them for any CREATE TABLE
    #    not covered by migrations (future-proof).
    from app.models.report import Report  # noqa
    from app.models.chunk import Chunk    # noqa

    # 3. Apply pending migrations.
    applied = _get_applied_migrations()
    all_files = _load_migration_files()

    pending = [f for f in all_files if f.name not in applied]
    if not pending:
        if verbose:
            print("[init_db] no pending migrations")
        return

    if verbose:
        print(f"[init_db] applying {len(pending)} pending migration(s)...")
    for filepath in pending:
        _apply_migration(filepath, verbose)

    # 4. Run SQLAlchemy create_all for any tables not covered by migrations.
    Base.metadata.create_all(bind=engine)
    if verbose:
        tables = sorted(Base.metadata.tables.keys())
        print(f"[init_db] OK. Tables: {tables}")


def reset_database(verbose: bool = True) -> None:
    """DANGER: Drop all tables and re-apply all migrations from scratch.

    Use ONLY in development. Production should never call this.
    """
    if verbose:
        print("[init_db] WARNING: RESET MODE: dropping all tables...")

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS reports CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS _schema_migrations CASCADE"))
        conn.commit()

    if verbose:
        print("[init_db] dropped all tables")

    # Re-run migrations from clean state.
    run_migrations(verbose)


def show_status() -> None:
    """Show migration status."""
    applied = _get_applied_migrations()
    all_files = _load_migration_files()

    print("Migration status:")
    print("-" * 60)
    if not all_files:
        print("  (no migration files found)")
        return

    for f in all_files:
        status = "[OK]" if f.name in applied else "[--]"
        print(f"  {f.name:<40} {status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Database initialization and migrations")
    parser.add_argument("--reset", action="store_true",
                        help="Drop all tables and re-apply all migrations (DEV ONLY)")
    parser.add_argument("--status", action="store_true",
                        help="Show migration status and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.reset:
        # Confirm in interactive mode.
        if sys.stdin.isatty():
            confirm = input("WARNING: This will DESTROY all data. Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                print("Aborted.")
                return 1
        reset_database()
    else:
        run_migrations()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())