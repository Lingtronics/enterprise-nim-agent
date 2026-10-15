"""SQLite state for local transactional tools and approval audit records."""

import sqlite3
from pathlib import Path


class ToolStorage:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self.connect()
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS deployments (
                    project_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL
                        CHECK(status IN ('pending','approved','deployed','failed')),
                    version INTEGER NOT NULL DEFAULT 0 CHECK(version >= 0),
                    security_scope TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES deployments(project_id),
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    request_id TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL
                );
            """)
            connection.executemany(
                "INSERT OR IGNORE INTO deployments(project_id,status,version,security_scope) "
                "VALUES(?,?,0,?)",
                [
                    ("atlas", "pending", "public"),
                    ("zephyr", "approved", "public"),
                    ("orion", "pending", "internal"),
                ],
            )
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        """Caller owns the connection, transaction, commit, and close."""
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection
