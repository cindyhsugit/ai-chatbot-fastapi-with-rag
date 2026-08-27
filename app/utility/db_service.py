import sqlite3
from pathlib import Path


def get_db_schema_context(db_path: str) -> str:
    """Connects to SQLite DB and builds a schema context string dynamically."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query all user tables and their column details
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    tables = cursor.fetchall()

    schema_parts = ["Tables:"]

    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info('{table_name}');")
        columns = cursor.fetchall()
        # col[1] is the column name in PRAGMA table_info
        col_names = [col[1] for col in columns]

        schema_parts.append(f"- {table_name}({', '.join(col_names)})")

    conn.close()
    return "\n".join(schema_parts)
