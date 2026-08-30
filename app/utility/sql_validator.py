from pathlib import Path
import sqlite3
from typing import Dict, Optional, Set, Tuple
import sqlglot
from sqlglot import exp


def load_db_schema(db_path: Path) -> Dict[str, Set[str]]:
    """
    Reads the ground-truth schema directly from SQLite metadata.
    Returns: Dict mapping lowercase table_name -> set(lowercase column_names).
    """
    schema = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}');")
            schema[table.lower()] = {row[1].lower() for row in cursor.fetchall()}

    return schema


def parse_and_validate_ast(
    raw_sql: str,
) -> Tuple[bool, Optional[exp.Expression], Optional[str]]:
    """
    Step 3.1: Ensures query is valid SQLite, a single SELECT statement,
    and free of mutation/forbidden SQL nodes.
    """
    cleaned = raw_sql.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    cleaned = cleaned.strip().rstrip(";")

    try:
        statements = sqlglot.parse(cleaned, read="sqlite")
    except sqlglot.errors.ParseError as e:
        return False, None, f"parse_error: {e}"

    if len(statements) != 1 or statements[0] is None:
        return False, None, "multiple_or_empty_statements_not_allowed"

    parsed = statements[0]
    if not isinstance(parsed, exp.Select):
        return False, None, "only_select_statements_allowed"

    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Attach,
        exp.Pragma,
        exp.Command,
    )
    if any(isinstance(node, forbidden) for node in parsed.walk()):
        return False, None, "forbidden_statement_type_found"

    return True, parsed, None


def verify_schema(
    parsed: exp.Expression, allowed_schema: Dict[str, Set[str]]
) -> Tuple[bool, Optional[str]]:
    """
    Step 3.2: Confirms every table and column referenced in the AST
    exists in allowed_schema (catches hallucinations and injection attempts).
    """
    # 1. Check referenced tables
    referenced_tables = {
        table.name.lower() for table in parsed.find_all(exp.Table) if table.name
    }
    for table_name in referenced_tables:
        if table_name not in allowed_schema:
            return False, f"hallucinated_table: '{table_name}'"

    # 2. Check referenced columns across allowed table schemas
    all_allowed_columns = set()
    for table_name in referenced_tables:
        all_allowed_columns.update(allowed_schema[table_name])

    for column_node in parsed.find_all(exp.Column):
        col_name = column_node.name.lower()
        if col_name == "*" or col_name in all_allowed_columns:
            continue
        return False, f"hallucinated_column: '{col_name}'"

    return True, None


def validate_sql(
    raw_sql: str, allowed_schema: Dict[str, Set[str]]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Main orchestrator method executing Step 3.1 and Step 3.2 sequentially.
    Returns: (is_valid: bool, safe_sql: str | None, error_message: str | None)
    """
    # Step 3.1: AST & Syntax Check
    is_valid, parsed_ast, err = parse_and_validate_ast(raw_sql)
    if not is_valid or parsed_ast is None:
        return False, None, err

    # Step 3.2: Schema Verification Check
    is_schema_valid, schema_err = verify_schema(parsed_ast, allowed_schema)
    if not is_schema_valid:
        return False, None, schema_err

    # Re-serialize clean SQL from AST & append default LIMIT
    safe_sql = parsed_ast.sql(dialect="sqlite")
    if "limit" not in safe_sql.lower():
        safe_sql += " LIMIT 50"

    return True, safe_sql, None
