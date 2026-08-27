import re

# Deterministic patterns matching SQL database query intents
SQL_PATTERNS = re.compile(
    r"\b("
    r"how many|count|total|average|avg|highest|lowest|max|min|"
    r"list all|show all|which episode|which season|season \d+|episode \d+|"
    r"directed by|written by|aired in|top \d+|bottom \d+"
    r")\b",
    re.IGNORECASE,
)


def is_sql_intent(question: str) -> bool:
    """
    Fast regex pre-check (< 0.001s) to determine if a question
    requires exact relational SQL query execution.
    """
    if not question or not question.strip():
        return False

    return bool(SQL_PATTERNS.search(question.strip()))
