import ollama
from pathlib import Path
from app.utility import db_service

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "simpsons.db"
SQL_SCHEMA_CONTEXT = db_service.get_db_schema_context(DB_PATH)

SQL_GENERATION_RULES = f"""You are a SQLite query generator. Given a natural language question
about The Simpsons, write ONE valid SQLite SELECT query that answers it.

{SQL_SCHEMA_CONTEXT}

Examples:

Q: How many episodes aired in season 5?
A: SELECT COUNT(*) FROM episodes WHERE season = 5

Q: What is the title of the 5th episode of season 5?
A: SELECT title FROM episodes WHERE season = 5 AND number_in_season = 5

Q: How many lines did Homer speak in season 1?
A: SELECT COUNT(*) FROM script_lines sl
   JOIN characters c ON sl.character_id = c.id
   JOIN episodes e ON sl.episode_id = e.id
   WHERE c.normalized_name = 'homer simpson' AND e.season = 1 AND sl.speaking_line = 1

Q: What locations does Milhouse appear in most?
A: SELECT l.normalized_name, COUNT(*) as line_count FROM script_lines sl
   JOIN characters c ON sl.character_id = c.id
   JOIN locations l ON sl.location_id = l.id
   WHERE c.normalized_name = 'milhouse van houten'
   GROUP BY l.normalized_name ORDER BY line_count DESC LIMIT 10

Rules:
- Output ONLY the SQL query. No explanation, no markdown fences.
- SELECT statements only. Never write INSERT, UPDATE, DELETE, DROP, ATTACH, PRAGMA, or CREATE.
- When filtering (WHERE clause) on character or location names, match against
  normalized_name using lowercase values. When displaying results (SELECT clause),
  use name for proper casing instead of normalized_name.
- Only filter on columns the question explicitly asks about. Do not add extra
  WHERE conditions (like gender, season, or other attributes) unless the question
  specifically mentions them.
- Use JOINs, GROUP BY, ORDER BY, WHERE, aggregate functions as needed.
- Include a LIMIT (default 50) unless the question is asking for a count or aggregate.
"""


async def generate_sql_with_llm(question: str) -> str:
    response = ollama.chat(
        model="gemma4:e4b",
        messages=[
            {"role": "system", "content": SQL_GENERATION_RULES},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]
