"""Load Simpsons CSV data into a local SQLite database for text-to-SQL RAG.

Source: "The Simpsons by the Data" dataset (Kaggle / data.world rehost).
Run once to (re)build simpsons.db from the raw CSVs.
"""

import sqlite3
from pathlib import Path
import pandas

# Adjust however many .parent calls are needed to reach your project root
# from app/utility/load_csv_to_db.py — this assumes app/utility/ is two
# levels below the root (root/app/utility/load_csv_to_db.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
DB_PATH = PROJECT_ROOT / "simpsons.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    name TEXT,
    normalized_name TEXT,
    gender TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    image_url TEXT,
    imdb_rating REAL,
    imdb_votes REAL,
    number_in_season INTEGER,
    number_in_series INTEGER,
    original_air_date TEXT,
    original_air_year INTEGER,
    production_code TEXT,
    season INTEGER,
    title TEXT,
    us_viewers_in_millions REAL,
    video_url TEXT,
    views REAL
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY,
    name TEXT,
    normalized_name TEXT
);

CREATE TABLE IF NOT EXISTS script_lines (
    id INTEGER PRIMARY KEY,
    episode_id INTEGER,
    number INTEGER,
    raw_text TEXT,
    timestamp_in_ms TEXT,
    speaking_line TEXT,
    character_id INTEGER,
    location_id INTEGER,
    raw_character_text TEXT,
    raw_location_text TEXT,
    spoken_words TEXT,
    normalized_text TEXT,
    word_count INTEGER,
    FOREIGN KEY (episode_id) REFERENCES episodes(id),
    FOREIGN KEY (character_id) REFERENCES characters(id),
    FOREIGN KEY (location_id) REFERENCES locations(id)
);

CREATE INDEX IF NOT EXISTS idx_script_episode ON script_lines(episode_id);
CREATE INDEX IF NOT EXISTS idx_script_character ON script_lines(character_id);
CREATE INDEX IF NOT EXISTS idx_script_location ON script_lines(location_id);
CREATE INDEX IF NOT EXISTS idx_episodes_season ON episodes(season);
"""


def load_csv_to_db() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)

    characters_df = pandas.read_csv(DATA_DIR / "simpsons_characters.csv")
    episodes_df = pandas.read_csv(DATA_DIR / "simpsons_episodes.csv")
    locations_df = pandas.read_csv(DATA_DIR / "simpsons_locations.csv")
    script_lines_df = pandas.read_csv(
        DATA_DIR / "simpsons_script_lines.csv", low_memory=False
    )

    script_lines_df["character_id"] = pandas.to_numeric(
        script_lines_df["character_id"], errors="coerce"
    )
    script_lines_df["word_count"] = pandas.to_numeric(
        script_lines_df["word_count"], errors="coerce"
    )

    characters_df.to_sql("characters", connection, if_exists="replace", index=False)
    episodes_df.to_sql("episodes", connection, if_exists="replace", index=False)
    locations_df.to_sql("locations", connection, if_exists="replace", index=False)
    script_lines_df.to_sql("script_lines", connection, if_exists="replace", index=False)

    connection.commit()

    for table in ("characters", "episodes", "locations", "script_lines"):
        count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} rows loaded")

    connection.close()


if __name__ == "__main__":
    load_csv_to_db()
