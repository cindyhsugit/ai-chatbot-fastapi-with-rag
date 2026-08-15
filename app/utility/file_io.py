import os
from dotenv import load_dotenv

load_dotenv()


# app/utility/file_io.py — unchanged, still reads one file at a time
def safely_open_input_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {filepath}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"File {filepath} is not valid UTF-8 — check encoding",
        )
    if not text.strip():
        raise ValueError(f"Input file is empty or whitespace-only: {filepath}")
    return text
