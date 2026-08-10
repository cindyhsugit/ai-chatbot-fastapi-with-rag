import os
from dotenv import load_dotenv

load_dotenv()


def safely_open_input_file(filepath: str | None = None) -> str:
    """
    Reads the input text file, raising a clear error if it's missing,
    unreadable, or empty. Defaults to the INPUT_FILE path from .env
    if no filepath is passed explicitly.
    """
    if filepath is None:
        raise ValueError("No filepath provided")

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
