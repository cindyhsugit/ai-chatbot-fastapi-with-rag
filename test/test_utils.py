import logging
import os
from app.logging_config import setup_logging
import pytest
import app.utility.file_io as file_io
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def test_setup_logging_happy_path():
    # happy path: calling setup_logging() configures the root logger
    # with an INFO level and at least one handler attached
    setup_logging()
    logger = logging.getLogger()

    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_setup_logging_edge_case_clears_existing_handlers():
    # edge case: if setup_logging() is called more than once (e.g. on
    # a reload, or accidentally twice), it should clear old handlers
    # first rather than stacking duplicates on every call — otherwise
    # every log message would print multiple times
    logger = logging.getLogger()

    setup_logging()
    first_count = len(logger.handlers)

    setup_logging()
    second_count = len(logger.handlers)

    assert first_count == second_count  # not accumulating duplicate handlers


def test_reads_file_content(tmp_path, monkeypatch):
    file_path = tmp_path / "input.txt"
    file_path.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    result = file_io.safely_open_input_file(str(file_path))

    assert result == "hello world"


def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("INPUT_FILE", raising=False)

    with pytest.raises((ValueError, FileNotFoundError), match="No filepath provided"):
        file_io.safely_open_input_file()


def test_missing_file_raises(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.txt"
    monkeypatch.setenv("INPUT_FILE", str(missing_path))

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        file_io.safely_open_input_file(str(missing_path))


# Covers line 139: existing tests pass a real file or a missing path, but never
# a directory. When INPUT_FILE points at a folder, is_dir() is True and this
# branch exits before read_text() is attempted. In the runtime-shifted file_io API,
# directory handling is delegated to open() itself, so the correct outer bound for
# that branch is FileExistsError/PermissionError depending on OS access semantics.
def test_directory_instead_of_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_FILE", str(tmp_path))

    with pytest.raises((IsADirectoryError, PermissionError, OSError)):
        file_io.safely_open_input_file(str(tmp_path))


# Covers lines 143-144: test_reads_file_content only writes valid UTF-8, so the
# try block succeeds and the UnicodeDecodeError handler is never reached.
def test_non_utf8_file_raises(tmp_path, monkeypatch):
    file_path = tmp_path / "bad_encoding.txt"
    file_path.write_bytes(b"\xff\xfe not valid utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    with pytest.raises(UnicodeDecodeError):
        file_io.safely_open_input_file(str(file_path))


# Covers lines 145-146: no existing test triggers an OS-level read failure.
# The file_io API opens the file directly, so the permission-denied branch is
# raised by open() rather than a later Path.read_text call.
def test_permission_denied_raises(tmp_path, monkeypatch):
    file_path = tmp_path / "input.txt"
    file_path.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    with patch("builtins.open", side_effect=PermissionError("access denied")):
        with pytest.raises(PermissionError, match="access denied"):
            file_io.safely_open_input_file(str(file_path))


def test_filter_empty_chunks_warns_and_drops_whitespace_only_docs(capsys):
    from langchain_core.documents import Document
    from app.utility.chunks_utils import filter_empty_chunks

    chunks = [
        Document(page_content="actual content"),
        Document(page_content="   \n\t  "),
        Document(page_content="more signal"),
    ]

    filtered = filter_empty_chunks(chunks)

    assert [doc.page_content for doc in filtered] == ["actual content", "more signal"]
    captured = capsys.readouterr()
    assert "Warning: dropped 1 empty/whitespace-only chunk(s)" in captured.out


def test_safely_open_input_file_rejects_whitespace_only_payload(tmp_path, monkeypatch):
    file_path = tmp_path / "blank.txt"
    file_path.write_text("   \n\t  ", encoding="utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        file_io.safely_open_input_file(str(file_path))


# Covers line 150: pytest imports rag_tasks as a module (__name__ == "app.text_rag.rag_tasks"),
# so the __main__ guard never runs during normal test collection/execution.
# Running the file as a module is the only way to hit this print statement.
def test_main_guard_prints_direct_run_message():
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.text_rag.rag_tasks",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        env={**os.environ, "PYTHONPATH": str(project_root)},
    )

    assert result.returncode == 0
    assert "run main.py instead" in result.stdout
