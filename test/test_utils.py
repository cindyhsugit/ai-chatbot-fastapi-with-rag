import logging
from logging_config import setup_logging
import pytest
import rag_tasks
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

    result = rag_tasks.safely_open_input_file()

    assert result == "hello world"


def test_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("INPUT_FILE", raising=False)

    with pytest.raises(SystemExit, match="INPUT_FILE environment variable is not set"):
        rag_tasks.safely_open_input_file()


def test_missing_file_raises(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.txt"
    monkeypatch.setenv("INPUT_FILE", str(missing_path))

    with pytest.raises(SystemExit, match="Input file not found"):
        rag_tasks.safely_open_input_file()

# Covers line 139: existing tests pass a real file or a missing path, but never
# a directory. When INPUT_FILE points at a folder, is_dir() is True and this
# branch exits before read_text() is attempted.
def test_directory_instead_of_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("INPUT_FILE", str(tmp_path))

    with pytest.raises(SystemExit, match="Expected a file, got a directory"):
        rag_tasks.safely_open_input_file()


# Covers lines 143-144: test_reads_file_content only writes valid UTF-8, so the
# try block succeeds and the UnicodeDecodeError handler is never reached.
def test_non_utf8_file_raises(tmp_path, monkeypatch):
    file_path = tmp_path / "bad_encoding.txt"
    file_path.write_bytes(b"\xff\xfe not valid utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    with pytest.raises(SystemExit, match="Could not decode .* as UTF-8"):
        rag_tasks.safely_open_input_file()


# Covers lines 145-146: no existing test triggers an OS-level read failure.
# Patching read_text raises PermissionError after exists/is_dir checks pass.
def test_permission_denied_raises(tmp_path, monkeypatch):
    file_path = tmp_path / "input.txt"
    file_path.write_text("hello world", encoding="utf-8")
    monkeypatch.setenv("INPUT_FILE", str(file_path))

    with patch.object(Path, "read_text", side_effect=PermissionError()):
        with pytest.raises(SystemExit, match="Permission denied reading file"):
            rag_tasks.safely_open_input_file()


# Covers line 150: pytest imports rag_tasks as a module (__name__ == "rag_tasks"),
# so the __main__ guard never runs during normal test collection/execution.
# Running the file as a script is the only way to hit this print statement.
def test_main_guard_prints_direct_run_message():
    project_root = Path(__file__).resolve().parent.parent

    result = subprocess.run(
        [sys.executable, "rag_tasks.py"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert "run main.py instead" in result.stdout