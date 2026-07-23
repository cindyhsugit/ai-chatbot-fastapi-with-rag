import logging
from logging_config import setup_logging


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

import pytest
import rag_tasks


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