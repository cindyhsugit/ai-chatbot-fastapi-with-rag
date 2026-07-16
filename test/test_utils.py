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