import logging

from prodml.logging_conf import (
    CorrelationIdFilter,
    configure_logging,
    reset_correlation_id,
    set_correlation_id,
)


def test_correlation_id_context():
    """Correlation ID should be stored in the current context."""
    token = set_correlation_id("test-correlation-id")

    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )

        log_filter = CorrelationIdFilter()

        assert log_filter.filter(record) is True
        assert record.correlation_id == "test-correlation-id"

    finally:
        reset_correlation_id(token)


def test_correlation_id_resets_to_system():
    """Resetting the context should restore the system value."""
    token = set_correlation_id("temporary-id")

    reset_correlation_id(token)

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="test message",
        args=(),
        exc_info=None,
    )

    log_filter = CorrelationIdFilter()

    log_filter.filter(record)

    assert record.correlation_id == "system"


def test_configure_logging():
    """Logging configuration should install the JSON logging handler."""
    root_logger = logging.getLogger()

    configure_logging()

    assert root_logger.level == logging.INFO
    assert len(root_logger.handlers) >= 1

    handler = root_logger.handlers[0]

    assert len(handler.filters) >= 1
    assert isinstance(
        handler.filters[0],
        CorrelationIdFilter,
    )