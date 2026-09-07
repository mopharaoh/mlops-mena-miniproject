import logging
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger


correlation_id_context: ContextVar[str] = ContextVar(
    "correlation_id",
    default="system",
)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_context.get()
        return True


def configure_logging() -> None:
    """Configure application-wide JSON logging."""

    handler = logging.StreamHandler()

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s"
    )

    handler.setFormatter(formatter)
    handler.addFilter(CorrelationIdFilter())

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)