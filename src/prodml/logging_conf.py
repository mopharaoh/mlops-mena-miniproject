import contextvars
import logging
import sys

from pythonjsonlogger import jsonlogger


correlation_id_context: contextvars.ContextVar[str] = (
    contextvars.ContextVar(
        "correlation_id",
        default="system",
    )
)


def set_correlation_id(
    correlation_id: str,
) -> contextvars.Token[str]:
    """Set the correlation ID for the current context."""

    return correlation_id_context.set(
        correlation_id
    )


def reset_correlation_id(
    token: contextvars.Token[str],
) -> None:
    """Reset the correlation ID context."""

    correlation_id_context.reset(token)


class CorrelationIdFilter(logging.Filter):
    """Inject correlation_id into every log record."""

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        record.correlation_id = (
            correlation_id_context.get()
        )

        return True


def configure_logging() -> None:
    """Configure structured JSON logging."""

    handler = logging.StreamHandler(
        sys.stdout
    )

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s "
        "%(message)s %(correlation_id)s"
    )

    handler.setFormatter(formatter)
    handler.addFilter(
        CorrelationIdFilter()
    )

    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    root_logger.addHandler(handler)

    root_logger.setLevel(
        logging.INFO
    )