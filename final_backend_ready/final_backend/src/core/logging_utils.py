# src/core/logging_utils.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterable, Optional

from src.core.settings import get_settings

_settings = get_settings()

# -----------------------------------------------------------------------------
# Correlation / request context
# -----------------------------------------------------------------------------
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

def new_correlation_id() -> str:
    cid = uuid.uuid4().hex[:16]
    correlation_id.set(cid)
    return cid

def get_correlation_id() -> str:
    return correlation_id.get()


# -----------------------------------------------------------------------------
# JSON / pretty formatters
# -----------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "cid": get_correlation_id(),
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        # include any extra structured fields
        for k, v in getattr(record, "__dict__", {}).items():
            if k.startswith("_") or k in base or k in ("msg", "args", "levelname", "levelno",
                                                       "pathname", "filename", "module", "exc_info",
                                                       "exc_text", "stack_info", "lineno", "funcName",
                                                       "created", "msecs", "relativeCreated", "thread",
                                                       "threadName", "processName", "process"):
                continue
            try:
                json.dumps(v)  # ensure serializable
                base[k] = v
            except Exception:
                base[k] = repr(v)
        return json.dumps(base, ensure_ascii=False)

class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        cid = get_correlation_id()
        prefix = f"[{record.levelname:<5}] {record.name} cid={cid} - "
        text = super().format(record)
        return prefix + text


# -----------------------------------------------------------------------------
# Logger setup
# -----------------------------------------------------------------------------
def _make_stream_handler(json_mode: bool) -> logging.Handler:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if json_mode else PrettyFormatter("%(message)s"))
    return h

def _make_file_handler(path: str, json_mode: bool) -> logging.Handler:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(JsonFormatter() if json_mode else PrettyFormatter("%(message)s"))
    return h

def setup_logging(
    level: str | int = None,
    *,
    json_mode: bool | None = None,
    file_path: Optional[str] = None,
) -> None:
    """
    Initialize root logger. Call this once in main.py at startup.

    Args:
        level: e.g., "INFO", "DEBUG". Defaults to settings.LOG_LEVEL.
        json_mode: True -> JSON logs; False -> pretty; None -> auto(JSON in prod).
        file_path: optional rotating logfile path, e.g., "logs/api.log".
    """
    lvl = level or _settings.LOG_LEVEL
    json_enabled = _settings.APP_ENV.lower() != "development" if json_mode is None else json_mode

    logging.captureWarnings(True)  # route warnings to logging
    root = logging.getLogger()
    root.setLevel(lvl)

    # Clear existing handlers (uvicorn might set defaults)
    for h in list(root.handlers):
        root.removeHandler(h)

    root.addHandler(_make_stream_handler(json_enabled))
    if file_path:
        root.addHandler(_make_file_handler(file_path, json_enabled))

    # Quiet noisy libs if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------------
class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)


# -----------------------------------------------------------------------------
# Structured event helpers
# -----------------------------------------------------------------------------
def _log(event: str, level: int, **fields: Any) -> None:
    logger = logging.getLogger("app")
    extra = {"event": event, **fields}
    logger.log(level, fields.get("msg", event), extra=extra)

def log_sql_execution(
    sql: str,
    *,
    row_count: int | None = None,
    duration_ms: int | None = None,
    tables: Iterable[str] | None = None,
    ok: bool = True,
    error: str | None = None,
) -> None:
    _log(
        "sql.execute",
        logging.INFO if ok else logging.ERROR,
        sql=sql,
        row_count=row_count,
        duration_ms=duration_ms,
        tables=list(tables or []),
        ok=ok,
        error=error,
    )

def log_text2sql_event(
    question: str,
    *,
    picked_tables: Iterable[str] | None = None,
    prompt_tokens: int | None = None,
    sql: str | None = None,
    ok: bool = True,
    error: str | None = None,
    duration_ms: int | None = None,
) -> None:
    _log(
        "t2sql.generate",
        logging.INFO if ok else logging.ERROR,
        question=question,
        tables=list(picked_tables or []),
        prompt_tokens=prompt_tokens,
        sql=sql,
        ok=ok,
        error=error,
        duration_ms=duration_ms,
    )

def log_request(event: str, **fields: Any) -> None:
    _log(event, logging.INFO, **fields)


# -----------------------------------------------------------------------------
# FastAPI middleware to attach correlation id & access logs
# -----------------------------------------------------------------------------
def install_fastapi_middleware(app) -> None:
    """
    Usage in src/main.py:

        from src.core.logging_utils import setup_logging, install_fastapi_middleware
        setup_logging(file_path="logs/api.log")
        app = FastAPI()
        install_fastapi_middleware(app)
    """
    from fastapi import Request
    from starlette.responses import Response

    @app.middleware("http")
    async def _logging_middleware(request: Request, call_next):
        # Correlation id: header or new
        cid = request.headers.get("x-correlation-id") or new_correlation_id()

        # Store for this context
        correlation_id.set(cid)

        timer = Timer()
        try:
            response: Response = await call_next(request)
            status = response.status_code
            log_request(
                "http.access",
                method=request.method,
                path=str(request.url.path),
                query=str(request.url.query),
                status=status,
                duration_ms=timer.ms,
                client=str(request.client.host if request.client else "-"),
            )
            # propagate correlation id back
            response.headers["x-correlation-id"] = cid
            return response
        except Exception as e:
            # Ensure we still log with timing & cid
            log_request(
                "http.error",
                method=request.method,
                path=str(request.url.path),
                status=500,
                duration_ms=timer.ms,
                error=f"{type(e).__name__}: {e}",
            )
            raise
