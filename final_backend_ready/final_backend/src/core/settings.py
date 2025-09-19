# src/core/settings.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional
from urllib.parse import quote_plus

try:
    # If python-dotenv is installed, load .env automatically
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _get_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(val: Optional[str], default: List[str] | None = None) -> List[str]:
    if not val:
        return list(default or [])
    return [p.strip() for p in val.split(",") if p.strip()]


def _normalize_url(u: Optional[str]) -> str:
    if not u:
        return ""
    return u.rstrip("/")


def _build_mssql_url_from_parts() -> str:
    """
    Build a SQLAlchemy MSSQL URL with pyodbc.
    Priority order:
      1) DATABASE_URL (if provided)
      2) Compose from DB_* parts (MSSQL only)
    """
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct

    host = os.getenv("DB_HOST", "dev-data.crmemperor.com")
    port = os.getenv("DB_PORT", "1433")
    name = os.getenv("DB_NAME", "CRM_00001")
    user = os.getenv("DB_USER", "dhruv")
    pwd = os.getenv("DB_PASSWORD", "5GN0QHvn2dXMphk")

    # Common Windows driver; change if you use v18 or Linux.
    driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

    # URL-encode the driver
    driver_enc = quote_plus(driver)
    # Example:
    # mssql+pyodbc://USER:PWD@HOST:1433/DBNAME?driver=ODBC+Driver+17+for+SQL+Server
    return f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{name}?driver={driver_enc}"


class Settings:
    # --- App ---
    APP_NAME: str = os.getenv("APP_NAME", "SecureDocAI API")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = _get_bool(os.getenv("DEBUG"), default=True)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- API prefix (if reverse-proxied under a base path) ---
    API_PREFIX: str = os.getenv("API_PREFIX", "/api")

    # --- MSSQL only ---
    DATABASE_URL: str = _build_mssql_url_from_parts()
    SQL_ECHO: bool = _get_bool(os.getenv("SQL_ECHO"), default=False)

    # Connection pool / timeouts
    POOL_SIZE: int = int(os.getenv("POOL_SIZE", "10"))
    POOL_MAX_OVERFLOW: int = int(os.getenv("POOL_MAX_OVERFLOW", "20"))
    POOL_RECYCLE_SECS: int = int(os.getenv("POOL_RECYCLE_SECS", "1800"))
    DB_CONNECT_TIMEOUT: int = int(os.getenv("DB_CONNECT_TIMEOUT", "15"))
    STMT_TIMEOUT_SECS: int = int(os.getenv("STMT_TIMEOUT_SECS", "20"))

    # --- Text-to-SQL safety & scope ---
    SQL_ROW_LIMIT: int = int(os.getenv("SQL_ROW_LIMIT", "100"))      # used to inject TOP
    ALLOW_SCHEMAS: List[str] = _split_csv(os.getenv("ALLOW_SCHEMAS", "dbo"))
    READONLY_ENFORCED: bool = _get_bool(os.getenv("READONLY_ENFORCED"), default=True)
    SQL_BLOCKLIST: List[str] = _split_csv(
        os.getenv("SQL_BLOCKLIST", "INSERT,UPDATE,DELETE,DROP,ALTER,TRUNCATE,MERGE,EXEC,xp_")
    )

    # --- LLM (Ollama) for Text-to-SQL ---
    OLLAMA_URL: str = _normalize_url(os.getenv("OLLAMA_URL") or "http://192.168.0.88:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-7b-sql:latest")
    ENABLE_FEWSHOTS: bool = _get_bool(os.getenv("ENABLE_FEWSHOTS"), default=True)

    # --- Schema/Catalog cache ---
    SCHEMA_CACHE_TTL_SECS: int = int(os.getenv("SCHEMA_CACHE_TTL_SECS", "600"))
    CATALOG_PATH: str = os.getenv("CATALOG_PATH", "data/catalog.json")
    CATALOG_TTL_SECS: int = int(os.getenv("CATALOG_TTL_SECS", "1200"))

    # --- CORS ---
    ALLOWED_ORIGINS: List[str] = _split_csv(os.getenv("ALLOWED_ORIGINS", "*"))
    ALLOW_CREDENTIALS: bool = _get_bool(os.getenv("ALLOW_CREDENTIALS"), default=True)
    ALLOWED_METHODS: List[str] = _split_csv(os.getenv("ALLOWED_METHODS", "GET,POST,PUT,PATCH,DELETE,OPTIONS"))
    ALLOWED_HEADERS: List[str] = _split_csv(os.getenv("ALLOWED_HEADERS", "*"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
