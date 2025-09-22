# src/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # Server
    PORT: int = 8080

    # ---- Files & Directories (now project-relative, not absolute drive paths) ----
    # Optional override for the data folder; defaults to "<repo>/src/data"
    DATA_DIR: str | None = None
    # Legacy quickchat storage file name (kept for compatibility). Will be placed under DATA_DIR.
    DATA_FILE: str = "./data/quickchat.json"
    # Optional override for the schema/catalog JSON; if not set, "<DATA_DIR>/catalog.json"
    CATALOG_PATH: str | None = None

    # ---- Model / Text-to-SQL provider ----
    TEXT_TO_SQL_PROVIDER: str = "none"  # openai | ollama | none
    OPENAI_API_KEY: str | None = None
    OLLAMA_HOST: str = "http://192.168.0.88:11434/api/generate"
    OLLAMA_MODEL: str = "sqlcoder:7b"
    MODEL_NAME: str | None = None  # optional alias

    # ---- Database ----
    # e.g. mssql+pyodbc://user:pass@SERVER/DB?driver=ODBC+Driver+17+for+SQL+Server
    #      postgresql+psycopg2://user:pass@host:5432/db
    #      sqlite:///./local.db
    DATABASE_URL: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ---------- Computed helpers (paths are resolved & directories ensured) ----------
    @property
    def APP_ROOT(self) -> str:
        # folder where this file lives (usually src/)
        return str(Path(__file__).resolve().parent)

    @property
    def DATA_DIR_RESOLVED(self) -> str:
        d = self.DATA_DIR or str(Path(self.APP_ROOT) / "data")
        Path(d).mkdir(parents=True, exist_ok=True)
        return d

    @property
    def QUICKCHAT_DATA_FILE(self) -> str:
        # Always place the quickchat file inside DATA_DIR (ignore nested relative prefixes)
        p = Path(self.DATA_FILE)
        if not p.is_absolute():
            p = Path(self.DATA_DIR_RESOLVED) / p.name
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def CATALOG_PATH_RESOLVED(self) -> str:
        p = Path(self.CATALOG_PATH) if self.CATALOG_PATH else Path(self.DATA_DIR_RESOLVED) / "catalog.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def DB_FAMILY(self) -> str:
        u = (self.DATABASE_URL or "").lower()
        if u.startswith("mssql"):
            return "mssql"
        if u.startswith("postgres") or u.startswith("postgresql"):
            return "postgres"
        if u.startswith("sqlite"):
            return "sqlite"
        return "unknown"

    def ensure_dirs(self):
        # Touch directories so other modules don't crash on import
        Path(self.DATA_DIR_RESOLVED).mkdir(parents=True, exist_ok=True)
        Path(self.QUICKCHAT_DATA_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(self.CATALOG_PATH_RESOLVED).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
# Ensure folders exist at import time so nothing tries to use an absolute E:\ path
settings.ensure_dirs()
