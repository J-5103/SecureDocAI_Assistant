import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import text, inspect
from .config import settings

_engine: AsyncEngine | None = None

def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine

@asynccontextmanager
async def db_conn():
    eng = get_engine()
    async with eng.connect() as conn:
        yield conn

async def reflect_schema() -> dict[str, list[str]]:
    """
    Return {table: [cols...]} for the connected DB.
    """
    eng = get_engine()
    async with eng.connect() as conn:
        # SQLAlchemy async inspector still uses sync engine beneath for reflection.
        sinsp = inspect(conn.sync_connection().engine)
        schema: dict[str, list[str]] = {}
        for t in sinsp.get_table_names():
            cols = [c["name"] for c in sinsp.get_columns(t)]
            schema[t] = cols
        return schema

async def run_select(sql: str, params: dict, max_rows: int = 200):
    """
    Run parameterized SELECT and bound result to dicts.
    """
    if "limit" not in sql.lower():
        sql = f"{sql.rstrip()} LIMIT {max_rows}"
    async with db_conn() as conn:
        res = await conn.execute(text(sql), params)
        rows = [dict(r._mapping) for r in res]
    return rows
