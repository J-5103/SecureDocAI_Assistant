from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PORT: int = 8080
    DATA_FILE: str = "./data/quickchat.json"

    TEXT_TO_SQL_PROVIDER: str = "none"     # openai | ollama | none
    OPENAI_API_KEY: str | None = None
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "sqlcoder:latest"

    DATABASE_URL: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
