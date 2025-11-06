from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI and LLM credentials and connection settings
    OPENAI_API_URL: str
    OPENAI_API_KEY: str
    LLM_MODEL_NAME: str

    # The Redis server connection URL, used for application state, caching, and memory persistence
    REDIS_URL: str

    # Generation parameters
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1024

    # Default system prompt
    SYSTEM_PROMPT: str = "You are a helpful and concise AI assistant."


# Create a single instance of settings for the entire application
settings = Settings()  # type: ignore
