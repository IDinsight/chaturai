"""This module contains the main configurations for the backend.

Any configurations added to backend/.env should be added to `BackendSettings` as well.
"""

# Standard Library
import os

from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Third Party Library
import requests

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Pydantic settings for backend."""

    # Internal settings
    _INTERNAL_GRAPH_MAPPING: Optional[dict[str, Any]] = None

    # Agents
    AGENTS_CHATUR_AGENT: str = "chatur-agent"
    AGENTS_LOGIN_STUDENT: str = "login-student-agent"
    AGENTS_PROFILE_COMPLETION: str = "profile-completion-agent"
    AGENTS_REGISTER_STUDENT: str = "register-student-agent"

    # Chat
    CHAT_ENV: str = "dev"

    # FastAPI
    FASTAPI_API_KEY: str

    # Graphs
    GRAPHS_CHATUR_AGENT: str = "Chatur_Agent_Graph"
    GRAPHS_LOGIN_STUDENT: str = "Login_Student_Graph"
    GRAPHS_PROFILE_COMPLETION: str = "Profile_Completion_Graph"
    GRAPHS_REGISTER_STUDENT: str = "Register_Student_Graph"

    # LiteLLM
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "dummy-key")
    LITELLM_ENDPOINT: str = os.getenv("LITELLM_ENDPOINT", "http://localhost:4000")
    LITELLM_MODEL_CHAT: str = os.getenv("LITELLM_MODEL_CHAT", "openai/chat")
    LITELLM_MODEL_DEFAULT: str = os.getenv("LITELLM_MODEL_DEFAULT", "openai/default")
    LITELLM_MODEL_EMBEDDING: str = os.getenv(
        "LITELLM_MODEL_EMBEDDING", "openai/embedding"
    )

    # Logfire #
    LOGFIRE_ENVIRONMENT: str = "dev"
    LOGFIRE_READ_TOKEN: str = os.getenv("LOGFIRE_READ_TOKEN", "")
    LOGFIRE_SEND_TO_LOGFIRE: bool = False
    LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN", "")

    # Logging
    LOGGING_LOG_LEVEL: str = "INFO"

    # Models
    MODELS_EMBEDDING_OPENAI: str = "openai/text-embedding-3-large"
    MODELS_EMBEDDING_ST: str = "sentence-transformers/all-mpnet-base-v2"
    MODELS_LLM: str = "openai/gpt-4o-mini"

    # OpenAI #
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Playwright
    PLAYWRIGHT_HEADLESS: bool = False
    PLAYWRIGHT_PAGE_TTL: int = 600

    # Postgres
    POSTGRES_ASYNC_API: str = Field("asyncpg", validation_alias="POSTGRES_ASYNC_API")
    POSTGRES_DB: str = Field("chaturai", validation_alias="POSTGRES_DB")
    POSTGRES_DB_POOL_SIZE: int = Field(
        10, validation_alias="POSTGRES_DB_POOL_SIZE"
    )  # Number of connections in the pool
    POSTGRES_DB_TYPE: str = Field("postgresql", validation_alias="POSTGRES_DB_TYPE")
    POSTGRES_HOST: str = Field("localhost", validation_alias="POSTGRES_HOST")
    POSTGRES_PASSWORD: str = Field("postgres", validation_alias="POSTGRES_PASSWORD")
    POSTGRES_PORT: str = Field("5432", validation_alias="POSTGRES_PORT")
    POSTGRES_SYNC_API: str = Field("psycopg2", validation_alias="POSTGRES_SYNC_API")
    POSTGRES_USER: str = Field("postgres", validation_alias="POSTGRES_USER")

    # Prometheus
    PROMETHEUS_MULTIPROC_DIR: str = "/tmp"

    # Redis
    REDIS_CACHE_PREFIX_BROWSER_STATE: str = os.getenv(
        "REDIS_CACHE_PREFIX_BROWSER_STATE", "browser_state"
    )
    REDIS_CACHE_PREFIX_CHAT: str = os.getenv("REDIS_CACHE_PREFIX_CHAT", "chat_sessions")
    REDIS_CACHE_PREFIX_CHAT_CLEANED: str = os.getenv(
        "REDIS_CACHE_PREFIX_CHAT_CLEANED", "chat_sessions_cleaned"
    )
    REDIS_CACHE_PREFIX_GRAPH_CHATUR: str = os.getenv(
        "REDIS_CACHE_PREFIX_GRAPH_CHATUR", "graph_chatur"
    )
    REDIS_CACHE_PREFIX_CHATUR_AGENT: str = (
        f"{REDIS_CACHE_PREFIX_GRAPH_CHATUR}_Chatur_Agent"
    )
    REDIS_CACHE_PREFIX_LOGIN_STUDENT: str = (
        f"{REDIS_CACHE_PREFIX_GRAPH_CHATUR}_Login_Student"
    )
    REDIS_CACHE_PREFIX_PROFILE_COMPLETION: str = (
        f"{REDIS_CACHE_PREFIX_GRAPH_CHATUR}_Profile_Completion"
    )
    REDIS_CACHE_PREFIX_REGISTER_STUDENT: str = (
        f"{REDIS_CACHE_PREFIX_GRAPH_CHATUR}_Register_Student"
    )
    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis://localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # Sentry
    SENTRY_DSN: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 1.0

    # Scraper
    SCRAPER_MAX_VALID_DAYS: int = 182
    SCRAPER_TIMEZONE: str = "Asia/Kolkata"

    # Text generation parameters.
    TEXT_GENERATION_BEDROCK: dict[str, Any] = {
        "frequency_penalty": 0.0,
        "n": 1,
        "presence_penalty": 0.0,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    TEXT_GENERATION_DEFAULT: dict[str, Any] = {
        "frequency_penalty": 0.0,
        "n": 1,
        "presence_penalty": 0.0,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    TEXT_GENERATION_GEMINI: dict[str, Any] = {
        "frequency_penalty": 0.0,
        "n": 1,
        "presence_penalty": 0.0,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    TEXT_GENERATION_OPENAI: dict[str, Any] = {
        "frequency_penalty": 0.0,
        "n": 1,
        "presence_penalty": 0.0,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @classmethod
    def create_sync_postgres_db_url(cls) -> str:
        """Create the synchronous PostgreSQL database URL.

        Returns
        -------
        str
            The PostgreSQL database URL.
        """

        return f"postgresql+{cls().POSTGRES_SYNC_API}://{cls().POSTGRES_USER}:{cls().POSTGRES_PASSWORD}@{cls().POSTGRES_HOST}:{cls().POSTGRES_PORT}/{cls().POSTGRES_DB}"

    @classmethod
    def get_model_info_from_litellm(cls, *, litellm_model_name: str) -> dict[str, Any]:
        """Get model information from the LiteLLM endpoint.

        NB: This method should remain within `BackendSettings` since it requires the
        class itself. Moving this to another module can inadvertently cause circular
        imports since `Settings` is imported in a lot of other modules.

        Parameters
        ----------
        litellm_model_name
            The name of the LiteLLM model (e.g., chat).

        Returns
        -------
        dict[str, Any]
            Dictionary containing the model information.

        Raises
        ------
        KeyError
            If the model name is not in the LiteLLM endpoint.
        """

        model_info_endpoint = f"{cls().LITELLM_ENDPOINT.rstrip('/')}/model/info"
        model_info = requests.get(
            model_info_endpoint, headers={"accept": "application/json"}, timeout=600
        ).json()
        for dict_ in model_info["data"]:
            if dict_["model_name"] == litellm_model_name:
                return dict_
        raise KeyError(f"Model {litellm_model_name} not found in LiteLLM endpoint.")

    @field_validator(
        "MODELS_EMBEDDING_OPENAI", "MODELS_EMBEDDING_ST", "MODELS_LLM", mode="before"
    )
    @classmethod
    def validate_embedding_model_name(cls, value: str) -> str:
        """Ensure that the model names starts with either 'openai/' or
        'sentence-transformers/'.

        Parameters
        ----------
        value
            The model name to validate.

        Returns
        -------
        str
            The validated model name.

        Raises
        ------
        ValueError
            If the model name does not start with the allowed prefixes.
        """

        allowed_prefixes = ("openai/", "sentence-transformers/")
        if not value.startswith(allowed_prefixes):
            raise ValueError(
                f"Invalid model name: '{value}'. "
                f"Must start with one of {allowed_prefixes}."
            )
        return value

    @field_validator("*", mode="before")
    @classmethod
    def validate_litellm_models(cls, value: Any, info: Any) -> Any:
        """Validate all fields that start with "LITELLM_MODEL_".

        Parameters
        ----------
        value
            The value to validate.
        info
            Metadata about the field being validated, including its name.

        Returns
        -------
        Any
            The validated value if it passes the checks.

        Raises
        ------
        ValueError
            If the value does is an empty string or does not start with "openai/".
        """

        if info.field_name.startswith("LITELLM_MODEL_") and not value.startswith(
            "openai/"
        ):
            raise ValueError(
                f"{info.field_name} must be a non-empty string that starts with "
                f"'openai/'."
            )
        return value

    @field_validator("SCRAPER_TIMEZONE")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        """Make sure timezone value is a valid input for `ZoneInfo`.

        Parameters
        ----------
        v
            The timezone string to validate.

        Returns
        -------
        str
            The validated timezone string.

        Raises
        ------
        ValueError
            If the timezone string is invalid or not found.
        """

        try:
            ZoneInfo(v)
            return v
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Invalid timezone: {v}") from exc


Settings: BackendSettings = BackendSettings()
