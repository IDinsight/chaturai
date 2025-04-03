from pydantic import Field
from pydantic_settings import BaseSettings


class DefaultSettings(BaseSettings):
    """
    Default settings of env variables for the application
    """

    LOG_LEVEL: str = "INFO"
    DOMAIN: str = "localhost"
    BACKEND_ROOT_PATH: str = ""
    API_KEY: str


class AppDBSettings(BaseSettings):
    """Application database settings"""

    APP_DB_SYNC_API: str = Field("psycopg2", validation_alias="POSTGRES_SYNC_API")
    APP_DB_ASYNC_API: str = Field("asyncpg", validation_alias="POSTGRES_ASYNC_API")
    APP_DB_TYPE: str = Field("postgresql", validation_alias="POSTGRES_DB_TYPE")
    APP_DB_USER: str = Field("postgres", validation_alias="POSTGRES_USER")
    APP_DB_PASSWORD: str = Field("password", validation_alias="POSTGRES_PASSWORD")
    APP_DB_HOST: str = Field("localhost", validation_alias="POSTGRES_DB_HOST")
    APP_DB_PORT: str = Field("5432", validation_alias="POSTGRES_DB_PORT")
    APP_DB: str = Field("app_db", validation_alias="POSTGRES_DB")
    APP_DB_POOL_SIZE: int = Field(
        10, validation_alias="POSTGRES_DB_POOL_SIZE"
    )  # Number of connections in the pool


settings = DefaultSettings()
app_db_settings = AppDBSettings()
