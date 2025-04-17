from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, URL, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from ..settings import app_db_settings

Base = declarative_base()
SYNC_DB_API = "psycopg2"
ASYNC_DB_API = "asyncpg"

# Global so we don't create more than one engine per process.
# Outside of being best practice, this is needed so we can properly pool connections
# and not create a new pool on every request.
_SYNC_ENGINE: Engine | None = None
_ASYNC_ENGINE: AsyncEngine | None = None


def get_connection_url(
    *,
    db: str = app_db_settings.APP_DB,
    db_api: str = ASYNC_DB_API,
    host: str = app_db_settings.APP_DB_HOST,
    password: str = app_db_settings.APP_DB_PASSWORD,
    port: int | str = app_db_settings.APP_DB_PORT,
    user: str = app_db_settings.APP_DB_USER,
) -> URL:
    """Return a connection string for the given database.

    Parameters
    ----------
    db
        The database name.
    db_api
        The database API.
    host
        The database host.
    password
        The database password.
    port
        The database port.
    user
        The database user.

    Returns
    -------
    URL
        A connection string for the given database.
    """

    return URL.create(
        database=db,
        drivername="postgresql+" + db_api,
        host=host,
        password=password,
        port=int(port),
        username=user,
    )


def get_engine() -> Engine:
    """Return a SQLAlchemy engine.

    Returns
    -------
    Engine
        A SQLAlchemy engine.
    """

    global _SYNC_ENGINE
    if _SYNC_ENGINE is None:
        connection_string = get_connection_url(db_api=SYNC_DB_API)
        _SYNC_ENGINE = create_engine(connection_string)
    return _SYNC_ENGINE


def get_sqlalchemy_async_engine() -> AsyncEngine:
    """Return a SQLAlchemy async engine generator.

    Returns
    -------
    AsyncEngine
        A SQLAlchemy async engine.
    """

    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        connection_string = get_connection_url()
        _ASYNC_ENGINE = create_async_engine(
            connection_string, pool_size=app_db_settings.APP_DB_POOL_SIZE
        )
    return _ASYNC_ENGINE


def get_session():
    """Create a new database session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def get_async_engine() -> AsyncEngine:
    """Return a SQLAlchemy async engine generator.

    Returns
    -------
    AsyncEngine
        A SQLAlchemy async engine.
    """

    global _ASYNC_ENGINE
    if _ASYNC_ENGINE is None:
        connection_string = get_connection_url()
        _ASYNC_ENGINE = create_async_engine(
            connection_string, pool_size=app_db_settings.APP_DB_POOL_SIZE
        )
    return _ASYNC_ENGINE
