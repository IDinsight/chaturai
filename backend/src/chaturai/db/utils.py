"""This module contains utilities for databases."""

# pylint: disable=global-statement
# Standard Library
from collections.abc import AsyncGenerator

# Third Party Library
from loguru import logger
from sqlalchemy import URL, Engine, MetaData, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session

# Package Library
from chaturai.config import Settings

POSTGRES_ASYNC_API = Settings.POSTGRES_ASYNC_API
POSTGRES_DB = Settings.POSTGRES_DB
POSTGRES_DB_POOL_SIZE = Settings.POSTGRES_DB_POOL_SIZE
POSTGRES_HOST = Settings.POSTGRES_HOST
POSTGRES_PASSWORD = Settings.POSTGRES_PASSWORD
POSTGRES_PORT = Settings.POSTGRES_PORT
POSTGRES_SYNC_API = Settings.POSTGRES_SYNC_API
POSTGRES_USER = Settings.POSTGRES_USER

# Global so we don't create more than one engine per process. Outside of being best
# practice, this is needed so we can properly pool connections and not create a new
# pool on every request.
_ASYNC_ENGINE: AsyncEngine | None = None
_SYNC_ENGINE: Engine | None = None


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


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
            connection_string, pool_size=POSTGRES_DB_POOL_SIZE
        )
    return _ASYNC_ENGINE


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a SQLAlchemy async session.

    Yields
    ------
    AsyncGenerator[AsyncSession, None]
        An async session generator.
    """

    async with AsyncSession(
        get_async_engine(), expire_on_commit=False
    ) as async_session:
        yield async_session


def get_connection_url(
    *,
    db: str = POSTGRES_DB,
    db_api: str = POSTGRES_ASYNC_API,
    host: str = POSTGRES_HOST,
    password: str = POSTGRES_PASSWORD,
    port: int | str = POSTGRES_PORT,
    user: str = POSTGRES_USER,
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
        connection_string = get_connection_url(db_api=POSTGRES_SYNC_API)
        _SYNC_ENGINE = create_engine(connection_string)
    return _SYNC_ENGINE


def get_session() -> Session:
    """Return a SQLAlchemy session.

    Returns
    -------
    Session
        A SQLAlchemy session.
    """

    return Session(get_engine(), expire_on_commit=False)


def test_db_connection() -> None:
    """Test the database connection before scraping.

    Raises
    ------
    Exception
        If the database connection fails.
    """

    try:
        session = get_session()
        session.execute(text("SELECT 1 FROM public.opportunities"))
        logger.info("Database connection successful.")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise
    finally:
        session.close()
