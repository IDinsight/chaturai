"""This module contains the Alembic environment setup for database migrations."""

# Standard Library
import sys

from logging.config import fileConfig
from pathlib import Path

# Third Party Library
from alembic import context  # type: ignore
from sqlalchemy import create_engine, engine_from_config, pool

# Append the framework path.
env_script_directory = Path(__file__).resolve().parents[1]
package_path = env_script_directory / "src"

if package_path not in sys.path:
    print(f"Appending '{package_path}' to system path...")
    sys.path.append(str(package_path))

# Package Library
from chaturai.config import BackendSettings  # noqa: E402
from chaturai.db.utils import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an Engine is
    acceptable here as well. By skipping the Engine creation we don't even need a DBAPI
    to be available.

    Calls to context.execute() here emit the given string to the script output.
    """

    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In normal migration scenarios, we need to create an Engine and associate a
    connection with the context.

    For `pytest-alembic`, the connection is provided by `pytest-alembic` at runtime.
    Thus, we create a check to see if `connectable` already exists. This allows the same
    `env.py` to be used for both `pytest-alembic` and regular migrations.
    """

    # Override the default sqlalchemy.url with a constructed URL using psycopg2.
    db_url = BackendSettings.create_sync_postgres_db_url()

    connectable = create_engine(db_url)

    if connectable is None:
        connectable = engine_from_config(
            config.get_section(config.config_ini_section, {}),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
