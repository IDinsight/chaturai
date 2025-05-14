"""This module contains the main entry point for the backend application.

From the backend directory of this project, this entry point can be invoked from the
command line via:

python -m src.chaturai.entries.main

or

python src/chaturai/entries/main.py

If the `chaturai` package has been pip installed, then the entry point can be
invoked from the command line via:

chaturai --help
"""

# Standard Library
import os
import sys

from pathlib import Path

# Third Party Library
import typer
import uvicorn

from loguru import logger
from uvicorn_worker import UvicornWorker

# Append the framework path. NB: This is required if this entry point is invoked from
# the command line. However, it is not necessary if it is imported from a pip install.
if __name__ == "__main__":
    # Path to src directory containing chaturai
    package_path = Path(__file__).resolve().parents[2]

    if package_path not in sys.path:
        print(f"Appending '{package_path}' to system path...")
        sys.path.append(str(package_path))

# Package Library
from chaturai import create_app
from chaturai.config import Settings

assert (
    sys.version_info.major >= 3 and sys.version_info.minor >= 11
), "chaturai requires at least Python 3.11!"

# Instantiate typer apps for the command line interface.
cli = typer.Typer()


app = create_app()


class Worker(UvicornWorker):
    """Custom worker class to allow `root_path` to be passed to Uvicorn."""

    CONFIG_KWARGS = {"root_path": Settings.PATHS_BACKEND_ROOT}


@cli.command()
def start(host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> None:
    """Start the FastAPI application using Uvicorn.

    The process is as follows:

    1. Run the Uvicorn server.

    Parameters
    ----------
    host
        The host address to bind the server to.
    port
        The port number to bind the server to.
    reload
        Specifies whether the server should automatically reload when changes are
        detected.
    """

    logger.info("Starting FastAPI with loguru...")

    # 1.
    project_dir = Path(os.getenv("PATHS_PROJECT_DIR", ""))
    assert project_dir.is_dir(), f"'{project_dir}' is not a directory."
    uvicorn.run(
        "chaturai.entries.main:app",
        host=host,
        port=port,
        log_config=None,  # Disable Uvicorn's default logging config
        log_level=Settings.LOGGING_LOG_LEVEL.lower(),
        reload=reload,
        reload_dirs=[str(project_dir / "backend" / "src")],
        root_path=Settings.PATHS_BACKEND_ROOT,
    )


if __name__ == "__main__":
    cli()
