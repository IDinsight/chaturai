"""This module contains the main entry point for the backend application.

From the backend directory of this project, this entry point can be invoked from the
command line via:

python -m src.naukriwaala.entries.main

or

python src/naukriwaala/entries/main.py

If the `naukriwaala` package has been pip installed, then the entry point can be
invoked from the command line via:

naukriwaala --help
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
    PACKAGE_PATH_ROOT = str(Path(__file__).resolve())
    PACKAGE_PATH_SPLIT = PACKAGE_PATH_ROOT.split(os.path.join("backend"))
    PACKAGE_PATH = Path(PACKAGE_PATH_SPLIT[0]) / "backend" / "src"
    if PACKAGE_PATH not in sys.path:
        print(f"Appending '{PACKAGE_PATH}' to system path...")
        sys.path.append(str(PACKAGE_PATH))

# Package Library
from naukriwaala import create_app
from naukriwaala.config import Settings

assert (
    sys.version_info.major >= 3 and sys.version_info.minor >= 11
), "naukriwaala requires at least Python 3.11!"

# Instantiate typer apps for the command line interface.
cli = typer.Typer()

# Globals.
PATHS_BACKEND_ROOT = Settings.PATHS_BACKEND_ROOT
app = create_app()


class Worker(UvicornWorker):
    """Custom worker class to allow `root_path` to be passed to Uvicorn."""

    CONFIG_KWARGS = {"root_path": PATHS_BACKEND_ROOT}


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
        "naukriwaala.entries.main:app",
        host=host,
        port=port,
        log_config=None,  # Disable Uvicorn's default logging config
        log_level=Settings.LOGGING_LOG_LEVEL.lower(),
        reload=reload,
        reload_dirs=[str(project_dir / "backend" / "src")],
    )


if __name__ == "__main__":
    cli()
