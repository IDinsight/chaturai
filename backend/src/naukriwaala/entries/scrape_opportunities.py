"""This module contains the main entry point for scraping opportunities from
apprenticeshipindia.gov.in.

From the backend directory of this project, this entry point can be invoked from the
command line via:

python -m src.naukriwaala.entries.scrape_opportunities

or

python src/naukriwaala/entries/scrape_opportunities.py
"""

# Standard Library
import os
import signal
import sys

from pathlib import Path

# Third Party Library
import typer

from sqlalchemy import text

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
from naukriwaala.config import Settings
from naukriwaala.db.utils import get_session, test_db_connection
from naukriwaala.scrapers.opportunities_scraper import OpportunitiesScraper
from naukriwaala.utils.general import cleanup
from naukriwaala.utils.logging_ import initialize_logger

assert (
    sys.version_info.major >= 3 and sys.version_info.minor >= 11
), "naukriwaala requires at least Python 3.11!"

# Instantiate typer apps for the command line interface.
cli = typer.Typer()

logger = initialize_logger()


@cli.command()
def main(
    *,
    start_date: str = typer.Option(
        None, "--start-date", help="Start date for scraping (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        None, "--end-date", help="End date for scraping (YYYY-MM-DD)"
    ),
    stop_at_outdated: bool = typer.Option(
        False,
        "--stop-at-outdated",
        help="Stop scraping when reaching outdated opportunities",
    ),
    outdated_after: int = typer.Option(
        Settings.SCRAPER_MAX_VALID_DAYS,
        "--outdated-after",
        help="Number of days after which an opportunity is considered outdated",
    ),
) -> None:
    """Scrape apprenticeship opportunities from apprenticeshipindia.gov.in.

    For regular scraping you should set start_date=None, end_date=None,
    stop_at_outdated=True, and outdated_after=Settings.SCRAPER_MAX_VALID_DAYS.

    For historical data scraping you should set (e.g.) start_date="2023-01-01",
    end_date="2023-12-31", stop_at_outdated=False, and
    outdated_after=Settings.SCRAPER_MAX_VALID_DAYS

    Parameters
    ----------
    start_date
        Start date for scraping in YYYY-MM-DD format. If None, scrape from the most
        recent opportunity.
    end_date
        End date for scraping in YYYY-MM-DD format. If None, scrape until the oldest
        opportunity.
    stop_at_outdated
        If True, stop scraping when reaching opportunities older than `outdated_after`
        days.
    outdated_after
        Number of days after which an opportunity is considered outdated.

    Raises
    ------
    typer.BadParameter
        If `start_date` is provided without `end_date`, or vice versa.
    Exception
        If the scraper encounters an error.
    """

    if (start_date and not end_date) or (end_date and not start_date):
        raise typer.BadParameter(
            "If you provide `start_date`, you must also provide `end_date`."
        )

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    with logger.catch():
        test_db_connection()

        try:
            scraper = OpportunitiesScraper(delay_between_requests=1.0, page_size=100)
            scraper.scrape_opportunities(
                start_date=start_date,
                end_date=end_date,
                stop_at_outdated=stop_at_outdated,
                outdated_after=outdated_after,
            )
        except Exception as e:
            logger.error(f"Scraper failed: {str(e)}")
            raise


if __name__ == "__main__":
    cli()
