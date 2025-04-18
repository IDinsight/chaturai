import requests
import argparse
import logging
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from ..models.base import get_session
from ..models.opportunity import Opportunity, Establishment
from ..settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpportunitiesScraper:
    """Scraper for apprenticeship opportunities from apprenticeshipindia.gov.in.

    This class handles fetching and processing apprenticeship opportunities from the
    Apprenticeship India API, including both opportunities and their associated
    establishments. It manages database operations for storing and updating the data.
    """

    BASE_URL = "https://api.apprenticeshipindia.gov.in"

    def __init__(self, page_size: int = 100, delay_between_requests: float = 1.0):
        """Initialize the scraper with configuration.

        Args:
            page_size: Number of records to fetch per page
            delay_between_requests: Delay in seconds between API requests
        """
        self.page_size = page_size
        self.delay = delay_between_requests
        self.session = get_session()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make an API request with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error making request to {url}: {str(e)}" f"with parameters: {params}"
            )
            raise
        finally:
            time.sleep(self.delay)  # Rate limiting

    def scrape_opportunities(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        stop_at_outdated: bool = True,
        outdated_after: int = settings.MAX_VALID_DAYS,
    ) -> None:
        """Main method to scrape all opportunities.

        The scraper will scrape most recent opportunities first,
        i.e. from end_date to start_date. If end_date is None, we start
        from the most recent opportunity. If start_date is None, we scrape
        until the oldest opportunity unless,
         - start_date is more than outdated_after days ago, and
         - stop_at_outdated is True

        Args:
            start_date (str | None, optional): Start date for scraping.
                Use YYYY-MM-DD format. Defaults to None.
            end_date (str | None, optional): End date for scraping (exclusive).
                Use YYYY-MM-DD format. Defaults to None.
            stop_at_outdated (bool, optional): If True, stop scraping when reaching
                opportunities older than outdated_after days. Defaults to True.
            outdated_after (int, optional): Number of days after the last
                updated date which an opportunity will be considered outdated.
                Defaults to settings.MAX_VALID_DAYS.
        """
        current_page = 1
        total_pages = None
        processed_count = 0
        outdated_from = datetime.now(ZoneInfo(settings.TIMEZONE)) - timedelta(
            days=outdated_after
        )
        start_datetime = (
            datetime.strptime(start_date, "%Y-%m-%d").replace(
                tzinfo=ZoneInfo(settings.TIMEZONE)
            )
            if start_date
            else None
        )
        end_datetime = (
            datetime.strptime(end_date, "%Y-%m-%d").replace(
                tzinfo=ZoneInfo(settings.TIMEZONE)
            )
            if end_date
            else None
        )

        updated_opportunity_ids = set()  # Track which opportunities were updated
        should_continue = True  # Flag to control when to stop scraping

        try:
            while should_continue and (
                total_pages is None or current_page <= total_pages
            ):
                logger.info(f"Processing page {current_page}")

                # Fetch page of opportunities
                response = self._make_request(
                    "opportunities/search",
                    params={"page": current_page, "page_size": self.page_size},
                )

                # Update total pages on first iteration
                if total_pages is None:
                    total_pages = response["meta"]["last_page"]
                    logger.info(f"Total pages to process: {total_pages}")

                # Process opportunities
                for opp_data in response["data"]:
                    try:
                        # Check if opportunity was last updated within the valid period
                        updated_at = datetime.strptime(
                            opp_data["updated_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
                        ).replace(tzinfo=ZoneInfo(settings.TIMEZONE))

                        logger.debug(
                            f"Opportunity {opp_data['id']} updated at {updated_at}"
                        )

                        if start_datetime is not None and updated_at < start_datetime:
                            # i.e. we've hit opportunities older than start_date
                            logger.info(
                                f"Reached opportunities older than {start_datetime} (start_date) at {current_page}. "
                                f"Stopping scrape."
                            )
                            should_continue = False
                            break

                        if end_datetime is not None and updated_at > end_datetime:
                            logger.debug(
                                f"Skipping opportunities newer than {end_datetime} at {current_page}."
                            )
                            continue

                        if stop_at_outdated and updated_at < outdated_from:
                            logger.info(
                                f"Reached opportunities older than {outdated_from} (oudated) at {current_page}. "
                                "Stopping scrape."
                            )
                            should_continue = False
                            break

                        # Process establishment first
                        establishment = (
                            Establishment.create_establishment_if_not_exists(
                                self.session, opp_data["establishment"]
                            )
                        )

                        # Process opportunity
                        Opportunity.create_or_update_opportunity(
                            self.session, opp_data, establishment.id
                        )
                        updated_opportunity_ids.add(opp_data["id"])

                        processed_count += 1
                        # Commit every 100 records
                        if processed_count % 100 == 0:
                            self.session.commit()
                            logger.info(f"Processed {processed_count} opportunities")

                    except Exception as e:
                        logger.error(
                            f"Error processing opportunity {opp_data.get('id')}: {str(e)}"
                        )
                        self.session.rollback()

                if should_continue:
                    current_page += 1

            # Update statuses for all opportunities
            logger.info("Updating opportunity statuses...")
            status_counts = Opportunity.update_opportunity_statuses(
                self.session, updated_opportunity_ids, outdated_from
            )
            logger.info(
                f"Marked {status_counts['outdated']} opportunities as outdated and "
                f"{status_counts['filled']} opportunities as filled"
            )

            # Final commit
            self.session.commit()
            logger.info(f"Completed processing {processed_count} opportunities")

        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            self.session.rollback()
            raise
        finally:
            self.session.close()


def main(
    start_date: str, end_date: str, stop_at_outdated: bool, outdated_after: int
) -> None:
    """Entry point for the scraper."""
    try:
        scraper = OpportunitiesScraper(page_size=100, delay_between_requests=1.0)
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
    parser = argparse.ArgumentParser(
        description="Scrape apprenticeship opportunities from apprenticeshipindia.gov.in"
    )
    parser.add_argument(
        "start_date", type=str, help="Start date for scraping (YYYY-MM-DD)"
    )
    parser.add_argument(
        "end_date", type=str, help="End date for scraping (YYYY-MM-DD)", default=None
    )
    parser.add_argument(
        "--stop_at_outdated",
        action="store_true",
        help="Stop scraping when reaching outdated opportunities",
    )
    parser.add_argument(
        "--outdated_after",
        type=int,
        default=settings.MAX_VALID_DAYS,
        help="Number of days after the last updated date which an opportunity will be considered outdated",
    )

    args = parser.parse_args()
    logger.info("Starting scraper with arguments: %s", args)

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        stop_at_outdated=args.stop_at_outdated,
        outdated_after=args.outdated_after,
    )
    # For regular scraping you should call
    # main(start_date=None, end_date=None, stop_at_outdated=True, outdated_after=settings.MAX_VALID_DAYS)
    # For historical data scraping you should call, for example,
    # main(start_date="2023-01-01", end_date="2023-12-31", stop_at_outdated=False, outdated_after=settings.MAX_VALID_DAYS)
