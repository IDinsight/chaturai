"""This module contains the scraper for apprenticeship opportunities from
apprenticeshipindia.gov.in.
"""

# Standard Library
import time

from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

# Third Party Library
import requests

from loguru import logger

# Package Library
from chaturai.config import Settings
from chaturai.db.opportunity import Establishment, Opportunity
from chaturai.db.utils import get_session

SCRAPER_MAX_VALID_DAYS = Settings.SCRAPER_MAX_VALID_DAYS
SCRAPER_TIMEZONE = Settings.SCRAPER_TIMEZONE


class OpportunitiesScraper:
    """Scraper for apprenticeship opportunities from apprenticeshipindia.gov.in.

    This class handles fetching and processing apprenticeship opportunities from the
    Apprenticeship India API, including both opportunities and their associated
    establishments. It manages database operations for storing and updating the data.
    """

    BASE_URL = "https://api.apprenticeshipindia.gov.in"

    def __init__(self, *, delay_between_requests: float = 1.0, page_size: int = 100):
        """Initialize the scraper with configuration.

        Parameters
        ----------
        delay_between_requests
            Delay in seconds between API requests
        page_size
            Number of records to fetch per page
        """

        self.delay = delay_between_requests
        self.page_size = page_size
        self.session = get_session()

    def _make_request(
        self, *, endpoint: str, params: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Make an API request with rate limiting.

        Parameters
        ----------
        endpoint
            The API endpoint to request, e.g. "opportunities/search"
        params
            Dictionary to send in the query string for the request.

        Returns
        -------
        Mapping[str, Any]
            The JSON response from the API as a dictionary.

        Raises
        ------
        requests.exceptions.RequestException
            If there is an error making the request.
        """

        url = f"{self.BASE_URL}/{endpoint}"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=600)
            response.raise_for_status()
            response_json = response.json()
            assert isinstance(response_json, dict)
            return response_json
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error making request with parameters: {params} to {url}: {str(e)}"
            )
            raise
        finally:
            time.sleep(self.delay)  # Rate limiting

    def scrape_opportunities(  # pylint:disable=R0915,R1260
        self,
        *,
        end_date: str | None = None,
        outdated_after: int = SCRAPER_MAX_VALID_DAYS,
        start_date: str | None = None,
        stop_at_outdated: bool = True,
    ) -> None:
        """Scrape all opportunities.

        The scraper will scrape most recent opportunities first, i.e. from end_date to
        start_date. If end_date is None, we start from the most recent opportunity. If
        start_date is None, we scrape until the oldest opportunity unless:
            1. start_date is more than outdated_after days ago, and
            2. stop_at_outdated is True

        Parameters
        ----------
        start_date
            Start date for scraping. Use YYYY-MM-DD format.
        end_date
            End date for scraping (exclusive). Use YYYY-MM-DD format.
        stop_at_outdated
            If True, stop scraping when reaching opportunities older than
            outdated_after days.
        outdated_after
            Number of days after the last updated date which an opportunity will be
            considered outdated.
        """

        tz = ZoneInfo(SCRAPER_TIMEZONE)
        start_datetime = (
            datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
            if start_date
            else None
        )
        end_datetime = (
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz)
            if end_date
            else None
        )
        outdated_from = datetime.now(tz) - timedelta(days=outdated_after)

        current_page = 1
        processed_count = 0
        should_continue = True  # Flag to control when to stop scraping
        total_pages = None
        updated_opportunity_ids = set()  # Track which opportunities were updated

        try:
            while should_continue and (
                total_pages is None or current_page <= total_pages
            ):
                logger.info(f"Processing page: {current_page}/{total_pages}")

                # Fetch page of opportunities.
                response = self._make_request(
                    endpoint="opportunities/search",
                    params={"page": current_page, "page_size": self.page_size},
                )

                # Update total pages on first iteration.
                if total_pages is None:
                    total_pages = response["meta"]["last_page"]
                    logger.info(f"Total pages to process: {total_pages}")

                # Process opportunities.
                for opp_data in response["data"]:
                    try:
                        # Check if opportunity was last updated within the valid period.
                        updated_at = datetime.strptime(
                            opp_data["updated_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
                        ).replace(tzinfo=tz)

                        logger.debug(
                            f"Opportunity {opp_data['id']} updated at {updated_at}"
                        )

                        # This occurs when we've hit opportunities older than
                        # start_date.
                        if start_datetime is not None and updated_at < start_datetime:
                            logger.info(
                                f"Reached opportunities older than {start_datetime} "
                                f"(start_date) at {current_page}. Stopping scrape."
                            )
                            should_continue = False
                            break

                        # This occurs when we've hit opportunities newer than end_date.
                        if end_datetime is not None and updated_at > end_datetime:
                            logger.debug(
                                f"Skipping opportunities newer than {end_datetime} "
                                f"(end_date) at {current_page}."
                            )
                            continue

                        # This occurs when we've hit opportunities older than
                        # outdated_from days.
                        if stop_at_outdated and updated_at < outdated_from:
                            logger.info(
                                f"Reached opportunities older than {outdated_from} at "
                                f"{current_page}. Stopping scrape."
                            )
                            should_continue = False
                            break

                        # Process establishment first.
                        establishment = (
                            Establishment.create_establishment_if_not_exists(
                                establishment_data=opp_data["establishment"],
                                session=self.session,
                            )
                        )

                        # Process opportunity.
                        Opportunity.create_or_update_opportunity(
                            establishment_id=establishment.id,
                            opportunity_data=opp_data,
                            session=self.session,
                        )

                        updated_opportunity_ids.add(opp_data["id"])
                        processed_count += 1

                        # Commit every 100 records.
                        if processed_count % 100 == 0:
                            self.session.commit()
                            logger.info(f"Processed {processed_count} opportunities.")

                    except Exception as e:  # pylint: disable=W0718
                        logger.error(
                            f"Error processing opportunity {opp_data.get('id')}: "
                            f"{str(e)}"
                        )
                        self.session.rollback()

                if should_continue:
                    current_page += 1

            # Update statuses for all opportunities.
            logger.info("Updating opportunity statuses...")
            status_counts = Opportunity.update_opportunity_statuses(
                cutoff_date=outdated_from,
                session=self.session,
                updated_opportunity_ids=updated_opportunity_ids,
            )
            logger.info(
                f"Marked {status_counts['outdated']} opportunities as outdated and "
                f"{status_counts['filled']} opportunities as filled."
            )

            # Final commit.
            self.session.commit()
            logger.info(f"Completed processing {processed_count} opportunities.")

        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            self.session.rollback()
            raise
        finally:
            self.session.close()
