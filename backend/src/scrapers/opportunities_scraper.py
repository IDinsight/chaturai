import requests
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

    def scrape_opportunities(self) -> None:
        """Main method to scrape all opportunities."""
        current_page = 1
        total_pages = None
        processed_count = 0
        six_months_ago = datetime.now(ZoneInfo(settings.TIMEZONE)) - timedelta(
            days=settings.MAX_VALID_DAYS
        )  # 6 months ago
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
                        # Check if opportunity is older than 6 months
                        created_at = datetime.strptime(
                            opp_data["created_at"]["date"], "%Y-%m-%d %H:%M:%S.%f"
                        ).replace(tzinfo=ZoneInfo(settings.TIMEZONE))
                        if created_at < six_months_ago:
                            logger.info(
                                f"Reached opportunities older than 6 months at page {current_page}. "
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
                self.session, updated_opportunity_ids, six_months_ago
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


def main():
    """Entry point for the scraper."""
    try:
        scraper = OpportunitiesScraper(page_size=100, delay_between_requests=1.0)
        scraper.scrape_opportunities()
    except Exception as e:
        logger.error(f"Scraper failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
