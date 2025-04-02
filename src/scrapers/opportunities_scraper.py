import requests
import logging
from datetime import datetime
import time
from typing import Optional, Dict, List
from ..models.base import get_session
from ..models.opportunity import Opportunity, Establishment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpportunitiesScraper:
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
        headers = {'accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            raise
        finally:
            time.sleep(self.delay)  # Rate limiting
    
    def _process_establishment(self, establishment_data: Dict) -> Establishment:
        """Process establishment data and return Establishment instance."""
        establishment = self.session.query(Establishment).filter_by(
            code=establishment_data['code']
        ).first()
        
        if not establishment:
            establishment = Establishment(
                id=establishment_data['code'],  # Using code as ID
                establishment_name=establishment_data['establishment_name'],
                code=establishment_data['code'],
                registration_type=establishment_data['registration_type'],
                working_days=establishment_data['working_days'],
                state_count=establishment_data['state_count']
            )
            self.session.add(establishment)
        
        return establishment
    
    def _process_opportunity(self, opportunity_data: Dict, establishment: Establishment) -> None:
        """Process opportunity data and update database."""
        opportunity = self.session.query(Opportunity).filter_by(
            id=opportunity_data['id']
        ).first()
        
        if opportunity:
            # Update existing opportunity
            opportunity.update_from_api(opportunity_data)
        else:
            # Create new opportunity
            opportunity = Opportunity(
                id=opportunity_data['id'],
                establishment_id=establishment.id
            )
            opportunity.update_from_api(opportunity_data)
            self.session.add(opportunity)
    
    def scrape_opportunities(self) -> None:
        """Main method to scrape all opportunities."""
        current_page = 1
        total_pages = None
        processed_count = 0
        
        try:
            while total_pages is None or current_page <= total_pages:
                logger.info(f"Processing page {current_page}")
                
                # Fetch page of opportunities
                response = self._make_request(
                    'opportunities/search',
                    params={'page': current_page, 'page_size': self.page_size}
                )
                
                # Update total pages on first iteration
                if total_pages is None:
                    total_pages = response['meta']['last_page']
                    logger.info(f"Total pages to process: {total_pages}")
                
                # Process opportunities
                for opp_data in response['data']:
                    try:
                        # Process establishment first
                        establishment = self._process_establishment(opp_data['establishment'])
                        
                        # Process opportunity
                        self._process_opportunity(opp_data, establishment)
                        
                        processed_count += 1
                        
                        # Commit every 100 records
                        if processed_count % 100 == 0:
                            self.session.commit()
                            logger.info(f"Processed {processed_count} opportunities")
                    
                    except Exception as e:
                        logger.error(f"Error processing opportunity {opp_data.get('id')}: {str(e)}")
                        self.session.rollback()
                
                current_page += 1
            
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