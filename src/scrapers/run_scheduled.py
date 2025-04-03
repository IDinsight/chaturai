import schedule
import time
import logging
from datetime import datetime
from .opportunities_scraper import main as run_scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def job():
    """Run the scraper job with error handling."""
    try:
        logger.info("Starting scheduled scraping job")
        start_time = datetime.now()
        
        run_scraper()
        
        duration = datetime.now() - start_time
        logger.info(f"Completed scraping job in {duration}")
    except Exception as e:
        logger.error(f"Error in scraping job: {str(e)}", exc_info=True)

def main():
    """Main function to run the scheduler."""
    # Schedule job to run daily at 1 AM
    schedule.every().day.at("01:00").do(job)
    
    # Run job immediately on startup
    logger.info("Running initial scraping job")
    job()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main() 