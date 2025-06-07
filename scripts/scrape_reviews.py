import logging
import pandas as pd
from google_play_scraper import Sort, reviews
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    filename='scrape_reviews.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def scrape_bank_reviews(app_id, bank_name, count=500, lang='en', sort=Sort.NEWEST):
    """
    Scrape reviews for a given bank app from Google Play Store.
    Args:
        app_id (str): Google Play app ID.
        bank_name (str): Name of the bank (e.g., 'CBE').
        count (int): Number of reviews to scrape.
        lang (str): Language of reviews.
        sort: Sorting method for reviews.
    Returns:
        list: List of review dictionaries.
    """
    try:
        result, _ = reviews(
            app_id,
            lang=lang,
            sort=sort,
            count=count,
            filter_score_with=None  # Collect all ratings # type: ignore
        )
        logging.info(f"Scraped {len(result)} reviews for {bank_name}")
        return [
            {
                'review': r['content'],
                'rating': r['score'],
                'date': r['at'].strftime('%Y-%m-%d'),
                'bank': bank_name,
                'source': 'Google Play'
            }
            for r in result
        ]
    except Exception as e:
        logging.error(f"Error scraping {bank_name}: {str(e)}")
        return []

def main():
    # Define bank apps and their IDs (to be updated with actual IDs)
    bank_apps = {
        'CBE': 'com.combanketh.mobilebanking',
        'BOA': 'com.boa.boaMobileBanking',
        'Dashen': 'com.dashen.dashensuperapp'
    }


    all_reviews = []
    
    for bank, app_id in bank_apps.items():
        logging.info(f"Starting scrape for {bank}")
        # Scrape in batches to avoid rate limits
        reviews_data = scrape_bank_reviews(app_id, bank, count=500)
        all_reviews.extend(reviews_data)
        time.sleep(2)  # Delay to avoid rate limits
    
    # Convert to DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Basic validation
    missing_data = df.isnull().sum()
    logging.info(f"Missing data: {missing_data.to_dict()}")
    if len(df) >= 1200 and df.isnull().mean().max() < 0.05:
        logging.info(f"Scraped {len(df)} reviews successfully")
    else:
        logging.warning(f"Data quality issue: {len(df)} reviews, missing data {df.isnull().mean().max():.2%}")
    
    # Save to CSV
    df.to_csv('raw_reviews.csv', index=False)
    logging.info("Saved raw reviews to raw_reviews.csv")

if __name__ == "__main__":
    main()