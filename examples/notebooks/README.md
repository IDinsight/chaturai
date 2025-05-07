# Data processing for opportunities recommendation engine

1. Clean positive examples data
   1. ❎ Link the students and opportunities data to the contract data.
      - can't link opportunities with contracts?
   2. De-duplicate (columns and rows)
   3. Add a label column with value "positive"
2. Clean negative examples data
   1. ✅ Scrape older opportunities
   2. Query opportunities data from Naukriwaala DB
   3. Add a label column with value "negative"
3. Create a full dataset
   1. Union 1 & 2
   2. De-duplicate based on opportunity code
4. Create a cross-validation set with 5+ folds
5. Decide on evaluation metrics
   - recall@3
6. Create student profile and preferences data structure
7. Write a rule based model and check cross validation score
8. ✅ Set up scraper
