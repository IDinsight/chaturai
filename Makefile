dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -p naukriwaala-dev up --build -d

stop:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -p naukriwaala-dev down

scrape:
	docker compose -f docker-compose.scrape.yml -p naukriwaala-scrape build scraper
	docker compose -f docker-compose.scrape.yml -p naukriwaala-scrape run --rm scraper
