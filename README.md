# Naukriwaala

Your guide for your apprenticeship journey (for the National Apprenticeship Promotion Scheme)

## Overview

This project includes a data scraper for apprenticeship opportunities from the National Apprenticeship Promotion Scheme (NAPS) portal. It fetches opportunities daily and stores them in a database for further processing and recommendation purposes.

## Architecture

- **Database**: PostgreSQL on Amazon RDS
- **Scraper**: Python-based scraper running on AWS ECS (Fargate)
- **Scheduling**: AWS EventBridge for daily execution
- **Monitoring**: CloudWatch for logs and metrics

## Setup

1. Create a `.env` file with your database configuration:
```
DATABASE_URL=postgresql://user:password@your-rds-instance:5432/naukriwaala
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
alembic upgrade head
```

## Local Development

To run the scraper locally:

```bash
python -m src.scrapers.opportunities_scraper
```

## AWS Deployment

1. Create an ECR repository:
```bash
aws ecr create-repository --repository-name naukriwaala-scraper
```

2. Build and push the Docker image:
```bash
aws ecr get-login-password --region region | docker login --username AWS --password-stdin your-account-id.dkr.ecr.region.amazonaws.com
docker build -t naukriwaala-scraper .
docker tag naukriwaala-scraper:latest your-account-id.dkr.ecr.region.amazonaws.com/naukriwaala-scraper:latest
docker push your-account-id.dkr.ecr.region.amazonaws.com/naukriwaala-scraper:latest
```

3. Create an ECS Task Definition (sample provided in `ecs-task-definition.json`)

4. Create an ECS Cluster and Service

5. Set up EventBridge rule for daily execution

## Database Schema

The database consists of two main tables:

1. `opportunities`: Stores apprenticeship opportunities
2. `establishments`: Stores information about organizations offering apprenticeships

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT
