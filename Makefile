dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -p naukriwaala-dev up --build -d

stop:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml -p naukriwaala-dev down

scrape:
	docker compose -f docker-compose.scrape.yml -p naukriwaala-scrape build scraper
	docker compose -f docker-compose.scrape.yml -p naukriwaala-scrape run --rm scraper

ecr-login:
	aws ecr get-login-password --profile ${AWS_PROFILE} --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

ecr-build-push:
	docker buildx build --platform linux/amd64 -f backend/Dockerfile.scraper -t naps-scraper ./backend
	docker tag naps-scraper:latest ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/naps-scraper:latest
	docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/naps-scraper:latest

run-scraper-aws:
	aws ecs run-task \
		--cluster naps-scraper-cluster \
		--task-definition naps-scraper \
		--launch-type FARGATE \
		--network-configuration "awsvpcConfiguration={subnets=[${AWS_SUBNET}],securityGroups=[${AWS_SG}],assignPublicIp=ENABLED}"
