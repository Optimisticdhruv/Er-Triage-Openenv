.PHONY: install run test docker-build docker-run baseline leaderboard validate deploy inference

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 7860

test:
	pytest tests/ -v --tb=short

docker-build:
	docker build -t er-triage-openenv .

docker-run:
	docker run -p 7860:7860 --env-file .env er-triage-openenv

baseline:
	python baseline/run_baseline.py --model gpt-4o-mini --seed 42

leaderboard:
	python baseline/run_baseline.py --leaderboard --seed 42

inference:
	python inference.py

validate:
	openenv validate --url http://localhost:7860

deploy:
	python deploy_to_hf.py
