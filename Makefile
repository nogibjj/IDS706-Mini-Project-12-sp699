install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

lint:
	ruff *.py

test:
	python -m pytest -vv --cov=main test_*.py

deploy:
	# deploy goes here
		
all: install format lint test