# Makefile for Fantasy Premier League Project

.PHONY: run install test clean

# Variables
VENV = venv
PYTHON = $(VENV)\Scripts\python
PIP = $(VENV)\Scripts\pip

.PHONY: install run clean

# Set up virtual environment and install dependencies
install:
	python3 -m venv $(VENV) 
	$(PIP) install -r requirements.txt
	.\venv\Scripts\activate
	

# Run the application
run:
		uvicorn app:app --reload --host 127.0.0.1 --port 8000


# Run tests
test:
	$(PYTHON) -m pytest tests/

# Clean up temporary files and virtual environment
clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Format code using black
format:
	$(PYTHON) -m black .

# Lint code using flake8
lint:
	$(PYTHON) -m flake8 app/

# Run security checks
security:
	$(PYTHON) -m bandit -r app/