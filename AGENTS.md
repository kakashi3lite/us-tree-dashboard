# Agent Handbook

## Directory Structure
- `app.py` - Dash application entry point
- `src/` - Application modules and utilities
- `tests/` - Unit and integration tests
- `notebooks/` - Example analysis notebooks
- `Dockerfile.jupyter` - Container image for Jupyter environment

## Coding Standards
- Use **Python 3.11**.
- Follow [PEP8](https://peps.python.org/pep-0008/) style guidelines.
- Prefer type hints and dataclasses for new code.
- Keep functions small with clear docstrings.

## Environment Setup
1. Create a Python 3.11 venv or use Docker.
2. `pip install -r requirements.txt`.
3. Run `pytest` or `docker-compose up`.
4. Start the app with `python app.py` or `docker-compose up`.

## Build & Test Commands
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Install dev tools:
  ```bash
  pip install pytest pytest-cov openai
  ```
- Run tests:
  ```bash
  pytest
  ```
- Launch the dashboard locally:
  ```bash
  python app.py
  ```

## Prompt Patterns
- Include file paths and function names in prompts when requesting changes.
- Break large tasks into sequenced steps.

## Workflow
1. Create a clean Python 3.11 environment or use `Dockerfile.jupyter`.
2. Install dependencies and dev tools as above.
3. Run the test suite to verify the setup.
4. Start the app with `python app.py` or via Docker.
5. Commit code with concise messages describing the change.

## Codex Environment
- Use the `setup_env.sh` script to prepare the environment. It verifies
  Python 3.11, creates a virtual environment, installs dependencies, and
  runs the tests.
- Configure Codex projects to execute this script during setup to ensure
  a consistent environment.

