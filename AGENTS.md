# Agent Handbook

## Directory Structure
- `app.py` - Dash application entry point
- `src/` - Application modules and utilities
- `tests/` - Unit and integration tests
- `notebooks/` - Example analysis notebooks
- `Dockerfile.jupyter` - Container image for Jupyter environment

## Coding Standards
- Use **Python 3.9**.
- Follow [PEP8](https://peps.python.org/pep-0008/) style guidelines.
- Prefer type hints and dataclasses for new code.
- Keep functions small with clear docstrings.

## Build & Test Commands
- Install dependencies:
  ```bash
  pip install -r requirements.txt
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
1. Create a clean Python 3.9 environment or use `Dockerfile.jupyter`.
2. Install dependencies and dev tools as above.
3. Run the test suite to verify the setup.
4. Start the app with `python app.py` or via Docker.
5. Commit code with concise messages describing the change.

