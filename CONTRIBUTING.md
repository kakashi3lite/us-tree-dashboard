# Contributing

## Development Setup
1. Use Python 3.11.
2. Create virtualenv and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests before committing:
   ```bash
   pytest
   ```

## Style
- Follow PEP8 and include type hints and docstrings.
- Format with `black` and import-sort with `isort` (see CI).

## Commit & PR Guidelines
- Use imperative commit messages (e.g., `fix: handle empty dataset`).
- Open PRs against `main` or `develop`; ensure CI passes.
- Label issues with `bug`, `enhancement`, or `documentation` as appropriate.

