# Repository Guidelines

## Project Structure & Module Organization
- Core app: `ai_celebrity_app.py` (entry point, scheduling, CLI flags).
- Config models: `ai_celebrity_config.py`.
- Image generation: `image_generator.py`.
- Instagram API client: `instagram_poster.py`.
- Prompt helpers: `prompt_generator.py`.
- Utility scripts/tests: `simple_api_test.py`, `test_facebook_simple.py`, `quick_start.py`.
- Assets/logs: generated images `*.jpg`, logs `ai_celebrity.log`, posting history `post_history.json`, config `config.json`.

## Build, Test, and Development Commands
- Setup env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Validate setup: `python ai_celebrity_app.py --validate`
- Single post (smoke): `python ai_celebrity_app.py --test-feed` or `--test-story`
- Run scheduler: `python ai_celebrity_app.py`
- API smoke tests: `python simple_api_test.py` (uses `.env` keys; saves images).

## Coding Style & Naming Conventions
- Language: Python 3.10+; 4‑space indentation; UTF‑8.
- Types and docstrings: prefer type hints and module/class/function docstrings.
- Naming: modules `snake_case.py`, functions/vars `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Logging: use `logging` (see `ai_celebrity_app.py`) over prints for library code.
- Optional tooling (recommended locally): `black` (format), `ruff` (lint); do not introduce new tooling in CI without discussion.

## Testing Guidelines
- No formal test runner yet; use:
  - `--validate`, `--test-feed`, `--test-story` for end‑to‑end checks.
  - `simple_api_test.py` for API reachability (requires keys).
- If adding unit tests, place in `tests/` or `*_test.py` with `pytest` or `unittest` style; keep tests isolated and avoid hitting external APIs by default (use fakes/mocks, gate live calls behind env flags).

## Commit & Pull Request Guidelines
- Use small, focused commits. Recommended: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
- PRs must include: summary, rationale, runnable steps (`pip install -r requirements.txt`, exact command), and screenshots or sample outputs when changing generation/posting behavior.
- Link related issues; note any config/env changes.

## Security & Configuration Tips
- Store secrets in `.env` (loaded via `python-dotenv` in scripts); never hardcode or commit tokens/keys.
- Required env examples: `STABILITY_API_KEY`, `FAL_API_KEY`, Instagram Graph credentials (`FACEBOOK_PAGE_ID`, business ID, access token).
- Avoid committing generated images/logs; add to `.gitignore` if needed.
- Be cautious with cloud credentials and file paths; prefer environment variables over hardcoded paths.

