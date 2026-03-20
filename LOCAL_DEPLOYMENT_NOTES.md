# Local Deployment Notes: asta-paper-finder

**Repo:** https://github.com/allenai/asta-paper-finder
**Machine:** MacBook Pro, Apple Silicon M3 Max, 128 GB RAM, macOS 14 (Sonoma)
**Date documented:** 2026-03-19

---

## Project Overview

PaperFinder is a paper-seeking agent that helps locate academic papers based on natural language queries. It is implemented as a FastAPI server that exposes a REST API at `/api/2/rounds`.

The architecture is a pipeline of LLM-driven components:
1. A **query analyzer** that parses the user query into structured objects.
2. An **execution planner** that routes the query to one of several search workflows.
3. Multiple **search agents** (keyword, dense/Vespa, snowball citation, metadata-based, author-based, etc.).
4. **Relevance judges** using LLMs to score retrieved abstracts/snippets.
5. A **ranking/sorting** step that combines content relevance with other signals.

The codebase is a monorepo using `uv` as the package manager. It contains:
- `agents/mabool/api/` — the main FastAPI server
- `libs/chain/` — LLM abstraction layer (OpenAI + Google Gemini via LangChain)
- `libs/config/` — TOML-based configuration with secrets loading
- `libs/dcollection/` — Document collection backed by Semantic Scholar API + Vespa
- `libs/di/` — Dependency injection framework
- `libs/common/` — Shared utilities

---

## What Worked

1. **Cloning:** Repository cloned successfully from GitHub.

2. **Dependency installation:** All Python dependencies installed via `uv sync --all-packages --dev`. This pulled:
   - 157 packages including PyPI packages and 3 git-sourced packages:
     - `semanticscholar` from `allenai/s2-api-client`
     - `mabwiser` from `allenai/mabwiser`
     - `microsoft-python-type-stubs` from Microsoft
   - Build was clean; no compilation errors on Apple Silicon.

3. **Python runtime:** `uv` resolved to Python 3.13.9 (from Anaconda). The repo requests 3.12.8 in per-subproject `.python-version` files, but the root `pyproject.toml` only requires `>=3.12.8`, so 3.13.9 works and `uv` was separately given 3.12.8 via `uv python install 3.12.8`.

4. **App import:** `from mabool.api.app import create_app` imports cleanly with no errors.

5. **App object creation:** `create_app()` completes successfully — the `FastAPI` app instance is built and config is loaded from TOML.

6. **Server startup attempt:** Gunicorn + Uvicorn worker starts, creates the app, then attempts to build the singleton DI scope — this is where the first hard blocker is hit.

---

## What Failed

### Blocker 1: `s2_api_key not found in config` (server crashes on startup)

**When:** During application startup, when the Uvicorn worker initializes the singleton DI scope.

**Error:**
```
KeyError: 's2_api_key not found in config'
ai2i.di.interface.errors.ProviderBuildError: 's2_api_key not found in config'
ai2i.di.interface.errors.ManagedScopeError: Failed opening scope: 'singleton',
  error in building the provider: mabool.utils.dc_deps.round_doc_collection_factory,
  cause: 's2_api_key not found in config'
```

**Root cause:** In `agents/mabool/api/mabool/utils/dc_deps.py:12`, the DI provider `round_doc_collection_factory` declares `s2_api_key: str = DI.config(cfg_schema.s2_api_key)` — no `default=None`. This means the DI framework requires the key to be present in config at startup, even though the underlying `DocumentCollectionFactory.__init__` accepts `None` and falls back to the unauthenticated S2 public API.

**Severity:** Fatal — prevents the server from starting at all. No API calls can be made until this is resolved.

**Fix (requires API key):** See "Required API Keys" section. Alternatively, a developer could patch `dc_deps.py` to add `default=None` but this changes behavior (unauthenticated S2 requests are heavily rate-limited to ~100 req/5min).

---

### Anticipated Blocker 2: LLM API keys (at query execution time)

Once the server starts (with S2 key), actual search queries will fail without LLM keys. The code in `mabool/utils/llm_utils.py` returns `None` gracefully for missing keys — but downstream LangChain / OpenAI / Google GenAI clients will fail when they try to make API calls.

Models used:
- **OpenAI** — GPT-4o, GPT-4 Turbo, and `gpt5mini-*` (reasoning models) for query analysis, relevance judgement, metadata planner, broad search, and keyword search agents.
- **Google Gemini** — `gemini-2.0-flash` and `gemini3flash-*` for LLM suggestion and specific-paper-by-title agents.
- **Cohere** — `rerank-english-v3.0` for re-ranking retrieved papers (initialized in singleton scope but with `default=None`, so it won't crash startup).

---

### Note on Vespa (Dense Retrieval)

The code references a Vespa backend (`libs/dcollection/ai2i/dcollection/external_api/dense/vespa.py`) for dense vector retrieval. This is AllenAI's internal Semantic Scholar Vespa index. There is no public endpoint for this. Dense-retrieval-based workflows will fail silently or error, but keyword/snowball workflows may still function.

---

## Required API Keys

| Key | Variable | Required When | Where to Get | Free Tier? |
|-----|----------|---------------|--------------|------------|
| Semantic Scholar | `S2_API_KEY` | **Server startup** (fatal without it) | https://www.semanticscholar.org/product/api#api-key-form | Yes (free tier, higher limits with key) |
| OpenAI | `OPENAI_API_KEY` | Query execution (most agents) | https://platform.openai.com/api-keys | No (paid) |
| Google AI | `GOOGLE_API_KEY` | Query execution (Gemini agents) | https://aistudio.google.com/app/apikey | Yes (free quota) |
| Cohere | `COHERE_API_KEY` | Query execution (reranking) | https://dashboard.cohere.com/api-keys | Yes (trial key) |

### Key Descriptions

- **S2_API_KEY**: Access to Semantic Scholar's academic paper graph API. Used for paper search, citation lookup, author search, and snippet/dense retrieval. Without an API key the public rate limit is very low (~100 requests per 5 min per IP). The server DI framework **requires** this key to be present in config at startup — there is no default/fallback.

- **OPENAI_API_KEY**: Access to OpenAI models. The config references internal model names like `gpt5mini-medium-reasoning-default` and `gpt5mini-minimal-reasoning-default` which appear to map to reasoning-enabled versions of GPT-4o Mini or future GPT-5 Mini. Standard GPT-4o (`gpt-4o-2024-11-20`) is also used. **Note:** The model name aliases (`gpt5mini-*`) are custom and handled by the chain library's `LLMModel` class — check `libs/chain/ai2i/chain/models.py` if these model names fail with OpenAI.

- **GOOGLE_API_KEY**: Access to Google Gemini models via the `google-genai` SDK. Used for `gemini-2.0-flash` and `gemini3flash-*` in the LLM suggestion and specific-paper-by-title agents.

- **COHERE_API_KEY**: Access to Cohere's rerank API. Used by `CohereRerankScorer` with model `rerank-english-v3.0` to rerank retrieved papers by relevance. The DI binding has `default=None` so the server **can start** without this, but reranking will fail at query time.

---

## How to Configure Secrets

Create the file `agents/mabool/api/conf/.env.secret` (a template has been created at this path):

```ini
S2_API_KEY=your_s2_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

The config loader (`libs/config/ai2i/config/loading.py:load_secrets_file`) reads this file automatically. Keys are loaded into the config dict and also pushed into environment variables for compatibility.

---

## Exact Command to Start the Service (Once Keys Are Ready)

```bash
cd <repo-root>/agents/mabool/api
APP_CONFIG_ENV=dev <repo-root>/.venv/bin/gunicorn \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --timeout 0 \
    --bind 0.0.0.0:8000 \
    --enable-stdio-inheritance \
    --access-logfile - \
    --reload \
    --env 'APP_CONFIG_ENV=dev' \
    'mabool.api.app:create_app()'
```

Or via the project's Makefile (from `agents/mabool/api/`):
```bash
cd <repo-root>/agents/mabool/api
<repo-root>/.venv/bin/uv run make start-dev
```

Once the server is running:
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health (returns HTTP 204)
- Root redirect: http://localhost:8000/ → /docs

### Example API Call

```bash
curl -X POST http://localhost:8000/api/2/rounds \
  -H "Content-Type: application/json" \
  -d '{
    "paper_description": "transformer attention mechanisms for NLP",
    "operation_mode": "fast"
  }'
```

---

## Step-by-Step Runbook

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Get API keys** (see "Required API Keys" table above). At minimum, `S2_API_KEY` is mandatory. The Semantic Scholar API key is free to apply for.

### Steps

1. **Clone the repo** (skip if already done):
   ```bash
   git clone https://github.com/allenai/asta-paper-finder <repo-root>
   ```

2. **Install Python 3.12.8** (required by subprojects):
   ```bash
   cd <repo-root>
   uv python install 3.12.8
   ```

3. **Install dependencies**:
   ```bash
   cd <repo-root>
   uv sync --all-packages --dev
   ```
   This creates `.venv/` in the repo root.

4. **Create the secrets file**:
   ```bash
   # Edit the template that was already created:
   nano <repo-root>/agents/mabool/api/conf/.env.secret
   # Fill in real API key values.
   ```

5. **Verify the app loads** (quick sanity check, does not require keys):
   ```bash
   cd <repo-root>/agents/mabool/api
   APP_CONFIG_ENV=dev <repo-root>/.venv/bin/python -c \
     "from mabool.api.app import create_app; app = create_app(); print('OK')"
   ```

6. **Start the server**:
   ```bash
   cd <repo-root>/agents/mabool/api
   APP_CONFIG_ENV=dev <repo-root>/.venv/bin/gunicorn \
       -k uvicorn.workers.UvicornWorker \
       --workers 1 \
       --timeout 0 \
       --bind 0.0.0.0:8000 \
       --enable-stdio-inheritance \
       --access-logfile - \
       --reload \
       --env 'APP_CONFIG_ENV=dev' \
       'mabool.api.app:create_app()'
   ```

7. **Verify health**:
   ```bash
   curl -i http://localhost:8000/health
   # Expect: HTTP/1.1 204 No Content
   ```

8. **Make a test query**:
   ```bash
   curl -X POST http://localhost:8000/api/2/rounds \
     -H "Content-Type: application/json" \
     -d '{"paper_description": "self-attention in transformers", "operation_mode": "fast"}'
   ```

---

## Apple Silicon / macOS-Specific Notes

- **No architecture issues observed.** All packages installed cleanly for `aarch64` (arm64). Binary wheels were available for all major packages (`numpy`, `scipy`, `pandas`, `scikit-learn`, `pydantic-core`, `cryptography`, `tokenizers`, `pillow`, `matplotlib`).

- **Python version note:** The repo was written for Python 3.12.8. Python 3.13.9 (Anaconda) is present on the machine and was used by `uv` for the virtual environment since the root `pyproject.toml` only requires `>=3.12.8`. This worked without issue. If you want strict 3.12.8 compliance, run `uv venv -p 3.12.8` before `uv sync`.

- **`uv` is required.** The project uses `uv` workspaces and the lockfile (`uv.lock`) should be used as-is for reproducible builds.

- **`gunicorn` on macOS:** Works fine on Apple Silicon. The `uvicorn.workers.UvicornWorker` is the correct ASGI adapter.

- **No GPU required.** All LLM inference is done via external APIs (OpenAI, Google, Cohere). No local models are used.

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Secrets file (create this) | `agents/mabool/api/conf/.env.secret` |
| Main config | `agents/mabool/api/conf/config.toml` |
| Fast mode config overlay | `agents/mabool/api/conf/config.extra.fast_mode.toml` |
| FastAPI app entry point | `agents/mabool/api/mabool/api/app.py` |
| Server startup script | `agents/mabool/api/dev.sh` |
| API key lookup logic | `agents/mabool/api/mabool/utils/llm_utils.py` |
| S2 API key DI binding (blocker) | `agents/mabool/api/mabool/utils/dc_deps.py` |
| Cohere client | `agents/mabool/api/mabool/external_api/rerank/cohere.py` |
| Virtual environment | `.venv/` (repo root) |

---

## Summary

| Step | Status | Notes |
|------|--------|-------|
| Clone repo | Done | Content copied from git clone |
| Inspect structure | Done | Monorepo with uv workspaces |
| Install Python 3.12.8 | Done | Via `uv python install 3.12.8` |
| Install dependencies | Done | 157 packages, no errors on Apple Silicon |
| Find all env vars | Done | 4 API keys required |
| Create `.env.secret` template | Done | At `agents/mabool/api/conf/.env.secret` |
| Boot server | BLOCKED | `S2_API_KEY` required at startup |
| Reach `/health` | Not yet | Blocked by above |
| Make a query | Not yet | Also needs OpenAI + Cohere + Google keys |
