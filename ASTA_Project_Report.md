# ASTA Paper Finder — Comprehensive Technical Report

**Project Path:** `/Users/ipanda/Documents/Code/asta`
**Report Date:** 2026-03-21

---

## Table of Contents

1. Project Overview
2. Project Structure
3. Tech Stack & Dependencies
4. Entry Points & How the App Starts
5. Architecture & Main Components
6. Data Flow & Connectivity
7. Configuration System
8. Database Models & Schema
9. API Endpoints
10. Frontend Components
11. Key Algorithms & Business Logic
12. Testing Setup
13. Build & Deployment
14. Known Limitations & Risks
15. Key Files Summary
16. Data Structures & Models
17. External Integrations
18. Summary

---

## 1. Project Overview

**ASTA Paper Finder** is an AI-driven academic paper search engine that helps users locate scholarly papers based on natural language queries. It is a fork of [allenai/asta-paper-finder](https://github.com/allenai/asta-paper-finder) with local deployment enhancements including a session-based SQLite backend and a web UI.

**Project Type:** FastAPI-based REST API microservice with LLM agent orchestration

**Total Codebase:** ~215 Python files (~10,000 lines in agents, ~100 in libs), 528 MB on disk (includes .venv)

**Key Location:** `/Users/ipanda/Documents/Code/asta`

---

## 2. Project Structure

```
asta/
├── agents/                          # Main application
│   └── mabool/api/                  # FastAPI web service
│       ├── conf/                    # Configuration (TOML + .env.secret)
│       ├── mabool/                  # Package code
│       │   ├── api/                 # Route handlers
│       │   │   ├── app.py           # FastAPI entry point
│       │   │   ├── round_v2_routes.py  # Search API
│       │   │   └── history_routes.py   # Session history API (new)
│       │   ├── agents/              # 14 search agents
│       │   ├── data_model/          # Pydantic data models
│       │   ├── infra/               # Infrastructure (Operative pattern, StateManager)
│       │   ├── services/            # DI module setup
│       │   ├── external_api/        # Cohere rerank integration
│       │   └── utils/               # Utilities
│       ├── static/                  # Web UI (index.html)
│       └── search_history.db        # SQLite database (generated at runtime)
│
├── libs/                            # Reusable libraries
│   ├── config/          (ai2i-config)       # TOML config + secrets loading
│   ├── di/              (ai2i-di)           # Async-aware dependency injection
│   ├── chain/           (ai2i-chain)        # LLM abstraction (OpenAI + Google)
│   ├── dcollection/     (ai2i-dcollection)  # Document collection + S2 API
│   └── common/          (ai2i-common)       # Shared utilities
│
├── pyproject.toml                   # uv workspace root
├── uv.lock                          # Dependency lock file
├── Makefile                         # Build commands
├── README.md
├── LOCAL_DEPLOYMENT_NOTES.md        # Deployment guide (new)
└── TECHNICAL_MANUAL.md              # Architecture docs (new, Chinese)
```

---

## 3. Tech Stack & Dependencies

| Layer | Technology |
|-------|-----------|
| **Web Framework** | FastAPI 0.115.6 + Uvicorn 0.34 + Gunicorn 23.0 |
| **LLM Access** | LangChain-Core 1.0, OpenAI SDK (gpt-4o, gpt-5-mini), Google GenAI (Gemini) |
| **Paper Data** | Semantic Scholar API (via `semanticscholar` client, git-sourced) |
| **Reranking** | Cohere Rerank v3.0 |
| **Config** | TOML-based with Python ContextVar isolation |
| **DI Framework** | Custom `ai2i-di` (async-aware, scope-based) |
| **Caching** | `aiocache` (file + memory), `cachetools` (TTL caches) |
| **Database** | SQLite3 (session/search history) |
| **Data Models** | Pydantic 2.10.4 |
| **Async** | asyncio, anyio, aiofiles |
| **Package Manager** | uv (Python) |
| **Code Quality** | Ruff, Pyright, Flake8 |
| **Testing** | pytest, pytest-asyncio, pytest-snapshot |

**Key Python Version:** 3.12.8+ (repo uses 3.13.9 at runtime)

---

## 4. Entry Points & How the App Starts

**Main Entry:** `agents/mabool/api/mabool/api/app.py`

The app factory function `create_app()` performs the following steps in sequence:

1. **Load configuration** from `conf/config.toml` and any `conf/config.extra.*.toml` overlay files.
2. **Initialize dependency injection context** using `create_app_context(services_module)`.
3. **Create a FastAPI app** with managed DI scopes via `create_managed_app(...)`.
4. **Setup CORS** — all origins allowed (`"*"`).
5. **Setup error handlers** and basic routes (`/health`, `/docs`, `/`).
6. **Include routers:**
   - `/api/2/rounds` — paper search endpoint
   - `/api/sessions` — session history management (new)
   - Streaming routes (if enabled)
7. **Mount static files** at `/ui` (the web frontend).
8. **Return the app** ready for Gunicorn/Uvicorn to serve.

**Startup Command:**

```bash
cd agents/mabool/api
APP_CONFIG_ENV=dev gunicorn \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 --bind 0.0.0.0:8000 \
    'mabool.api.app:create_app()'
```

**Access Points:**

| Endpoint | URL |
|----------|-----|
| Web UI | `http://localhost:8000/ui/` |
| API Swagger Docs | `http://localhost:8000/docs` |
| Health Check | `http://localhost:8000/health` |
| Search API | `POST http://localhost:8000/api/2/rounds` |
| Session API | `http://localhost:8000/api/sessions` |

---

## 5. Architecture & Main Components

### 5.1 Core Pipeline Architecture

```
User Query (natural language)
    ↓
[Query Analyzer Agent] — LLM (GPT-5 Mini) parses intent
    ↓
QueryAnalysisResult: query_type, extracted_fields, specifications
    ↓
[Router] — Routes by query_type + operation_mode
    ├── BROAD_BY_DESCRIPTION
    │   ├── fast      → FastBroadSearchAgent
    │   └── diligent  → BroadSearchAgent
    ├── SPECIFIC_BY_TITLE   → SpecificPaperByTitleAgent
    ├── SPECIFIC_BY_NAME    → SpecificPaperByNameAgent
    ├── BY_AUTHOR           → SearchByAuthorsAgent
    ├── METADATA_ONLY       → MetadataOnlySearchAgent
    └── REFUSAL             → QueryRefusalAgent
    ↓
[Search Agents] — Execute workflows, return DocumentCollection
    ↓
[Relevance Judgment] — LLM scores abstracts (GPT-5 Mini minimal reasoning)
    ↓
[Cohere Rerank] — Re-rank by relevance score
    ↓
[Sorting] — Weighted combination of relevance + recency + centrality
    ↓
[Explain] — LLM generates response_text summary
    ↓
JSON Response with ranked papers + metadata
```

### 5.2 The 14 Search Agents

All agents live under `agents/mabool/api/mabool/agents/`. Each is a folder with its own module:

1. **paper_finder/** — Main orchestrator. Routes queries to appropriate sub-agents via `PaperFinderAgent`. `run_agent()` is the entry point function.

2. **query_analyzer/** — Query understanding. `decompose_and_analyze_query_restricted()` uses an LLM to extract structured fields (content, authors, venues, time ranges, domains, refusal signals). Returns a `QueryAnalysisResult`.

3. **broad_search_by_keyword/** — Generates search terms via LLM and queries the Semantic Scholar keyword search API. Implemented by `BroadSearchByKeywordAgent`.

4. **complex_search/** — Multi-turn iterative search. `BroadSearchAgent` tries up to 3 rounds (diligent mode); `FastBroadSearchAgent` combines Dense + Snowball + Keyword in a single pass.

5. **dense/** — Vector semantic search. `DenseAgent` uses an internal Vespa index (only available within AllenAI infrastructure; skipped in local deployments).

6. **specific_paper_by_title/** — Exact title matching. `SpecificPaperByTitleAgent` uses Gemini LLM to match titles to a corpus_id.

7. **specific_paper_by_name/** — Citation reference matching. `SpecificPaperByNameAgent` resolves informal paper references by name.

8. **search_by_authors/** — Author-based search. `SearchByAuthorsAgent` uses an LLM to disambiguate author names, then queries the Semantic Scholar author API.

9. **metadata_only/** — Pure metadata filtering. `MetadataOnlySearchAgent` applies filters directly; `MetadataPlannerAgent` uses an LLM to plan the filtering operations.

10. **llm_suggestion/** — Direct LLM suggestions. `get_llm_suggested_papers()` prompts a reasoning model to guess corpus_ids directly without API search.

11. **snowball/** — Citation expansion. Expands a set of anchor papers by following citation links forward (papers citing the anchor) and backward (papers the anchor cites).

12. **by_citing_papers/** — Reverse citation tracking. `BroadBySpecificPaperCitationAgent` builds a candidate set from papers citing known relevant works.

13. **query_refusal/** — Query rejection. Handles queries that should not be answered, categorized as: similar_to, web_access, not_paper_finding, affiliation, or author_id queries.

14. **common/** — Shared utilities used by all agents: `computed_fields/` (relevance scoring), `sorting.py` (ranking), `explain.py` (response text generation), `common.py` (date/author filtering).

### 5.3 The Operative Pattern (Agent Framework)

**File:** `agents/mabool/api/mabool/infra/operatives/operatives.py`

All agents inherit from a generic base class:

```python
class Operative[INPUT, OUTPUT, STATE]:
    async def handle_operation(
        self, state: STATE | None, inputs: INPUT
    ) -> tuple[STATE | None, OperativeResponse[OUTPUT]]:
        ...

    def init_operative(self, cls, ...) -> SubOperative:
        # Create nested/child agent
        ...
```

`OperativeResponse[T]` is a sealed union type:
- `VoidResponse` — agent produced nothing
- `PartialResponse[T]` — intermediate result, more to come
- `CompleteResponse[T]` — final result

**StateManager:** In-memory TTLCache (24-hour TTL) for multi-turn conversation state.

### 5.4 Dependency Injection Framework

**Library:** `libs/di/` (`ai2i-di`)

A custom async-aware DI system with scopes. Scopes control when providers are created and destroyed:

- **singleton** — Created once at app startup, shared across all requests.
- **round_scope** — Created fresh per request (per `POST /api/2/rounds` call), isolated between concurrent requests.
- **transient** — Created fresh every time the dependency is requested.

```python
# Define a module and register providers
module = create_module("MyModule")

@module.provides(scope="singleton")
async def my_service(...) -> MyService:
    return MyService()

@module.global_init()
async def init_services(service: MyService = DI.requires(...)):
    await service.initialize()
```

---

## 6. Data Flow & Connectivity

### 6.1 Full Request Flow

```
[Client] POST /api/2/rounds
         { paper_description, operation_mode, inserted_before,
           anchor_corpus_ids, read_results_from_cache }
    ↓
[FastAPI] round_v2_routes.py::start_round()
    ↓
[Cache Check] FileBasedCache (key = hash(query, mode, anchor_ids))
    ├─ Hit  → return cached result immediately
    └─ Miss → continue
    ↓
[PrioritySemaphore] Limit to 3 concurrent requests
    ↓
[run_round_with_cache]
    ├─ Generate conversation_thread_id (UUID)
    ├─ Build PaperFinderInput
    └─ Call run_agent() → PaperFinderAgent.handle_operation()
         ↓
    [Query Analyzer] decompose_and_analyze_query_restricted(query)
         ├─ LLM: gpt5mini-medium-reasoning
         └─ Returns QueryAnalysisResult
         ↓
    [Router] Match query_type + operation_mode to sub-agent
         ↓
    [Sub-Agent Execution] e.g., FastBroadSearchAgent:
         ├─ [DenseAgent]              Vespa search (skipped locally)
         ├─ [SnowballAgent]           S2 citations + references
         ├─ [BroadSearchByKeyword]    LLM search terms → S2 keyword API
         └─ Merge all results → DocumentCollection (100–1000 candidates)
         ↓
    [Relevance Judgment]
         ├─ Model: gpt5mini-minimal-reasoning
         ├─ Batch size: adaptive growth (quota = 250 papers)
         ├─ Concurrency: up to 75 parallel LLM requests
         └─ Assigns relevance_judgement.score ∈ [0, 1]
         ↓
    [Cohere Rerank] (if API key configured)
         ├─ Model: rerank-english-v3.0
         └─ Assigns rerank_score ∈ [0, 1]
         ↓
    [Sorting]
         └─ final_agent_score = relevance×0.6 + recency×0.2 + centrality×0.2
         ↓
    [Explain]
         ├─ Select top-K papers (response_text_top_k = 10)
         └─ LLM generates natural language summary for user
         ↓
    [Return Response]
    {
      doc_collection: { documents: [...] },
      response_text: "Natural language summary...",
      input_query: "...",
      analyzed_query: {...},
      metrics: {...},
      token_breakdown_by_model: {...},
      session_id: "thrd:uuid"
    }
    ↓
[MaboolCallbackHandler] Records token usage per model
    ↓
[Cache Write] Stores result in FileBasedCache
    ↓
[Response to Client]
```

### 6.2 Document Collection & Semantic Scholar API Integration

**Library:** `libs/dcollection/` (`ai2i-dcollection`)

The `Document` dataclass is the central data unit:

```python
@dataclass
class Document:
    corpus_id: CorpusId
    title: str
    authors: list[Author]
    year: int | None
    abstract: str | None
    venue: str | None
    citation_count: int | None
    influential_citation_count: int | None
    references: list[CorpusId] | None
    citations: list[CorpusId] | None
    snippets: list[Snippet] | None
    relevance_judgement: Relevance | None
    rerank_score: float | None
    final_agent_score: float | None
```

`DocumentCollection` holds a list of `Document` objects and supports lazy field loading:

| Field Set | Included Fields |
|-----------|-----------------|
| `BASIC_FIELDS` | corpus_id, title, year, authors |
| `UI_REQUIRED_FIELDS` | + abstract, venue, citations, URL |
| `FULL_FIELDS` | + snippets, references |

**S2 API Endpoints Used:**

| Endpoint | Purpose |
|----------|---------|
| `GET /paper/search` | Keyword-based paper search |
| `GET /paper/{id}/citations` | Papers citing a given paper |
| `GET /paper/{id}/references` | Papers referenced by a given paper |
| `GET /paper/batch` | Batch metadata fetch |
| `GET /paper/{id}/search` | Snippet/full-text search |

### 6.3 LLM Abstraction Layer

**Library:** `libs/chain/` (`ai2i-chain`)

Wraps OpenAI and Google GenAI APIs behind a unified interface. Models are registered by alias:

| Alias | Actual Model |
|-------|-------------|
| `openai:gpt4o-default` | `gpt-4o-2024-11-20` |
| `openai:gpt5mini-medium-reasoning-default` | `gpt-5-mini-2025-08-07` |
| `openai:gpt5mini-minimal-reasoning-default` | `gpt-5-mini-2025-08-07` (lightweight) |
| `google:gemini3flash-medium-reasoning-default` | `gemini-3-flash-preview` |

**Retry Strategies:**
- `RetryWithTenacity` — Single retry for transient network errors.
- `RacingRetryWithTenacity` — Fires multiple parallel retries and returns the fastest response.

**Token Tracking:**
- `MaboolCallbackHandler` hooks into LangChain's `on_llm_end` event.
- Records `input_tokens`, `output_tokens`, `reasoning_tokens` per model per request.

---

## 7. Configuration System

**Config Library:** `libs/config/` (`ai2i-config`)

### Configuration Files

| File | Purpose |
|------|---------|
| `conf/config.toml` | Base application settings |
| `conf/config.extra.fast_mode.toml` | Mode-specific overrides |
| `conf/.env.secret` | API keys (not committed to git) |

### Loading Pipeline

1. Read `config.toml` → base `ConfigSettings`.
2. Glob all `config.extra.*.toml` files and merge them in alphabetical order.
3. Parse `.env.secret` (simple `key=value` format).
4. Inject secrets into `ConfigSettings` fields and `os.environ` (for SDK compatibility).
5. Use Python `ContextVar` for per-request config isolation.

### Key Configuration Sections

| Section | Key Settings |
|---------|-------------|
| `s2_api` | concurrency, timeouts, retries |
| `relevance_judgement` | LLM model, batch size, quota (250) |
| `query_analyzer_agent` | LLM model choice |
| `broad_search_agent` | Max rounds (diligent mode) |
| `dense_agent` | Vector search parameters |
| `snowball_agent` | Citation tracking top-k |
| `llm_abstraction` | Default model, temperature |
| `cache` | Enable/disable, TTL |
| `di` | DI scope timeout |

### Required API Keys

| Key | Required When | Where to Get |
|-----|--------------|-------------|
| `S2_API_KEY` | Server startup (fatal if missing) | semanticscholar.org/product/api |
| `OPENAI_API_KEY` | Every query execution | platform.openai.com/api-keys |
| `GOOGLE_API_KEY` | Gemini agent paths | aistudio.google.com/app/apikey |
| `COHERE_API_KEY` | Reranking step (optional) | dashboard.cohere.com/api-keys |

---

## 8. Database Models & Schema

**File:** `agents/mabool/api/mabool/api/history_routes.py`

**Database:** SQLite at `agents/mabool/api/search_history.db` (auto-created at runtime)

### Schema

**sessions** — Named groups of searches:

```sql
CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    name       TEXT NOT NULL
);
```

**searches** — Individual search records within a session:

```sql
CREATE TABLE searches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    query         TEXT NOT NULL,
    mode          TEXT,
    before_date   TEXT,
    anchor_ids    TEXT,
    s2_session_id TEXT,
    result_count  INTEGER,
    result_json   TEXT
);
```

**bookmarks** — Saved papers:

```sql
CREATE TABLE bookmarks (
    corpus_id  TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    title      TEXT,
    authors    TEXT,
    year       INTEGER,
    venue      TEXT,
    url        TEXT,
    abstract   TEXT,
    tags       TEXT DEFAULT '[]',
    note       TEXT DEFAULT ''
);
```

**paper_notes** — Per-paper annotations:

```sql
CREATE TABLE paper_notes (
    corpus_id  TEXT PRIMARY KEY,
    note       TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Concurrency Safety:** Uses atomic `INSERT ... WHERE EXISTS` patterns to prevent race conditions in concurrent writes.

---

## 9. API Endpoints

### 9.1 Search API (Core)

**`POST /api/2/rounds`**

Request body:

```json
{
  "paper_description": "string (required)",
  "operation_mode": "infer" | "fast" | "diligent",
  "inserted_before": "YYYY-MM-DD",
  "anchor_corpus_ids": ["corpus_id1"],
  "read_results_from_cache": false
}
```

Response body:

```json
{
  "doc_collection": {
    "documents": [
      {
        "corpus_id": "...",
        "title": "...",
        "authors": [...],
        "year": 2024,
        "abstract": "...",
        "venue": "NeurIPS",
        "citation_count": 42,
        "url": "https://...",
        "final_agent_score": 0.95,
        "rerank_score": 0.92
      }
    ]
  },
  "response_text": "Natural language summary...",
  "input_query": "original query",
  "analyzed_query": {...},
  "token_breakdown_by_model": {
    "gpt-4o": {"input": 1000, "output": 500, "reasoning": 200}
  },
  "session_id": "thrd:uuid"
}
```

### 9.2 Session History API (New)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/sessions` | Create a new named session |
| `GET` | `/api/sessions` | List all sessions with search counts |
| `PATCH` | `/api/sessions/{id}` | Rename a session |
| `DELETE` | `/api/sessions/{id}` | Delete session (cascades to searches) |
| `POST` | `/api/sessions/{id}/searches` | Save a search result to a session |
| `GET` | `/api/sessions/{id}/searches` | Retrieve all searches in a session |
| `DELETE` | `/api/sessions/{id}/searches/{sid}` | Delete a single search record |

---

## 10. Frontend Components

**File:** `agents/mabool/api/static/index.html` (~700 lines)

**Technology:** Zero external dependencies — pure HTML, CSS, and vanilla JavaScript in a single file. Served at `http://localhost:8000/ui/`.

### Layout

```
┌──────────────────────────────────────────────┐
│ Sidebar (220px)   │ Main Chat Area           │
├──────────────────────────────────────────────┤
│ • Brand logo      │ Chat Header              │
│ • New Search btn  │   (Session title, badge) │
│ • Sessions list   │ Chat Thread (scrollable) │
│ • Bookmarks       │ Search Bar (sticky)      │
│ • API status      │                          │
└──────────────────────────────────────────────┘
```

### Core JavaScript Functions

| Function | Purpose |
|----------|---------|
| `loadSessions()` | Fetch all sessions from `/api/sessions` |
| `newSession()` | Create a new session via API |
| `openSession(id)` | Switch the active session in view |
| `doSearch()` | Submit a query and track the async task |
| `renderThread(id)` | Display all search history for a session |
| `renderCard(doc)` | Render a single paper result card |
| `syncSearchingUI()` | Update spinners and badge status |

### Key Frontend State

```javascript
const activeTasks = new Map();  // sessionId → { query, pendingTurnId }
let sessions = [];
let activeSessionId = null;
```

### Concurrency Isolation

- Each session has its own entry in `activeTasks`, preventing duplicate submissions.
- Results only refresh the UI if the user is still viewing the session that completed.
- Sidebar shows a spinner next to sessions that are actively searching.

### Paper Card Display

Each result card shows:
- Title (linked to `semanticscholar.org`)
- Year, authors (max 4 shown), venue
- Relevance percentage score
- Abstract (collapsible, 3 lines by default)
- Citation count, influential citation count, fields of study

---

## 11. Key Algorithms & Business Logic

### 11.1 Query Analysis & Field Extraction

LLM extracts the following structured fields from every user query:

```python
ExtractedFields = {
    "content":           str | None,
    "authors":           list[str],
    "venues":            list[str],
    "recency":           "first" | "last" | None,
    "centrality":        "first" | "last" | None,
    "time_range":        ExtractedYearlyTimeRange,
    "broad_or_specific": "broad" | "specific" | "unknown",
    "by_name_or_title":  "name" | "title",
    "relevance_criteria": RelevanceCriteria,
    "domains":           DomainsIdentified,
    "possible_refusal":  PossibleRefusal
}
```

These fields map to one of six `QueryType` values:
`BROAD_BY_DESCRIPTION`, `SPECIFIC_BY_TITLE`, `SPECIFIC_BY_NAME`, `BY_AUTHOR`, `METADATA_ONLY_NO_AUTHOR`, `UNKNOWN`

### 11.2 Specification System

**File:** `agents/mabool/api/mabool/data_model/specifications.py`

Hierarchical specification objects encode the structured meaning of a query for downstream agents:

```python
AuthorSpec(name, affiliation, papers, min_authors)
PaperSpec(title, abstract, venue_names, keywords)
VenueSpec(name, acronym)
ContentSpec(keywords)
TimeRangeSpec(start, end)
FieldOfStudySpec(fields: list[FieldOfStudy])
```

`FieldOfStudy` supports 23+ academic disciplines: Computer Science, Medicine, Physics, Economics, Biology, Chemistry, Law, Mathematics, and more.

### 11.3 Relevance Judgment Pipeline

1. **LLM Batch Scoring** — Each paper's abstract is scored by the LLM. Batch size grows adaptively. Hard quota of 250 papers per query.
2. **Cohere Reranking** — Top candidates are re-ranked using Cohere's `rerank-english-v3.0` model.
3. **Weighted Final Score:**

```
final_agent_score = relevance × 0.6 + recency × 0.2 + centrality × 0.2
```

**Concurrency:** Up to 75 parallel LLM requests during the judgment phase.

### 11.4 Sorting & Ranking

**File:** `agents/mabool/api/mabool/agents/common/sorting.py`

```python
class SortPreferences:
    recent_first: bool
    recent_last: bool
    central_first: bool
    central_last: bool

def sorted_docs_by_preferences(docs, preferences) -> list[Document]:
    # Applies weighted combination of relevance, recency, and centrality
    return sorted_docs
```

Sort preferences are extracted from the user's query intent by the Query Analyzer.

---

## 12. Testing Setup

**Framework:** pytest + pytest-asyncio + pytest-snapshot

**Test Files:**

| File | What It Tests |
|------|--------------|
| `agents/mabool/api/tests/test_specifications.py` | Spec object validation |
| `agents/mabool/api/mabool/infra/operatives/tests/test_interaction.py` | Operative base class behavior |
| `agents/mabool/api/mabool/agents/metadata_only/test_plan.py` | Metadata planner agent logic |
| Various `conftest.py` files | Pytest fixtures per module |

**Dev Commands (from Makefile):**

```bash
make sync-dev      # Install all packages including dev dependencies
make type-check    # Run Pyright type checker
make lint          # Run Ruff linter
make format-check  # Check code formatting
make style         # Run all checks (type + lint + format)
```

---

## 13. Build & Deployment

**Package Manager:** uv (modern Python package/project manager)

**Workspace Configuration (pyproject.toml):**

```toml
[tool.uv.workspace]
members = [
    "agents/mabool/api",
    "libs/*"
]
```

All five internal packages (`ai2i-config`, `ai2i-di`, `ai2i-chain`, `ai2i-dcollection`, `ai2i-common`) are installed as editable local dependencies automatically.

**Full Install:**

```bash
make sync-dev
```

**Start Server:**

```bash
cd agents/mabool/api
bash start.sh
# Or manually:
APP_CONFIG_ENV=dev gunicorn \
  -k uvicorn.workers.UvicornWorker \
  --workers 1 --bind 0.0.0.0:8000 \
  --timeout 0 --reload \
  'mabool.api.app:create_app()'
```

**Deployment Notes:**

- **Single worker** — `--workers 1` is intentional; multi-worker would require a shared `StateManager`.
- **Timeout 0** — Disabled because agent queries can take 30–120 seconds.
- **Reload** — Auto-reload on file change (dev mode).
- **No authentication** — API is fully open; appropriate for local/research use.
- **CORS** — All origins allowed (`"*"`).

---

## 14. Known Limitations & Risks

| Issue | Type | Impact |
|-------|------|--------|
| Vespa dense retrieval unavailable | Architecture | `DenseAgent` path skipped; only available internally at AllenAI |
| Model alias brittleness | External dependency | `gpt5mini-*` names may break if OpenAI changes versioning |
| Single worker | Scalability | Cannot scale horizontally without a shared external `StateManager` |
| SQLite write concurrency | Database | Safe for single-user; high-concurrency load needs PostgreSQL |
| No authentication | Security | API fully open; suitable only for local/private research use |
| Cache shared across users | Design | Same `(query, mode, anchor_ids)` returns cached result to all users |
| `StateManager` is ephemeral | Data | In-memory storage; multi-turn agent state lost on server restart |
| No database backup automation | Operations | Manual backup required for `search_history.db` |

---

## 15. Key Files Summary

| File Path | Purpose | Lines | Status |
|-----------|---------|-------|--------|
| `agents/mabool/api/mabool/api/app.py` | FastAPI entry point | ~110 | Modified (UI mount) |
| `agents/mabool/api/mabool/api/round_v2_routes.py` | Search API routes | ~140 | Original |
| `agents/mabool/api/mabool/api/history_routes.py` | Session history API | ~250 | New |
| `agents/mabool/api/mabool/agents/paper_finder/paper_finder_agent.py` | Main orchestrator | ~300 | Original |
| `agents/mabool/api/mabool/agents/query_analyzer/query_analyzer.py` | Query parsing | ~200 | Original |
| `agents/mabool/api/mabool/infra/operatives/operatives.py` | Agent base class | ~400 | Original |
| `libs/config/ai2i/config/loading.py` | Config + secrets loader | ~100 | Original |
| `libs/di/ai2i/di/` | DI framework | ~300 | Original |
| `libs/chain/ai2i/chain/builders.py` | LLM call builder | ~200 | Original |
| `libs/dcollection/ai2i/dcollection/collection.py` | Document collection | ~200 | Original |
| `agents/mabool/api/static/index.html` | Web UI | ~700 | New |
| `agents/mabool/api/conf/config.toml` | Main config | ~400 | Original |
| `LOCAL_DEPLOYMENT_NOTES.md` | Deployment guide | ~150 | New |
| `TECHNICAL_MANUAL.md` | Tech architecture (Chinese) | ~986 | New |

---

## 16. Data Structures & Models

### Request/Response Models (Pydantic)

| Model | Fields |
|-------|--------|
| `RoundRequest` | paper_description, operation_mode, inserted_before, anchor_corpus_ids |
| `RoundResponse` | doc_collection, response_text, analyzed_query, token_breakdown, session_id |

### Query Analysis Models

| Model | Description |
|-------|-------------|
| `QueryAnalysisSuccess` | Fully parsed query result |
| `QueryAnalysisPartialSuccess` | Partial result with error annotations |
| `QueryAnalysisRefusal` | Query should not be answered |
| `QueryAnalysisFailure` | LLM parsing failed entirely |
| `AnalyzedQuery` | Parsed query with all extracted fields and specifications |

### Document Models

| Model | Description |
|-------|-------------|
| `Document` | Single paper with all metadata + computed scores |
| `DocumentCollection` | Container for documents with lazy field loading |
| `Relevance` | Relevance score with explanation |

### Agent Models

| Model | Description |
|-------|-------------|
| `Operative[INPUT, OUTPUT, STATE]` | Generic base class for all agents |
| `OperativeResponse[T]` | Sealed union: VoidResponse, PartialResponse, CompleteResponse |
| `StateManager` | In-memory TTLCache for multi-turn conversation state |

### Session/History Models

| Model | Description |
|-------|-------------|
| `Session` | id, name, created_at |
| `Search` | session_id, query, mode, result_json, timestamps |
| `Bookmark` | corpus_id, title, authors, year, venue, url, abstract, tags, note |

---

## 17. External Integrations

| Service | SDK/Method | Purpose |
|---------|-----------|---------|
| **Semantic Scholar** | Custom `semanticscholar` client | Paper data, keyword search, citations |
| **OpenAI** | `openai` Python SDK via LangChain | GPT-4o, GPT-5 Mini (query analysis, relevance judgment, explain) |
| **Google GenAI** | `google-generativeai` via LangChain | Gemini models (title matching) |
| **Cohere** | `cohere` Python SDK | Reranking (`rerank-english-v3.0`) |
| **LangChain Core** | Callbacks + chain abstractions | LLM abstraction layer, token tracking |

---

## 18. Summary

**ASTA Paper Finder** is a sophisticated multi-agent LLM system that bridges academic paper discovery with modern AI capabilities. Its key design principles are:

- **Structured query understanding** — Every natural language query is parsed into structured intents before any search happens.
- **Multi-path search** — Different query types route to specialized agents (keyword, citation snowball, dense, author, metadata) that each produce a candidate set.
- **Intelligent ranking** — Candidates are scored by LLM relevance judgment, optionally re-ranked by Cohere, then sorted by a weighted formula.
- **Session management** — A new SQLite-backed API layer tracks search history, bookmarks, and paper notes across sessions.
- **Minimal web UI** — A single-file, zero-dependency HTML frontend provides an interactive chat-like interface.
- **Production-ready infrastructure** — Custom DI framework, TOML config system, async caching, retry strategies, and token tracking are all baked in.

The codebase (~10,000 lines in agents, ~5 internal libraries) is well-architected with clear separation of concerns, extensive Pydantic type models, careful async handling, and a pluggable agent system that makes it straightforward to add new search strategies.

---

*Report generated by Claude Code on 2026-03-21*
