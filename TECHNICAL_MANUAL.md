# ASTA Paper Finder — 技术实现手册

> **Repo 来源**: `allenai/asta-paper-finder`
> **本地部署日期**: 2026-03-20
> **标注说明**: 🔵 = 原版 Repo 代码 | 🟠 = 本次新增/修改

---

## 目录

1. [项目全景](#1-项目全景)
2. [目录结构](#2-目录结构)
3. [配置与密钥系统](#3-配置与密钥系统)
4. [依赖注入框架](#4-依赖注入框架)
5. [LLM 抽象层（chain 库）](#5-llm-抽象层chain-库)
6. [文档集合层（dcollection 库）](#6-文档集合层dcollection-库)
7. [主 API 包：数据模型](#7-主-api-包数据模型)
8. [主 API 包：Agent 架构](#8-主-api-包agent-架构)
9. [主 API 包：基础设施层](#9-主-api-包基础设施层)
10. [主 API 包：服务层与 DI 组装](#10-主-api-包服务层与-di-组装)
11. [主 API 包：API 路由层](#11-主-api-包api-路由层)
12. [完整请求数据流](#12-完整请求数据流)
13. [🟠 新增：历史记录后端](#13--新增历史记录后端)
14. [🟠 新增：前端界面](#14--新增前端界面)
15. [🟠 新增：并发隔离设计](#15--新增并发隔离设计)
16. [启动与部署](#16-启动与部署)
17. [关键依赖一览](#17-关键依赖一览)
18. [已知限制与风险](#18-已知限制与风险)

---

## 1. 项目全景

ASTA Paper Finder 是一个 **AI 驱动的学术论文搜索引擎**。用户用自然语言描述想找的论文，系统通过一系列 LLM Agent + Semantic Scholar API 完成搜索、召回、相关性评分、排序，最终返回有序论文列表。

```
用户自然语言查询
       ↓
  Query Analyzer (LLM)
       ↓
  路由到子 Agent
  ┌────┴───────────────────────────────┐
  │  BroadSearch / SpecificByTitle /  │
  │  ByAuthor / MetadataOnly / Dense  │
  └────┬───────────────────────────────┘
       ↓
  Semantic Scholar API 召回候选论文
       ↓
  Relevance Judgment (LLM 批量评分)
       ↓
  Cohere Rerank + 排序
       ↓
  返回 DocumentCollection + response_text
```

**技术栈总览**:

| 层次 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn + Gunicorn |
| Agent 框架 | 自研 `Operative` 模式 |
| LLM 调用 | LangChain-Core + OpenAI SDK + Google GenAI SDK |
| 论文数据 | Semantic Scholar REST API (`semanticscholar` 库) |
| 重排序 | Cohere Rerank API |
| 配置 | TOML + `.env.secret` + Python 类型化 schema |
| 依赖注入 | 自研 `ai2i-di` 框架（async-aware，带 scope 管理）|
| 缓存 | `aiocache` (文件型 + 内存型) |
| 历史记录 | 🟠 SQLite（新增） |
| 前端 | 🟠 单文件 HTML/CSS/JS（新增） |

---

## 2. 目录结构

```
asta/
├── agents/
│   └── mabool/
│       └── api/                         🔵 主 API 包
│           ├── conf/
│           │   ├── config.toml          🔵 主配置文件
│           │   ├── config.extra.fast_mode.toml  🔵 fast 模式覆盖
│           │   └── .env.secret          🟠 密钥文件（模板由我们创建）
│           ├── mabool/
│           │   ├── api/
│           │   │   ├── app.py           🔵+🟠 FastAPI 应用入口（我们加了 static mount）
│           │   │   ├── round_v2_routes.py  🔵 搜索 API 路由
│           │   │   ├── route_utils.py   🔵 响应工具
│           │   │   └── history_routes.py   🟠 历史记录路由（全新）
│           │   ├── agents/              🔵 所有搜索 Agent
│           │   ├── dal/                 🔵 数据访问层 DI 模块
│           │   ├── data_model/          🔵 数据模型
│           │   ├── external_api/        🔵 Cohere 等外部 API
│           │   ├── infra/               🔵 基础设施（Operative、StateManager）
│           │   ├── services/            🔵 服务层 DI 组装
│           │   └── utils/               🔵 工具函数
│           ├── static/
│           │   └── index.html           🟠 前端界面（全新）
│           ├── start.sh                 🟠 启动脚本（新增）
│           └── search_history.db        🟠 SQLite 数据库（运行时生成）
├── libs/
│   ├── config/                          🔵 配置加载库
│   ├── di/                              🔵 依赖注入框架
│   ├── chain/                           🔵 LLM 抽象层
│   ├── dcollection/                     🔵 文档集合 + S2 API 集成
│   └── common/                          🔵 公共工具
├── pyproject.toml                       🔵 uv 工作区定义
├── uv.lock                              🔵 依赖锁定文件
├── LOCAL_DEPLOYMENT_NOTES.md            🟠 部署说明（新增）
└── TECHNICAL_MANUAL.md                  🟠 本文档（新增）
```

---

## 3. 配置与密钥系统

🔵 **库**: `libs/config/`，包名 `ai2i-config`

### 3.1 加载流程

```
load_conf(project_root / "conf")
    ↓
1. 读取 config.toml          → ConfigSettings (基础配置)
2. 读取 config.extra.*.toml  → 按 glob 合并所有 extra 文件
3. 读取 .env.secret          → 解析为 key=value，注入 ConfigSettings
4. update_environment_with_secrets()  → 同步写入 os.environ（兼容 SDK）
```

### 3.2 配置 Schema

```python
# mabool/data_model/config.py
cfg_schema = AppConfigSchema()
```

所有配置键都通过 `cfg_schema.xxx` 引用，类型安全，编译期可检查。主要分组：

| 分组 | 含义 |
|------|------|
| `s2_api` | S2 并发数、超时、重试 |
| `relevance_judgement` | LLM 评分模型、批大小、quota |
| `query_analyzer_agent` | 查询分析使用的 LLM |
| `broad_search_agent` | 宽泛搜索最大轮数 |
| `dense_agent` | 向量搜索参数 |
| `snowball_agent` | 引用追踪 top-k |
| `llm_abstraction` | GPT-4o 默认模型名、temperature |
| `cache` | 缓存开关、TTL |
| `di` | DI scope 超时 |

### 3.3 上下文变量

配置通过 Python `ContextVar` 传递，实现请求级别隔离：

```python
# libs/config/config.py
_app_config_context: ContextVar[ConfigSettings]

with application_config_ctx(AppConfig(config=config_settings, ...)):
    # 在此上下文中调用 config_value(cfg_schema.xxx) 均安全
```

### 3.4 密钥文件

🟠 **由本次部署创建的模板**：`agents/mabool/api/conf/.env.secret`

```ini
S2_API_KEY=...        # 启动必须，DI singleton scope 在启动时校验
OPENAI_API_KEY=...    # 查询时必须（大多数 Agent 使用）
COHERE_API_KEY=...    # 可选，缺少时跳过 rerank
GOOGLE_API_KEY=...    # 查询时必须（Gemini Agent）
```

---

## 4. 依赖注入框架

🔵 **库**: `libs/di/`，包名 `ai2i-di`

这是 AllenAI 自研的异步感知 DI 框架，不依赖任何第三方 DI 库。

### 4.1 核心概念

```
Module → 包含一组 Provider
Provider → 有 scope 的工厂函数
Scope → "singleton"（应用级）或 "round_scope"（请求级）
ApplicationContext → 管理所有 Module 和 Scope
```

### 4.2 注册方式

```python
module = create_module("MyModule")

@module.provides(scope="singleton")
async def my_singleton_service(dep: SomeType = DI.requires(...)) -> MyService:
    return MyService(dep)

@module.provides()   # 默认 transient
async def my_transient(...) -> MyTransient:
    ...

@module.global_init()
async def _init(service: MyService = DI.requires(...)):
    # 应用启动时运行
```

### 4.3 关键 DI 绑定（影响启动）

```python
# mabool/utils/dc_deps.py，第 12 行
s2_api_key: str = DI.config(cfg_schema.s2_api_key)
```

这是 **启动时 fatal 的绑定**——`scope="singleton"` 在 Uvicorn worker 启动时解析，没有默认值，`S2_API_KEY` 缺失直接崩溃。

### 4.4 Scope 生命周期

```
应用启动
  └─ singleton scope 打开 (全应用共享)
      ├─ DocumentCollectionFactory 创建
      ├─ CohereRerankScorer 创建
      └─ 其他单例...

每次 POST /api/2/rounds
  └─ round_scope 打开（请求级）
      ├─ RoundId 生成
      ├─ TurnId 生成
      └─ 请求完成后 round_scope 关闭
```

---

## 5. LLM 抽象层（chain 库）

🔵 **库**: `libs/chain/`，包名 `ai2i-chain`

### 5.1 模型注册表

`LLMModel` 类统一管理所有模型别名：

| 别名 | 实际模型 | 用途 |
|------|----------|------|
| `openai:gpt4o-default` | `gpt-4o-2024-11-20` | 元数据规划、宽泛搜索 |
| `openai:gpt5mini-medium-reasoning-default` | `gpt-5-mini-2025-08-07`（推理版）| 查询分析、关键词搜索 |
| `openai:gpt5mini-minimal-reasoning-default` | 同上轻量版 | 相关性评分 |
| `google:gemini3flash-medium-reasoning-default` | `gemini-3-flash-preview` | 特定论文查找、LLM 建议 |

> **注意**：这些是 AllenAI 内部命名，实际映射在 `libs/chain/ai2i/chain/models.py` 中定义。

### 5.2 调用链

```python
# 定义一个 LLM 调用
chain = define_chat_llm_call(
    messages=[system_message("..."), user_message("...")],
    model=LLMModel("openai:gpt4o-default"),
    endpoint=LLMEndpoint(timeouts=Timeouts(total=60)),
)

# 执行
result = await chain.ainvoke({"variable": value})
```

### 5.3 重试策略

```python
RetryWithTenacity         # 单次重试（网络抖动）
RacingRetryWithTenacity   # 并发多次取最快结果
```

### 5.4 回调追踪

```python
# round_v2_routes.py
class MaboolCallbackHandler(AsyncCallbackHandler):
    # 监听 on_llm_end 事件，记录 token 使用
    # 按模型分类：input_tokens, output_tokens, reasoning_tokens
```

---

## 6. 文档集合层（dcollection 库）

🔵 **库**: `libs/dcollection/`，包名 `ai2i-dcollection`

### 6.1 核心数据结构

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
    snippets: list[Snippet] | None      # 全文片段（S2 提供）
    relevance_judgement: Relevance | None
    rerank_score: float | None
    final_agent_score: float | None
```

```python
class DocumentCollection:
    documents: list[Document]
    # 惰性加载，支持字段级别按需拉取
    BASIC_FIELDS     # corpus_id, title, year, authors
    UI_REQUIRED_FIELDS  # + abstract, venue, citations, url
    FULL_FIELDS      # + snippets, references, full metadata
```

### 6.2 S2 API 集成

```python
class DocumentCollectionFactory:
    def __init__(self, s2_api_key: str | None, ...):
        # s2_api_key=None → 无鉴权（100 req/5min 限速）
        # s2_api_key=有值 → 鉴权（更高配额）
```

`S2Fetcher` 封装了：
- `GET /paper/search` — 关键词搜索
- `GET /paper/{corpus_id}/citations` — 引用追踪
- `GET /paper/{corpus_id}/references` — 参考文献
- `GET /paper/batch` — 批量获取元数据
- `GET /paper/{corpus_id}/search` — snippet 检索

### 6.3 Vespa 稠密检索（⚠️ 不可用）

`libs/dcollection/ai2i/dcollection/external_api/dense/vespa.py` 集成了 AllenAI 内部 Vespa 向量检索集群。该服务为私有，无公开端点，本地部署时 dense search 路径不可用。

---

## 7. 主 API 包：数据模型

### 7.1 请求与响应

```python
# mabool/data_model/rounds.py
class RoundRequest(BaseModel):
    paper_description: str                   # 必填：自然语言查询
    anchor_corpus_ids: list[CorpusId] = []  # 可选：锚点论文 ID
    operation_mode: AgentOperationMode = "infer"
    inserted_before: str | None = None      # 格式：YYYY-MM-DD / YYYY-MM / YYYY
    read_results_from_cache: bool = False

AgentOperationMode = Literal["infer", "fast", "diligent"]
```

响应为 JSON `dict`，主要字段：

```python
{
    "doc_collection": {
        "documents": [...]   # 每条包含 title/url/year/authors/abstract/venue
                             # /citation_count/final_agent_score/rerank_score
    },
    "response_text": str,        # LLM 生成的摘要文字
    "input_query": str,
    "analyzed_query": {...},
    "metrics": {...},
    "error": None | str,
    "token_breakdown_by_model": {"gpt-4o": {"input": N, "output": N}},
    "session_id": "thrd:uuid"
}
```

### 7.2 查询分析结果

```python
# mabool/data_model/agent.py
QueryAnalysisResult = Union[
    QueryAnalysisSuccess,        # 完整解析
    QueryAnalysisPartialSuccess, # 部分解析 + 错误
    QueryAnalysisRefusal,        # 拒绝（非论文查找）
    QueryAnalysisFailure,        # 解析失败
]

class QueryAnalysisSuccess:
    query_type: QueryType        # broad / specific / metadata_only 等
    extracted_fields: ExtractedFields  # 结构化字段
    specifications: Specifications     # 供 Agent 使用的规格对象
```

### 7.3 规格系统（Specifications）

Query 解析后转为结构化规格，驱动 Agent 路由决策：

```python
Specifications
├── AuthorSpec    (name, affiliation, papers, min_authors)
├── PaperSpec     (title, abstract, venue_names, keywords)
├── VenueSpec     (name, acronym)
├── ContentSpec   (keywords)
├── TimeRangeSpec (start, end)
└── FieldOfStudySpec (fields: list[FieldOfStudy])
```

`FieldOfStudy` 覆盖 25+ 学科，包括 CS、Medicine、Physics、Economics 等。

---

## 8. 主 API 包：Agent 架构

### 8.1 基础模式：Operative

🔵 所有 Agent 继承自 `Operative[INPUT, OUTPUT, STATE]`（`mabool/infra/operatives/`）：

```python
class Operative[INPUT, OUTPUT, STATE]:
    async def handle_operation(self, input: INPUT) -> OperativeResponse[OUTPUT]:
        # 核心逻辑

    def init_operative(self, cls, ...) -> SubOperative:
        # 创建子 Agent

OperativeResponse[T] = VoidResponse | PartialResponse[T] | CompleteResponse[T]
```

### 8.2 14 个 Agent 及其职责

```
agents/
├── paper_finder/           🎯 主 Agent（编排器）
│   ├── PaperFinderAgent    路由到子 Agent，汇总结果
│   └── paper_finder_agent.py  run_agent() 入口函数
│
├── query_analyzer/         🧠 查询理解
│   └── decompose_and_analyze_query_restricted()
│       LLM 解析自然语言 → QueryAnalysisResult
│
├── broad_search_by_keyword/ 🔍 关键词搜索
│   └── BroadSearchByKeywordAgent
│       LLM 生成搜索词 → S2 API 搜索
│
├── complex_search/          🔍🔍 迭代深度搜索
│   ├── BroadSearchAgent     多轮 LLM 建议 + S2（最多3轮）
│   └── FastBroadSearchAgent 快速模式：Dense + Snowball + S2
│
├── dense/                   🔍 向量语义搜索
│   └── DenseAgent           ⚠️ 需要 Vespa（内部服务，本地不可用）
│
├── specific_paper_by_title/ 📄 精确标题匹配
│   └── SpecificPaperByTitleAgent
│       Gemini LLM 匹配标题 → corpus_id
│
├── specific_paper_by_name/  📄 引用名匹配
│   └── SpecificPaperByNameAgent
│
├── search_by_authors/       👤 按作者搜索
│   └── SearchByAuthorsAgent
│       LLM 消歧作者名 → S2 作者 API
│
├── metadata_only/           📋 纯元数据过滤
│   ├── MetadataOnlySearchAgent
│   └── MetadataPlannerAgent   LLM 规划元数据操作
│
├── llm_suggestion/          💡 LLM 直接建议论文
│   └── get_llm_suggested_papers()
│       推理模型直接猜测论文 corpus_id
│
├── snowball/                ❄️ 引用扩展
│   ├── forward: 引用目标论文的文章
│   └── backward: 目标论文引用的文章
│
├── by_citing_papers/        🔗 反向引用
│
├── query_refusal/           ❌ 拒绝处理
│   拒绝类型: similar_to, web_access, not_paper_finding,
│             affiliation, author_id
│
└── common/                  🛠️ 共享逻辑
    ├── computed_fields/     相关性计算、排名、snippet 处理
    ├── sorting.py           SortPreferences + sorted_docs_by_preferences()
    ├── explain.py           生成 response_text 摘要
    └── common.py            时间过滤、作者过滤等工具函数
```

### 8.3 路由逻辑

`PaperFinderAgent` 根据 `QueryAnalysisResult` 和 `operation_mode` 路由：

```
QueryType
├── BROAD_BY_DESCRIPTION
│   ├── fast     → FastBroadSearchAgent（Dense + Snowball + S2）
│   └── diligent → BroadSearchAgent（多轮迭代）
├── SPECIFIC_BY_TITLE   → SpecificPaperByTitleAgent
├── SPECIFIC_BY_NAME    → SpecificPaperByNameAgent
├── BY_AUTHOR           → SearchByAuthorsAgent
├── METADATA_ONLY_*     → MetadataOnlySearchAgent
└── REFUSAL             → QueryRefusalAgent（返回拒绝消息）
```

### 8.4 相关性评分流程

每个 Agent 召回候选后，都经过相同的评分流程：

```
候选论文（可能100-1000篇）
        ↓
Relevance Judgment (LLM 批量评分)
  - 模型: gpt5mini-minimal-reasoning-default
  - batch_size: 动态增长（growth_factor=2）
  - quota: 250（最多评分这么多篇）
  - 并发: 75 个请求并发
        ↓
Cohere Rerank
  - 模型: rerank-english-v3.0
  - 对 LLM 高分论文做二次精排
        ↓
最终排序（recency + centrality + relevance 加权）
        ↓
返回 Top-K（response_text_top_k = 10）
```

---

## 9. 主 API 包：基础设施层

### 9.1 StateManager（状态管理）

```python
# mabool/infra/operatives/state_manager.py
class StateManager[STATE]:
    _state_dict: TTLCache[tuple[StateManagerId, SessionId], Any]
    # TTL = 24 小时，内存存储，重启丢失

    async def save(session_id, state)
    async def load(session_id) → STATE | None
    async def pop(session_id) → STATE | None
```

Agent 用它在多轮对话间保持状态（如已找到的论文集合）。

### 9.2 InteractionManager（交互管理）

```python
# mabool/infra/operatives/interaction_manager.py
class InteractionManager:
    _sessions: TTLCache[SessionId, SessionData]
    # 使用 anyio 内存 channel 实现 async 问答

    async def send_inquiry(session_id, question)  # Agent 向用户提问
    async def receive_inquiry(session_id) → Question
    async def send_reply(session_id, answer)       # 用户回答
    async def receive_reply(session_id) → Answer
```

当前版本 API 未暴露交互接口，此机制为未来多轮对话预留。

### 9.3 PrioritySemaphore（并发控制）

```python
# mabool/services/prioritized_task.py
round_semaphore = PrioritySemaphore(concurrency=3)

# 所有搜索请求共用此信号量
# 最多 3 个请求同时执行（包含跨会话的请求）
# DEFAULT_PRIORITY 所有请求优先级相同
```

---

## 10. 主 API 包：服务层与 DI 组装

```python
# mabool/services/services_deps.py
services_module = create_module(
    "Services",
    extends=[
        dal_deps.dal_module,              # DocumentCollectionFactory（S2 配置）
        external_api_deps.external_api_module,  # CohereRerankScorer
        context_deps.context_module,      # RoundId, TurnId, RequestAndBody
        dc_deps.dc_module,                # S2 API key 校验 ← 启动时 fatal
        tracing_deps.tracing_module,      # 追踪/度量
    ],
)

@services_module.global_init()
async def _setup_globals(use_cache: bool = DI.config(...)):
    if use_cache:
        set_llm_cache(InMemoryCache())   # LangChain LLM 缓存
```

---

## 11. 主 API 包：API 路由层

### 11.1 🔵 搜索路由（原版）

**文件**: `mabool/api/round_v2_routes.py`

```
POST /api/2/rounds
    ↓
start_round(request: RoundRequest)
    ↓
async with round_semaphore.priority_context(DEFAULT_PRIORITY):
    ↓
run_round_with_cache(paper_description, anchor_corpus_ids, operation_mode, ...)
    ├── @cached(FileBasedCache)  如果 read_results_from_cache=True
    ├── conversation_thread_id = generate_conversation_thread_id()
    ├── input = PaperFinderInput(query, anchor_corpus_ids, ...)
    └── response = await run_agent(input, conversation_thread_id)
                                      ↓
                          PaperFinderAgent.handle_operation(input)
```

**缓存键**：`(paper_description, anchor_corpus_ids, operation_mode)` 的哈希，**不含** session 信息 → 相同参数跨会话共享缓存。

### 11.2 🔵+🟠 应用入口（修改）

**文件**: `mabool/api/app.py`

🔵 原版功能：
- CORS `allow_origins=["*"]`
- `GET /` → 重定向 `/docs`
- `GET /health` → 204
- Error handlers

🟠 我们新增的两行：
```python
from mabool.api.history_routes import router as history_routes  # 新增
# ...
app.include_router(history_routes)                              # 新增
app.mount("/ui", StaticFiles(directory=project_root() / "static", html=True), name="ui")  # 新增
```

---

## 12. 完整请求数据流

```
浏览器 POST /api/2/rounds
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  FastAPI  round_v2_routes.py                                   │
  │  start_round()                                                  │
  │    └─ PrioritySemaphore.acquire()  (最多3个并发)               │
  │         └─ run_round_with_cache()                              │
  │              └─ run_agent(PaperFinderInput, thread_id)         │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  PaperFinderAgent.handle_operation()                           │
  │    1. decompose_and_analyze_query_restricted()                  │
  │       └─ LLM (gpt5mini-medium-reasoning)                       │
  │          → QueryAnalysisSuccess {                               │
  │              query_type, extracted_fields, specifications       │
  │            }                                                    │
  │                                                                 │
  │    2. 根据 query_type + operation_mode 选择子 Agent             │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼  (以 BROAD fast 路径为例)
  ┌─────────────────────────────────────────────────────────────────┐
  │  FastBroadSearchAgent                                          │
  │    ├─ DenseAgent (Vespa，本地不可用，跳过)                      │
  │    ├─ SnowballAgent                                            │
  │    │    ├─ anchor_corpus_ids 的 forward citations              │
  │    │    └─ anchor_corpus_ids 的 backward citations             │
  │    └─ BroadSearchByKeywordAgent                               │
  │         ├─ LLM 生成 N 个搜索词                                  │
  │         └─ S2 API keyword search (并发 10)                    │
  │                                                                 │
  │    → 合并去重，得到候选 DocumentCollection (可能数百篇)          │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Relevance Judgment                                            │
  │    ├─ LLM (gpt5mini-minimal-reasoning) 批量评分 abstract       │
  │    ├─ 并发 75 请求，quota=250                                   │
  │    └─ 每篇得到 relevance_judgement.score ∈ [0, 1]             │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Cohere Rerank                                                 │
  │    └─ rerank-english-v3.0 对高分候选二次精排                    │
  │       → 得到 rerank_score                                      │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Sorting（sorted_docs_by_preferences）                         │
  │    加权组合：relevance + recency + centrality                   │
  │    → final_agent_score                                         │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Explain（generate response_text）                             │
  │    Top-K 论文 → LLM 生成自然语言摘要                            │
  │    response_text_top_k = 10                                    │
  └─────────────────────────────────────────────────────────────────┘
         │
         ▼
  JSON Response → 浏览器
  {
    doc_collection: { documents: [...] },
    response_text: "...",
    token_breakdown_by_model: {...},
    session_id: "thrd:uuid",
    ...
  }
```

---

## 13. 🟠 新增：历史记录后端

**文件**: `mabool/api/history_routes.py`
**数据库**: `search_history.db`（SQLite，项目根目录，运行时自动创建）

### 13.1 数据库 Schema

```sql
CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    name       TEXT    NOT NULL           -- 默认 "New search"，第一条查询后自动更名
);

CREATE TABLE searches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    query         TEXT    NOT NULL,       -- 原始查询文本
    mode          TEXT,                  -- fast / infer / diligent
    before_date   TEXT,                  -- inserted_before 参数
    anchor_ids    TEXT,                  -- JSON 数组字符串
    s2_session_id TEXT,                  -- S2 返回的 session_id
    result_count  INTEGER,               -- 找到的论文数
    result_json   TEXT                   -- 完整 API 响应（JSON 字符串）
);
```

`ON DELETE CASCADE`：删除 session 时自动删除其所有 searches。

### 13.2 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/sessions` | 创建新会话 |
| `GET` | `/api/sessions` | 列出所有会话（含 search_count） |
| `PATCH` | `/api/sessions/{id}` | 重命名会话 |
| `DELETE` | `/api/sessions/{id}` | 删除会话（级联删除所有搜索） |
| `POST` | `/api/sessions/{id}/searches` | 保存一条搜索记录 |
| `GET` | `/api/sessions/{id}/searches` | 获取会话内所有搜索（含完整结果） |
| `DELETE` | `/api/sessions/{id}/searches/{sid}` | 删除单条搜索 |

### 13.3 并发安全设计

保存搜索时使用原子 INSERT：

```sql
-- 避免 TOCTOU（先检查 session 存在再插入的竞争条件）
INSERT INTO searches (session_id, ...)
SELECT ?, ...
WHERE EXISTS (SELECT 1 FROM sessions WHERE id = ?)
```

如果 session 不存在，`rowcount == 0`，返回 404，不产生孤行。

---

## 14. 🟠 新增：前端界面

**文件**: `static/index.html`（单文件，约 700 行，零外部依赖）

### 14.1 布局结构

```
┌─────────────────────────────────────────────────────────┐
│  sidebar (256px)     │  main                            │
│  ┌───────────────┐   │  ┌────────────────────────────┐  │
│  │ New search    │   │  │  chat-header               │  │
│  ├───────────────┤   │  │  (session title, searching │  │
│  │ session 1     │   │  │   badge)                   │  │
│  │ session 2 🔄  │   │  ├────────────────────────────┤  │
│  │ session 3     │   │  │  chat-thread (scroll)      │  │
│  │ ...           │   │  │  ┌──────────────────────┐  │  │
│  ├───────────────┤   │  │  │  turn: query bubble  │  │  │
│  │ ● API online  │   │  │  │  turn: results cards │  │  │
│  └───────────────┘   │  │  │  ...                 │  │  │
│                      │  │  │  [typing indicator]  │  │  │
│                      │  └──┴──────────────────────┘──┘  │
│                      │  search-bar-wrap (sticky bottom) │
│                      │  ┌────────────────────────────┐  │
│                      │  │ options: mode/date/anchors │  │
│                      │  │ [query input] [send btn]   │  │
│                      │  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 14.2 关键 JS 模块

```javascript
// 全局状态
const activeTasks = new Map();   // sessionId → {query, pendingTurnId}
let sessions = [];               // 从 /api/sessions 拉取
let activeSessionId = null;      // 当前显示的会话

// 核心函数
loadSessions()         → GET /api/sessions，更新侧边栏
newSession()           → POST /api/sessions，打开新会话
openSession(id)        → 切换会话，调用 renderThread(id)
doSearch()             → 并发搜索核心（见下节）
renderThread(id)       → GET /api/sessions/{id}/searches，渲染聊天记录
renderTurn(search, i)  → 渲染单条搜索（query bubble + 论文卡片列表）
renderCard(doc, key)   → 渲染单张论文卡片
syncSearchingUI()      → 根据 activeTasks 更新发送按钮和 header badge
```

### 14.3 搜索表单字段

| 字段 | 对应 API 参数 | 默认值 |
|------|---------------|--------|
| query textarea | `paper_description` | （必填）|
| Mode select | `operation_mode` | `"fast"` |
| Before date input | `inserted_before` | （可选，YYYY-MM-DD）|
| Anchor IDs input | `anchor_corpus_ids` | （可选，逗号分隔）|

### 14.4 论文卡片展示字段

```
标题（链接到 semanticscholar.org）
年份 · 作者（最多4位）+N · 期刊/会议（斜体）   [相关性%]
摘要（默认折叠3行，Show more 展开）
pills: ↗ N citations | ⭐ N influential | 研究领域...
```

---

## 15. 🟠 新增：并发隔离设计

### 15.1 问题分析

原版 Repo 在前端是单页面设计，没有多会话并发概念。我们加入 session 后，需要正确处理：

1. **会话 A 搜索中，切到会话 B 继续搜索** — 应互相独立
2. **会话 A 结果到达时，用户正在看会话 B** — 不应污染 B 的 UI
3. **同一会话防止重复提交**
4. **删除正在搜索的会话**

### 15.2 解决方案：per-session task Map

```javascript
// 替代全局 isSearching 布尔值
const activeTasks = new Map();
// key = sessionId, value = { query: string, pendingTurnId: string }
```

**搜索启动时**（`doSearch()` 关键代码逻辑）：

```javascript
// 1. 快照当前 session（整个 async 过程中此值不变）
const searchSessionId = activeSessionId;

// 2. 注册任务（在任何 await 之前）
activeTasks.set(searchSessionId, { query, pendingTurnId });

// 3. 立即更新 UI（侧边栏 spinner + header badge）
renderSessionList();
syncSearchingUI();
```

**搜索完成时**（`finally` 块）：

```javascript
activeTasks.delete(searchSessionId);    // 清理任务状态
await loadSessions();                   // 更新侧边栏计数 + 关闭 spinner

// 只有用户仍在看这个 session 时才刷新 thread
if (activeSessionId === searchSessionId) {
    await renderThread(searchSessionId);
    syncSearchingUI();
}
// 如果用户已切换到别的 session → 什么都不做，不打扰当前视图
```

### 15.3 行为矩阵

| 场景 | 行为 |
|------|------|
| 会话 A 搜索中，点击会话 B 的发送 | ✅ B 独立搜索，A 继续 |
| 会话 A 完成，用户在看 B | ✅ A 的结果安静写入 DB；A 的侧边栏计数更新；B 不受影响 |
| 会话 A 搜索中，再次点击发送 | 🚫 `activeTasks.has(A)` 阻止重复提交 |
| 切换到正在搜索的会话 | ✅ `renderThread()` 检测到 `activeTasks.has(id)`，在 thread 末尾显示 typing indicator |
| 删除正在搜索的会话 | 🚫 弹出提示，拒绝删除 |
| 侧边栏显示 | 🟡 搜索中的会话显示橙色旋转动画 + "searching…" |

### 15.4 后端并发限制（原版）

原版 `PrioritySemaphore(concurrency=3)` 是全局的——所有会话共享 3 个槽位。这不是 bug，是 AllenAI 有意的 API 配额保护。并发的多个会话会各自独立占用槽位，超出时排队等待。

---

## 16. 启动与部署

### 16.1 启动命令

```bash
# 方式 1：脚本（推荐）
<repo-root>/agents/mabool/api/start.sh

# 方式 2：手动
cd <repo-root>/agents/mabool/api
APP_CONFIG_ENV=dev <repo-root>/.venv/bin/gunicorn \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 --timeout 0 --bind 0.0.0.0:8000 \
    --enable-stdio-inheritance --access-logfile - --reload \
    'mabool.api.app:create_app()'
```

### 16.2 访问地址

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/ui/` | 前端界面（🟠 新增）|
| `http://localhost:8000/docs` | Swagger API 文档（原版）|
| `http://localhost:8000/health` | 健康检查（原版）|
| `http://localhost:8000/api/2/rounds` | 搜索 API（原版）|
| `http://localhost:8000/api/sessions` | 会话历史 API（🟠 新增）|

### 16.3 环境变量

| 变量 | 说明 | 必须 |
|------|------|------|
| `APP_CONFIG_ENV` | 配置环境（`dev`/`test`）| 是，运行时设置 |
| `S2_API_KEY` | Semantic Scholar | 是，写入 `.env.secret` |
| `OPENAI_API_KEY` | OpenAI | 查询时必须 |
| `GOOGLE_API_KEY` | Google Gemini | 查询时必须 |
| `COHERE_API_KEY` | Cohere Rerank | 可选 |

---

## 17. 关键依赖一览

| 包 | 版本 | 用途 | 原版/新增 |
|----|------|------|----------|
| `fastapi` | ~0.115.6 | Web 框架 | 🔵 |
| `uvicorn` + `gunicorn` | ~0.34 | ASGI 服务器 | 🔵 |
| `pydantic` | 2.10.4 | 数据模型 | 🔵 |
| `langchain-core` | ~1.0 | LLM 回调/链 | 🔵 |
| `langchain-openai` | ~1.0 | OpenAI 集成 | 🔵 |
| `google-genai` | >=1.10.0 | Gemini 集成 | 🔵 |
| `cohere` | ~5.13.4 | Rerank API | 🔵 |
| `semanticscholar` | git (allenai) | S2 API 客户端 | 🔵 |
| `mabwiser` | git (allenai) | 多臂老虎机采样 | 🔵 |
| `aiocache` | ~0.12.3 | 异步缓存 | 🔵 |
| `tenacity` | ~9.1.2 | 重试逻辑 | 🔵 |
| `anyio` | - | 异步原语 | 🔵 |
| `cachetools` | - | TTLCache（StateManager）| 🔵 |
| `deepmerge` | 2.0 | 配置合并 | 🔵 |
| `python-dotenv` | ~1.0.1 | `.env.secret` 加载 | 🔵 |
| `pandas` | ~2.2.3 | 数据处理 | 🔵 |
| `aiofiles` | 25.1.0 | 静态文件服务 | 🟠（我们安装）|
| `sqlite3` | 内置 | 历史记录存储 | 🟠 |

---

## 18. 已知限制与风险

| 问题 | 类型 | 说明 |
|------|------|------|
| Vespa 稠密检索不可用 | 🔵 架构限制 | AllenAI 内部服务，无公开端点；Dense Agent 路径静默跳过 |
| 无认证层 | 🔵 设计决策 | API 完全开放，CORS `"*"`；本地研究用途可接受 |
| 单 Worker | 🔵+🟠 部署限制 | gunicorn `--workers 1`；多 worker 需共享 StateManager |
| SQLite 并发写 | 🟠 已缓解 | 本地单用户场景足够；高并发需换 PostgreSQL |
| 模型别名可能失效 | 🔵 外部依赖 | `gpt5mini-*` / `gemini3flash-*` 是 AllenAI 内部命名映射，模型版本更新后可能需更新 `libs/chain/models.py` |
| 缓存跨会话共享 | 🔵 设计决策 | 相同 `(query, mode, anchor_ids)` 参数在所有会话间共用缓存结果 |
| StateManager 重启丢失 | 🔵 内存存储 | 服务器重启后 Agent 内部状态清空；历史记录（SQLite）不受影响 |
| `search_history.db` 无备份 | 🟠 运维 | 定期备份此文件即可保留所有历史记录 |
