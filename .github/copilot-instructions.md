# LangGraph Demo - AI Agent Instructions

## Project Overview

This is a **LangGraph-based multi-agent system** for financial analysis with intelligent routing, automatic fault tolerance, and user profile management. The system classifies intents, applies tier-based rules, routes to specialized agents, handles missing user data with followup prompts, and automatically retries failed nodes using MongoDB checkpointing.

## Architecture

### Core Components

- **`app.py`**: Main orchestrator with StateGraph and MongoDB checkpointing. Flow:
  - Entry: `intent_classifier` → `rule_builder_node` → `finance_agent_node`
  - If finance agent has `missing_info`: → `followup_agent_node` → END
  - On error: Dual-layer retry (internal graph + external wrapper)
  - Uses `with_error_handling()` wrapper for all nodes except `error_node`
  - External retry via `run_with_auto_retry()` with configurable strategies
- **`nodes/`**: Mix of LangChain agents and data processing functions:
  - **Agent nodes**: `intent_classifier`, `finance_agent_node`, `followup_agent_node` (use `create_agent()`)
  - **Processing nodes**: `rule_builder_node`, `tier_detector`, `error_node` (direct state manipulation)
  - Structure: `__init__.py` exports main function, `agent.py` contains logic, `prompt.py` has `SYSTEM_PROMPT`
- **`tools/`**: LangChain `@tool` wrappers for agent consumption:
  - `select_intent.py`: Intent classification
  - `fetch_user_info.py`, `fetch_credit_info.py`: User data retrieval
  - `retrieve_knowledge.py`: RAG retrieval tool
  - `web_search_tool.py`: DuckDuckGo search
- **`prompts/`**: Text files loaded via `utils/prompt_loader.py`:
  - `finance_agent_prompt.txt`: Base prompt with `{{response_style}}` placeholder
  - `rule_builder_node` injects tier-specific rules into placeholders
- **`services/`**: Shared initialization:
  - `opanai_service.py`: ChatOpenAI model (gpt-4o-mini, temp=0.1)
  - `qdrant_service.py`: QdrantClient for vector DB
  - `checkpoint_service.py`: MongoDB for conversation persistence
- **`retriever/`**: `MMRRetriever` class for user-filtered vector search with diversity
- **`repositories/`**: Data access layer for Qdrant queries
- **`utils/`**:
  - `intents.py`: Intent constants
  - `prompt_loader.py`: File-based prompt loading
  - `retry_handler.py`: **Automatic retry with checkpointing** (4 strategies: exponential, linear, fixed, fibonacci)
  - `agent_message_builder.py`: Message formatting utilities

### State Management

State flows through graph as TypedDict in `states.py`:

```python
class State(TypedDict):
    query: str                          # User's question
    is_profile_complete: bool           # Profile status flag
    intent: str                         # "finance" (hardcoded in current impl)
    tier: str                           # "PAID" or "FREE" (affects response style)
    user_id: str                        # UUID for Qdrant filtering
    final_answer: str                   # User-facing response
    rule_prompt: str                    # Tier-customized prompt from rule_builder
    finance_agent: FinanceAgentState    # {answer: Dict, missing_info: list}
    error: Optional[ErrorState]         # Error tracking for retries
```

**CRITICAL**: LangGraph throws `InvalidUpdateError` if multiple nodes update same state key concurrently. Use sequential routing with conditional edges. Each node should update unique state keys or return full state dict.

### Error Handling Pattern

**Dual-layer retry system**: Internal graph retry (disabled by default) + External wrapper retry

All nodes except `error_node` are wrapped with `with_error_handling(node_name, error_type, retryable=True)`:

```python
graph.add_node(
    "finance_agent_node",
    with_error_handling("finance_agent_node", "LLM")(finance_agent_node)
)
```

On error:

1. Wrapper catches exception, populates `state["error"]` with node name, message, type, attempt count
2. Routes to `error_node` which logs error and increments retry counter
3. `retry_router` checks `attempt < MAX_RETRIES` (default: 0) and routes back to failed node or END
4. External `run_with_auto_retry()` wrapper provides cross-restart retry capability
5. **MUST map all retryable nodes** in `error_node` conditional edges:
   ```python
   graph.add_conditional_edges("error_node", retry_router, {
       "intent_classifier": "intent_classifier",
       "finance_agent_node": "finance_agent_node",
       # ... all other nodes
       END: END
   })
   ```

**Production workflow for fixing errors:**

1. Node fails → Checkpoint saved with error state
2. Developer fixes bug in code
3. App resumes with `payload=None` → Clears error with `{"error": None}` → Retries with fixed code

### External Dependencies

- **MongoDB** (`localhost:27017`): Conversation checkpointing with `MongoDBSaver`
- **Qdrant Vector DB** (`localhost:6333` from `.env`):
  - Collection: `VECTOR_DB_COLLECTION_NAME`
  - Fields: `user_id`, `category`, `summary`, `topics`, vector embeddings
  - Use `query_points()` with `Filter` and `FieldCondition`
- **OpenAI API**: gpt-4o-mini, embeddings: `text-embedding-3-small`

## Key Patterns

### 0. Automatic Retry with Checkpointing (CRITICAL)

**Simple usage** (recommended):

```python
from utils.retry_handler import run_with_auto_retry

result = await run_with_auto_retry(
    app=app,
    payload=initial_state,  # or None to resume from checkpoint
    config={"configurable": {"thread_id": "user-123"}},
    max_retries=3,
    initial_backoff=1.0
)
```

**How it works:**

- Attempt 1: `app.ainvoke(payload)` → Node fails → Checkpoint saved
- Attempt 2+: `app.ainvoke({"error": None})` → Clears error, resumes from checkpoint
- Skips successful nodes, retries only failed node with exponential backoff

**Advanced with callbacks:**

```python
from utils.retry_handler import AutoRetryHandler, RetryStrategy

handler = AutoRetryHandler(
    max_retries=5,
    initial_backoff=2.0,
    strategy=RetryStrategy.FIBONACCI,  # exponential, linear, fixed, fibonacci
    on_retry=lambda attempt, backoff: send_alert(f"Retry {attempt}"),
    on_error=lambda error, attempt: log_failure(error)
)
result = await handler.run(app, payload, config)
```

**Resume after app restart:**

```python
# Use same thread_id, pass None to resume from checkpoint
result = await run_with_auto_retry(
    app=app,
    payload=None,  # ← Resumes from last checkpoint
    config={"configurable": {"thread_id": "user-123"}},
    max_retries=3
)
```

### 1. Dual Architecture: Agents vs Nodes

**Agents** (use LangChain tools):

```python
# nodes/finance_agent_node/agent.py
finance_agent = create_agent(model, tools=[retriever_tool], system_prompt=state["rule_prompt"])
async def finance_agent_node(state: State):
    result = await finance_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    # Parse JSON response with answer and missing_info
    state["finance_agent"] = {"answer": {...}, "missing_info": [...]}
    return state
```

**Nodes** (direct data processing):

```python
# nodes/rule_builder_node/__init__.py
async def rule_builder_node(state: State) -> State:
    tier = state.get("tier", "FREE")
    base_prompt = load_prompt("finance_agent_prompt")
    prompt = inject_rules(base_prompt=base_prompt, rules=RULES[tier])
    return {"rule_prompt": prompt}  # Partial state update
```

### 2. Tier-Based Prompt Customization

`rule_builder_node` injects tier-specific rules into prompt placeholders:

```python
# prompts/finance_agent_prompt.txt contains: {{response_style}}
# RULES["PAID"]["response_style"] = ["Provide detailed explanations...", ...]
# Result: Placeholder replaced with bullet list of rules
```

### 3. Missing Info Flow (Finance → Followup)

`finance_agent_node` returns JSON with `missing_info` array:

```python
{"answer": "Sorry, we don't have this data.", "missing_info": ["credit score"]}
```

`route_after_finance_agent()` checks for `missing_info` and routes to `followup_agent_node`:

```python
if finance_agent_response.get("missing_info") and len(missing_info) > 0:
    return "followup_agent_node"
```

`followup_agent_node` asks user to complete profile with examples:

```python
# Prompt: "Please complete your profile. You can mention your credit score (e.g., 750)."
```

### 4. JSON Response Parsing Pattern

Finance agent returns strict JSON. Node parses with fallback:

```python
try:
    parsed = json.loads(response)
    state["finance_agent"] = {
        "answer": parsed.get("answer", {}),
        "missing_info": parsed.get("missing_info", [])
    }
except (json.JSONDecodeError, TypeError):
    # Fallback: wrap raw response
    state["finance_agent"] = {"answer": {"response": response}, "missing_info": []}
```

### 5. Conditional Routing Pattern

All routing functions must map all possible return values:

```python
def route_after_finance_agent(state: State) -> str:
    if state.get("error"):
        return "error_node"

    finance_agent_response = state.get("finance_agent", {})
    missing_info = finance_agent_response.get("missing_info", [])

    if missing_info and len(missing_info) > 0:
        return "followup_agent_node"

    return END

# In app.py graph definition:
graph.add_conditional_edges(
    "finance_agent_node",
    route_after_finance_agent,
    {
        "error_node": "error_node",
        "followup_agent_node": "followup_agent_node",
        END: END  # ALL possible returns must be mapped
    }
)
```

### 6. Qdrant Query Patterns

**With category filter**:

```python
Filter(must=[
    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
    FieldCondition(key="category", match=MatchValue(value="CREDIT_SCORE"))
])
qdrant_client.query_points(collection_name=..., query_filter=filter, limit=10).points
```

**With vector search**:

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embeddings.embed_query("credit score")
qdrant_client.query_points(collection_name=..., query=vector, query_filter=user_filter, limit=5).points
```

## Development Workflow

### Running the Application

```bash
python app.py  # Executes query at bottom of file
```

### Testing Retry Mechanism

```bash
python test_retry.py  # Runs 5 test scenarios for retry functionality
```

### Production API Example

```bash
pip install fastapi uvicorn  # If not installed
python api_example.py  # FastAPI server with automatic retry on port 8000
```

### Environment Setup

`.env` file must contain:

```bash
OPENAI_API_KEY=sk-proj-...
VECTOR_DB_URL=http://localhost:6333
VECTOR_DB_COLLECTION_NAME=credit_data
```

**MongoDB** must be running on `localhost:27017` for checkpointing.

### Adding a New Agent Node

1. `mkdir nodes/{name}` with `__init__.py`, `agent.py`, `prompt.py`
2. In `prompt.py`: Define `SYSTEM_PROMPT = "..."`
3. In `agent.py`:
   - Import: `from services.opanai_service import model`
   - Import tools: `from tools.{tool} import {tool}`
   - Create: `agent = create_agent(model, tools=[...], system_prompt=SYSTEM_PROMPT)`
   - Implement: `async def {name}_node(state: State): ...` (update specific state key)
4. In `__init__.py`: `from .agent import {name}_node`
5. In `app.py`:
   - Add node with error handling: `graph.add_node("{name}", with_error_handling("{name}", "LLM")({name}_node))`
   - Add to conditional edges mappings
   - Update routing logic

### Simulating Errors for Testing

To test retry mechanism, add a simulated error in any node:

```python
# In nodes/finance_agent_node/agent.py
async def finance_agent_node(state: State):
    # Simulate error for testing
    raise ValueError("🧪 Simulated error for testing automatic retry")

    # Normal code below...
```

Run the app and observe automatic retry attempts with exponential backoff.

### Adding a New Tool

1. In `tools/{name}.py`: Use `@tool` decorator
2. Tool wraps either:
   - Node function: `result = node_function(state); return result["context"]`
   - Repository function: `return fetch_from_db(user_id)`
3. Add comprehensive docstring (agent uses this to decide when to call)
4. Import in agent's `agent.py` and add to tools list

### Adding Semantic Search to Node

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
for topic in topics:
    vector = embeddings.embed_query(topic)
    results = qdrant_client.query_points(
        collection_name=os.getenv("VECTOR_DB_COLLECTION_NAME"),
        query=vector,
        query_filter=user_filter,
        limit=5
    ).points
```

## Project-Specific Conventions

- **Import paths**: Absolute from project root
  - ✅ `from services.opanai_service import model`
  - ❌ `from ..services.opanai_service import model`
- **Naming**:
  - Agent nodes: `{name}_node` (e.g., `finance_agent_node`)
  - State keys for answers: `{agent}` or `{agent}_answer` (e.g., `finance_agent`, `web_answer`)
  - Repository functions: `fetch_{data}_from_{source}`
  - Async functions: All node functions must be `async def`
- **State keys**: Each node writes to unique state key to avoid conflicts
- **Conditional routing**: Always map all possible returns including `END`
- **Qdrant API**: Use `query_points()`, not `search()` or `search_points()`
- **Error handling**: Flexible parsing for OpenAI responses (handle dict/list formats)
- **Prompt loading**: Use `load_prompt(name)` from `utils/prompt_loader.py` for external prompts
- **Checkpointing**:
  - MongoDB stores full state after each node execution
  - Pass `None` to `app.ainvoke()` to resume from last checkpoint
  - Use unique `thread_id` per user/session: `f"user-{user_id}-session-{session_id}"`
  - On retry: Pass `{"error": None}` to clear error state while resuming from checkpoint

## Common Patterns

### Wrapping Node as Tool

```python
# tools/fetch_user_info.py
from nodes.user_info_fetch import user_info_fetch as user_info_fetch_node

@tool
def fetch_user_info(user_id: str, query: str):
    """Tool docstring for agent"""
    state = {"user_id": user_id, "query": query, "context": {}}
    result_state = user_info_fetch_node(state)
    return result_state["context"]
```

### OpenAI Classification Pattern

```python
response = model.invoke([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query}
])
classification = json.loads(response.content)
topics = classification.get("requires", ["default", "topics"])
```

### Deduplication After Multi-Topic Search

```python
unique_results = {}
for result in all_results:
    if result.id not in unique_results:
        unique_results[result.id] = result
return list(unique_results.values())
```

### Prompt Template Loading and Variable Injection

```python
# Load prompt from prompts/ directory
base_prompt = load_prompt("finance_agent_prompt")

# Replace placeholder {{variable_name}} with actual value
prompt = base_prompt.replace("{{response_style}}", "\n".join(rules))
```
