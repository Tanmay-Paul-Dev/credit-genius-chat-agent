# LangGraph Finance Agent - AI Coding Instructions

## Architecture Overview

This is a **LangGraph-based conversational finance agent** designed for AWS Lambda deployment. The system routes user queries through a multi-node graph where each node performs specific tasks (intent classification, data retrieval, agent reasoning, memory management).

**Key Flow**: User Query → Intent Classification → Memory Retrieval → Data Retrieval → Finance/Chat Agent → Response + Memory Storage

## Core Architecture Patterns

### 1. **Graph Structure** (`app.py`, `lambda_handler.py`)

- All nodes are wrapped with `with_error_handling(node_name, error_type)` decorator for centralized error handling
- Nodes must have signature: `async def node(state: State, config: RunnableConfig, store: BaseStore)`
- Conditional routing is defined in `repositories/conditional_repository.py` - routes return node names as strings
- Graph compilation uses `StateGraph(State)` with MongoDBStore for persistence
- Entry points use `START` constant, exits use `END` constant or named exit nodes

### 2. **State Management** (`states.py`)

- Central `State` TypedDict contains all graph state (query, messages, intent, retrieved_data, memory_context, etc.)
- Structured output models use Pydantic BaseModel (e.g., `IntentClassifierState`, `FinanceAgentState`)
- Error state tracking: `ErrorState` with node, message, type, retryable, and attempt count
- State updates return partial dict that merges into existing state

### 3. **Memory System** (MongoDB-backed)

- Uses LangGraph's `MongoDBStore` for user memory persistence
- Namespace pattern: `("user", user_id, "details")` for user-specific memories
- Memory retrieval happens BEFORE intent classification to check if query answerable from memory
- Memory storage happens AFTER chat response in background (see `save_response_memory_background`)
- LLM-driven memory decisions using `MemoryDecision` model (determines what's worth storing)

### 4. **Async Patterns**

- **All node functions are async** - use `await` for LLM calls, tool invocations, store operations
- Terminal chat uses `asyncio.run(chat())` - see `app.py` main block
- Lambda handler uses `asyncio.run(process_message(...))` for AWS Lambda compatibility
- Background tasks run synchronously after main response (memory storage is not awaited)

## Node Development Conventions

### Adding New Nodes

1. Create file in `nodes/` with async function matching signature: `async def my_node(state: State, config: RunnableConfig, store: BaseStore)`
2. Add error handling wrapper in graph builder: `with_error_handling("node_name", "LLM")(my_node)`
3. Add conditional routing logic in `repositories/conditional_repository.py` if needed
4. Connect edges in graph builder using `graph.add_edge()` or `graph.add_conditional_edges()`

### Node Implementation Pattern

```python
async def example_node(state: State, config: RunnableConfig, store: BaseStore):
    # 1. Extract data from state
    query = state.get("query", "")
    user_id = config["configurable"]["user_id"]

    # 2. Access memory store if needed
    ns = ("user", user_id, "details")
    items = list(store.search(ns))

    # 3. Call LLM with structured output
    model = large_model.with_structured_output(MyOutputModel)
    result = await model.ainvoke(prompt)

    # 4. Return state update (partial dict)
    return {"field_name": result.field_value}
```

## Service & Integration Patterns

### OpenAI Service (`services/opanai_service.py`)

- Pre-initialized models: `small_model` (gpt-4o-mini), `large_model` (gpt-5.2)
- Use `.with_structured_output(PydanticModel)` for type-safe LLM responses
- Structured output uses `method="function_calling"` parameter

### Vector Store (`services/pinecone_service.py`)

- Pinecone vectorstore initialized with namespace from env (`PINECONE_NAMESPACE`)
- MMR retrieval via custom `MMRRetriever` class in `retriever/__init__.py`
- User-scoped queries filter by `{"userId": {"$eq": user_id}}` metadata
- Retriever pattern: `retriever.retrieve(query, user_id=user_id)` returns list of Documents

### Prompt Loading (`utils/prompt_loader.py`)

- Prompts stored as `.txt` files in `prompts/` directory
- Load with: `load_prompt("prompt_name")` - auto-resolves to `prompts/prompt_name.txt`
- Supports variable substitution: `load_prompt("name", var1="value")` replaces `{{var1}}` in prompt

## Critical Workflows

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env)
OPENAI_API_KEY=sk-...
MONGODB_URI=mongodb://localhost:27017
PINECONE_API_KEY=...
PINECONE_DB_NAME=credit-genius-production

# Run terminal chat interface
python app.py
```

### Lambda Deployment

- Entry point: `lambda_handler.handler(event, context)` in `lambda_handler.py`
- Expects event with: `{"user_id": "...", "query": "...", "thread_id": "..."}`
- Returns: `{"statusCode": 200, "body": json_response}`
- Lambda graph is simplified (fewer nodes than `app.py` development version)

### Testing Individual Nodes

```python
# Create minimal test state
test_state = {
    "query": "test query",
    "messages": [],
    "user_id": "test-user",
    "intent": {},
    "tier": "PAID"
}
test_config = {"configurable": {"user_id": "test-user", "thread_id": "test"}}
# Call node directly (requires store initialization)
```

## Key Differences & Gotchas

1. **Two Graph Definitions**: `app.py` (full development graph with all nodes) vs `lambda_handler.py` (simplified production graph)
2. **Config Access**: User context via `config["configurable"]["user_id"]`, thread via `config["configurable"]["thread_id"]`
3. **Message Trimming**: Use `trim_messages()` from langchain_core to stay within context limits (see `finance_agent_node.py`)
4. **Store vs Repository**: `store` is LangGraph's memory system, `repositories/` contains routing logic (not data access)
5. **Error Retry Logic**: Errors increment `attempt` counter, but actual retry logic must be implemented in routing functions
6. **Tool Usage**: Tools defined in `tools/` are passed to agents (see `finance_agent_node.py` using `create_agent(tools=[...])`)

## Common Tasks

**Add new intent type**: Update `IntentEnum` in `states.py` + add routing case in `route_after_intent_classification()`

**Modify prompt**: Edit `.txt` file in `prompts/` - changes apply immediately (no reload needed)

**Add new retrieval field**: Update `required_info` or `optional_info` lists in intent classifier prompt

**Change memory behavior**: Modify `save_response_memory_background()` in `chat_node.py` - controls what gets persisted

**Debug graph execution**: Check console output - nodes print status with emoji prefixes (🧠, 📌, 🔧, etc.)

**Adjust model selection**: Import and use `small_model` or `large_model` from `services/opanai_service.py` based on task complexity
