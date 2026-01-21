import json
from langchain.agents import create_agent
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from services.opanai_service import model
from tools.retrieve_knowledge import retriever_tool
from tools.web_search_tool import web_search_tool
from states import State
from langchain_core.runnables import RunnableConfig

from langgraph.store.base import BaseStore
from states import State, MemoryDecision, FinanceAgentState
from langchain_core.runnables import RunnableConfig
from retriever import MMRRetriever
from services.pinecone_service import vectorstore
from utils.prompt_loader import load_prompt


async def finance_agent_node(state: State, config: RunnableConfig, store: BaseStore):
    query = state.get("query", "")
    user_id = config["configurable"]["user_id"]
    retrieved_data = state.get("retrieved_data", {})
    required_info = state["intent"].get("required_info", {})

    # Trim messages for context window
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=2000,
    )

    memories = (
        "\n".join(f"- {it}" for it in state.get("memory_context", []))
        if state.get("memory_context")
        else "(empty)"
    )

    context_message = json.dumps(
        {
            "user_query": query,
            "requirements": {"required_info": required_info},
            "retrieved_data": {
                "resolved": retrieved_data.get("required_info", {}),
                "optional": retrieved_data.get("optional_info", {}),
                "evidence": retrieved_data.get("retrived_intent_info"),
            },
            "memory": memories,
            "conversation": state["messages"],
        },
        indent=2,
    )

    # 3️⃣ Create agent with retriever and web_search tools
    finance_agent: FinanceAgentState = create_agent(
        model=model,
        tools=[web_search_tool],
        system_prompt=load_prompt("finance_agent_prompt"),
    )

    # 4️⃣ Invoke agent with proper message format
    agent_messages = [{"role": "user", "content": context_message}]
    result = await finance_agent.ainvoke({"messages": agent_messages})

    # 🔍 Debug: Log tool calls
    print("\n" + "=" * 60)
    print("🔧 TOOL CALLS DEBUG")
    print("=" * 60)
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"\n📞 Tool: {tool_call.get('name', 'unknown')}")
                print(f"   Args: {json.dumps(tool_call.get('args', {}), indent=8)}")
        if hasattr(msg, "type") and msg.type == "tool":
            print(f"\n✅ Tool Response: {msg.name}")
            content_preview = (
                str(msg.content)[:200] + "..."
                if len(str(msg.content)) > 200
                else str(msg.content)
            )
            print(f"   Content: {content_preview}")
    print("=" * 60 + "\n")

    raw_content = result["messages"][-1].content

    # 5️⃣ Safe JSON parsing
    try:
        response = json.loads(raw_content)
    except json.JSONDecodeError:
        response = {"answer": "Sorry, I couldn't process your request properly."}

    return {
        **state,
        "messages": state["messages"],
        "finance_agent": response,
    }
