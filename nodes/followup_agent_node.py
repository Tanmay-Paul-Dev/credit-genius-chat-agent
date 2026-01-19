from langchain.agents import create_agent
from typing import TypedDict
from services.opanai_service import model
from tools.fetch_user_info import fetch_user_info
from tools.retrieve_knowledge import retriever_tool
from states import State
from utils.prompt_loader import load_prompt
import json

from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.runnables import RunnableConfig

followup_agent = create_agent(
    model,
    tools=[],
    system_prompt=load_prompt("followup_agent_prompt"),
)


async def followup_agent_node(state: State, config: RunnableConfig, store: BaseStore):
    user_id = state.get("user_id", "")
    finance_agent_response = state.get("finance_agent", {})
    missing_info = finance_agent_response.get("missing_info", [])

    # Construct query with missing info context
    full_query = f"User ID: {user_id}\nMissing information: {', '.join(missing_info)}\n\nPlease ask the user to provide these missing details."

    result = await followup_agent.ainvoke(
        {"messages": [{"role": "user", "content": full_query}]}
    )

    # Extract the final answer from the agent's response
    response = json.loads(result["messages"][-1].content)
    return {"followup_agent": response, "final_answer": response.get("answer", "")}
