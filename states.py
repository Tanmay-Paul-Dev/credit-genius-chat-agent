from typing import TypedDict
from typing import Optional, Literal, Dict
from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import StateGraph, START, END, MessagesState
from enum import Enum
from typing import List, Any
from pydantic import BaseModel, Field

ErrorType = Literal["LLM", "TOOL", "NETWORK", "LOGIC"]


class IntentEnum(str, Enum):
    finance = "finance"
    faq = "faq"
    non_finance = "non_finance"
    greeting = "greeting"


class ErrorState(TypedDict):
    node: str
    message: str
    type: ErrorType
    retryable: bool
    attempt: int


class FinanceAgentState(BaseModel):
    answer: str = Field(description="Answer to the user query")
    missing_info: list[str] = Field(description="List of missing information")


class IntentClassifierState(BaseModel):
    query_type: str = Field(
        description="Type of query: finance, faq, non_finance, greeting"
    )
    intent: str = Field(description="Specific user intent")
    required_info: List[str] = Field(
        default_factory=list, description="Required info fields"
    )
    optional_info: List[str] = Field(
        default_factory=list, description="Optional info fields"
    )


class IntentClassification(TypedDict):
    intent: str
    required_info: Dict[str, Any]


class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(
        description="True if this memory is NEW and should be stored. False if duplicate/already known."
    )


class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether to store any memories")
    memories: List[MemoryItem] = Field(
        default_factory=list, description="Atomic user memories to store"
    )


class MemoryLookupResult(BaseModel):
    found_in_memory: bool = Field(
        description="True if the query can be answered from existing memories"
    )
    memories: List[str] = Field(
        default_factory=list,
        description="List of relevant memories if found, otherwise empty list",
    )


class State(TypedDict):
    query: str
    found_in_memory: bool
    memory_context: Optional[List[str]]  # Retrieved memories relevant to query
    messages: MessagesState
    is_profile_complete: bool
    intent: IntentClassification
    tier: str  # TODO: Make it subscription_type
    user_id: str
    final_answer: Optional[str]
    rule_prompt: Optional[str]
    finance_agent: Optional[FinanceAgentState]
    retrieved_data: Optional[Dict[str, any]]  # Data retrieved from vector store
    error: Optional[ErrorState]
