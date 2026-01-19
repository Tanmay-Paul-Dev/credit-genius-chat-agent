from states import State
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

MAX_TOKEN = 150


def agent_message_builder(state: State):
    messages = []

    # Add messages from state
    state_messages = state.get("messages", [])
    messages.extend(state_messages)

    return messages
