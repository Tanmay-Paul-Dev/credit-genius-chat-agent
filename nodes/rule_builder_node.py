from states import State
from typing import Dict, List
from utils.prompt_loader import load_prompt

from langgraph.store.base import BaseStore
from states import State, MemoryDecision
from langchain_core.runnables import RunnableConfig

RULES = {
    "FREE": {
        "response_style": [
            "Keep responses short and focused.",
            "Use simple, everyday language. Avoid technical terms unless necessary.",
        ]
    },
    "PAID": {
        "response_style": [
            "Provide detailed and thorough explanations.",
            "Include relevant context, background, and reasoning.",
            "Explain concepts step by step when appropriate.",
            "Use clear structure so the response is easy to follow.",
            "Cover edge cases, limitations, or important considerations when relevant.",
            "Prioritize clarity and completeness over brevity.",
        ]
    },
}


def render_rules(rule_list: List[str]) -> str:
    """
    Converts rule list into prompt-ready bullet text
    """
    return "\n".join(f"- {rule}" for rule in rule_list)


def inject_rules(
    base_prompt: str,
    rules: Dict[str, List[str]],
) -> str:
    """
    Injects rules into placeholders like {{response_style}}
    """
    rendered_prompt = base_prompt

    for key, rule_list in rules.items():
        placeholder = f"{{{{{key}}}}}"
        rendered_prompt = rendered_prompt.replace(placeholder, render_rules(rule_list))

    return rendered_prompt


# TODO: RENAME THIS FUNCTION TO A MORE DESCRIPTIVE NAME
# TODO: Return only field(s) that are modified


async def rule_builder_node(
    state: State, config: RunnableConfig, store: BaseStore
) -> State:
    tier = state.get("tier", "FREE")

    base_prompt = load_prompt("finance_agent_prompt")

    tier_rules = RULES.get(tier, {})

    prompt = inject_rules(base_prompt=base_prompt, rules=tier_rules)

    return {"rule_prompt": prompt}
