from pathlib import Path

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, **vars):
    path = PROMPT_DIR / f"{name}.txt"
    text = path.read_text()

    for k, v in vars.items():
        text = text.replace(f"{{{{{k}}}}}", str(v))

    return text
