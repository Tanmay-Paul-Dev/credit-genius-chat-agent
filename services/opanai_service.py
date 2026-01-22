from langchain_openai import ChatOpenAI
import os


def initialize_model(model_name="gpt-4o-mini"):
    """Initialize and return the OpenAI model for the web agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return ChatOpenAI(model=model_name, temperature=0.1, api_key=api_key)


small_model = initialize_model()
large_model = initialize_model("gpt-5.2")
