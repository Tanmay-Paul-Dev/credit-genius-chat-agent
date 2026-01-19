from langchain_openai import ChatOpenAI
import os


def initialize_model():
    """Initialize and return the OpenAI model for the web agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=api_key)


model = initialize_model()
