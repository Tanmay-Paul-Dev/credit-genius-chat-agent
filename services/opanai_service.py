from langchain_openai import ChatOpenAI
import os


def initialize_model():
    """Initialize and return the OpenAI model for the web agent."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.8)


model = initialize_model()
