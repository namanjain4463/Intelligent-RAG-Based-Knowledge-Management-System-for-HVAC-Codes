"""
LLM Initialization for HVAC GraphRAG System
Uses OpenAI GPT-4 for Cypher generation and reasoning
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import Config

# Initialize LLM
llm = ChatOpenAI(
    model=Config.OPENAI_MODEL,
    temperature=0.0,  # Deterministic for Cypher generation
    api_key=Config.OPENAI_API_KEY
)

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model=Config.OPENAI_EMBEDDING_MODEL,
    api_key=Config.OPENAI_API_KEY
)
