import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Connect to Neo4j
from langchain_neo4j import Neo4jGraph

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)