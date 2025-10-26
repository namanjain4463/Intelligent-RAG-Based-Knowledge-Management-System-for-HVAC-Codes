"""
Neo4j Graph Database Connection
"""
from langchain_neo4j import Neo4jGraph
from config import Config

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=Config.NEO4J_URI,
    username=Config.NEO4J_USERNAME,
    password=Config.NEO4J_PASSWORD,
    database=Config.NEO4J_DATABASE
)

print(f"[OK] Connected to Neo4j at {Config.NEO4J_URI}")

