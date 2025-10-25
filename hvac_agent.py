"""
HVAC Agent with CORRECT schema matching property_rich_hvac.py database
Updated to use: CodeSection, Standard, OccupancyType, Equipment, HVACChunk
"""

__all__ = ['generate_response']

from llm import llm
from graph import graph
from tools.vector import search_knowledge_base
from tools.cypher import cypher

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

# Create tools
HVACInfoTool = Tool.from_function(
    search_knowledge_base,
    name="HVACInfo",
    description="Get detailed information about HVAC codes, sections, standards, equipment, and occupancy requirements from semantic search of the HVAC Neo4j database."
)

CypherQueryTool = Tool.from_function(
    cypher,
    name="CypherQuery",
    description="Execute structured queries against the HVAC Neo4j knowledge graph to find relationships between code sections, standards, equipment, and occupancy types."
)

tools = [HVACInfoTool, CypherQueryTool]

AGENT_TEMPLATE = """
You are a STRICT HVAC codes assistant with access ONLY to a Neo4j database 
containing HVAC code information, equipment specifications, installation 
requirements, and regulatory standards.

*** CRITICAL RESTRICTIONS: ***
- You can ONLY answer questions about HVAC equipment, codes, sections, 
  standards, and occupancy requirements that exist in the database
- You MUST NOT answer questions about anything outside of HVAC codes 
  (e.g., general knowledge, people, companies, current events)
- If a question is not related to HVAC or the database, respond: 
  "I can only answer questions about HVAC codes and standards from our database."
- You MUST use the HVACInfo or CypherQuery tools for ALL HVAC questions

*** Strictly use only these node types and relationships for queries: ***

Nodes (ACTUAL DATABASE SCHEMA):
- CodeSection (number, title, contentPreview, requirements, referencedSections, hasTable, hasException)
- Standard (code, purpose, equipmentTypes, organization)
- OccupancyType (name, defaultAirflowCfm, combinedAirflowCfm, sourceSection, category)
- Equipment (name, applicableSections, requirements)
- HVACChunk (chunkId, text, section, index, chunkType)

You must only use these relationships:
- (HVACChunk)-[:BELONGS_TO_SECTION]->(CodeSection)
- (CodeSection)-[:REFERENCES]->(CodeSection)
- (Equipment)-[:REGULATED_BY]->(CodeSection)
- (OccupancyType)-[:DEFINED_IN]->(CodeSection)
- (Standard)-[:TESTS]->(Equipment)
- (HVACChunk)-[:MENTIONS]->(Standard)

Do not use or invent any other relationship types.

IMPORTANT: The database contains CodeSection, Standard, OccupancyType, Equipment, 
and HVACChunk nodes with the relationships listed above.

*** Example Cypher queries: ***
1. MATCH (s:CodeSection) RETURN s.number, s.title LIMIT 10
2. MATCH (e:Equipment)-[:REGULATED_BY]->(s:CodeSection) RETURN s.number, s.title
3. MATCH (o:OccupancyType) WHERE toFloat(o.defaultAirflowCfm) > 0.5 RETURN o.name, o.defaultAirflowCfm
4. MATCH (std:Standard)-[:TESTS]->(e:Equipment) RETURN std.code, std.purpose, e.name
5. MATCH (s1:CodeSection)-[:REFERENCES]->(s2:CodeSection) RETURN s1.number, s2.number
6. MATCH (c:HVACChunk)-[:BELONGS_TO_SECTION]->(s:CodeSection) RETURN c.text LIMIT 5
7. MATCH (o:OccupancyType) RETURN o.name, o.defaultAirflowCfm

Always try to use the database tools first for HVAC-specific questions.

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do - if the question is not 
         about HVAC, decline to answer. Otherwise, prioritize using HVACInfo 
         or CypherQuery tools for HVAC-specific questions
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought:{agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(AGENT_TEMPLATE)

agent = create_react_agent(llm, tools, agent_prompt)

executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=30,
    early_stopping_method="force"
)

def generate_response(user_input: str):
    """Generate response using the HVAC agent"""
    exec_obj = globals().get('executor') or globals().get('chat_agent') or globals().get('agent_executor')
    if exec_obj is None:
        raise RuntimeError("Agent executor not found; ensure you create the executor before calling generate_response.")
    return exec_obj.invoke({"input": user_input})
