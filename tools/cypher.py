"""
HVAC Cypher Generator with CORRECT schema matching property_rich_hvac.py database
Updated to use: CodeSection, Standard, OccupancyType, Equipment, HVACChunk
"""

from llm import llm
from graph import graph

from langchain.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain

# CORRECTED CYPHER GENERATION TEMPLATE - MATCHES ACTUAL DATABASE SCHEMA
CYPHER_GENERATION_TEMPLATE = """
You are a STRICT HVAC codes assistant with access to a Neo4j database 
containing HVAC code information, equipment specifications, installation 
requirements, and regulatory standards.

*** Strictly use only these node types and relationships for queries: ***

Nodes (ACTUAL DATABASE SCHEMA):
- CodeSection {{number, title, contentPreview, requirements, referencedSections, hasTable, hasException}}
- Standard {{code, purpose, equipmentTypes, organization}}
- OccupancyType {{name, defaultAirflowCfm, combinedAirflowCfm, sourceSection, category}}
- Equipment {{name, applicableSections, requirements}}
- HVACChunk {{chunkId, text, section, index, chunkType}}

You must only use these relationships:
- (HVACChunk)-[:BELONGS_TO_SECTION]->(CodeSection)
- (CodeSection)-[:REFERENCES {{type}}]->(CodeSection)
- (Equipment)-[:REGULATED_BY {{basis}}]->(CodeSection)
- (OccupancyType)-[:DEFINED_IN]->(CodeSection)
- (Standard)-[:TESTS {{applicability}}]->(Equipment)
- (HVACChunk)-[:MENTIONS]->(Standard)

Do not create any other relationship types.

IMPORTANT: The database contains CodeSection, Standard, OccupancyType, Equipment, 
and HVACChunk nodes with the relationships listed above.

*** Example Cypher queries: ***
1. MATCH (s:CodeSection) RETURN s.number, s.title LIMIT 10
   // List all code sections

2. MATCH (e:Equipment {{name:"Exhaust Fan"}})-[:REGULATED_BY]->(s:CodeSection) 
   RETURN s.number, s.title, s.requirements
   // Find sections regulating exhaust fans

3. MATCH (o:OccupancyType) WHERE toFloat(o.defaultAirflowCfm) > 0.5 
   RETURN o.name, o.defaultAirflowCfm ORDER BY toFloat(o.defaultAirflowCfm) DESC
   // Find occupancies with high airflow requirements

4. MATCH (std:Standard)-[:TESTS]->(e:Equipment) 
   RETURN std.code, std.purpose, e.name
   // Find which standards test which equipment

5. MATCH (s1:CodeSection)-[:REFERENCES]->(s2:CodeSection) 
   RETURN s1.number, s2.number
   // Find cross-references between sections

6. MATCH (c:HVACChunk)-[:BELONGS_TO_SECTION]->(s:CodeSection {{number:"403.3.1.1"}}) 
   RETURN c.text ORDER BY c.index
   // Get full text of a specific section

7. MATCH (o:OccupancyType {{category:"Residential"}}) 
   RETURN o.name, o.defaultAirflowCfm
   // Get residential occupancy types

8. MATCH (s:CodeSection) WHERE s.number STARTS WITH "501" 
   RETURN s.number, s.title
   // Find all sections in Chapter 5 (Exhaust Systems)

9. MATCH (std:Standard {{organization:"UL"}}) 
   RETURN std.code, std.purpose
   // Find all UL standards

10. MATCH (e:Equipment)-[:REGULATED_BY]->(s:CodeSection) 
    WHERE s.number CONTAINS "403" 
    RETURN e.name, s.number, s.title
    // Find equipment regulated by Chapter 4 (Ventilation)

IMPORTANT: Always write your response in clear, single-line format without 
unnecessary line breaks or formatting.

Schema: {schema}
Question: {question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    return_intermediate_steps=True
)

last_query = None
last_generated_cypher = None

def cypher(query: str) -> str:
    """
    Execute a Cypher query using the GraphCypherQAChain.
    If the query looks like raw Cypher, execute it directly.
    Otherwise, use the LLM to generate Cypher from natural language.
    """
    global last_query, last_generated_cypher
    
    try:
        query_upper = query.upper().strip()
        if query_upper.startswith(("MATCH", "CREATE", "MERGE", "DELETE", "RETURN", "WITH")):
            # Execute raw Cypher directly
            result = graph.query(query)
            last_query = query
            last_generated_cypher = query
            
            if result:
                return str(result)
            else:
                return "No results found."
        else:
            # Use LLM to generate Cypher from natural language
            result = cypher_qa.invoke({"query": query})
            last_query = query
            
            # Extract the generated Cypher from intermediate steps
            if "intermediate_steps" in result and result["intermediate_steps"]:
                for step in result["intermediate_steps"]:
                    if isinstance(step, dict) and "query" in step:
                        last_generated_cypher = step["query"]
                        break
                    elif isinstance(step, str) and ("MATCH" in step or "RETURN" in step):
                        last_generated_cypher = step
                        break
                else:
                    last_generated_cypher = "Generated Cypher query (unable to capture)"
            else:
                last_generated_cypher = "Generated Cypher query (no intermediate steps)"
            
            response = result.get("result", str(result))
            return ' '.join(response.split())
    except Exception as e:
        last_query = None
        last_generated_cypher = None
        return f"Error executing Cypher query: {str(e)}"
