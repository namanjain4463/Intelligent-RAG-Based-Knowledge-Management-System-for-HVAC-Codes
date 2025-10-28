"""
Main HVAC Agent with 3 Tools (Vector, Cypher, Hybrid)
Uses LangChain ReAct agent for intelligent tool selection
"""
import re
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from llm import llm, embeddings
from graph import graph
from cypher_generator import generate_cypher, execute_cypher, extract_entities
from config import Config

# ==================== TOOL 1: Vector Search ====================

def vector_search(query: str) -> str:
    """
    Semantic search using vector embeddings - FALLBACK ONLY when Cypher fails
    
    Strategy: Cypher-first, vector-fallback
    - Prefer structured graph queries (Cypher) for domain relationships
    - Use vector search ONLY when:
      1. Cypher query returns no results
      2. User asks for general explanations/procedures
      3. Need to search unstructured section text
    
    Returns: Relevant text chunks with section references
    """
    try:
        
        # Check if vector index exists
        index_check = graph.query("SHOW INDEXES")
        vector_index_exists = any(
            idx.get('name') == Config.VECTOR_INDEX_NAME 
            for idx in index_check
        )
        
        if not vector_index_exists:
            return "Vector search unavailable: index not found. Try using CypherQuery for structured domain questions."
        
        # Generate embedding for query
        query_embedding = embeddings.embed_query(query)
        
        # Search vector index
        vector_query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
        YIELD node, score
        MATCH (node)-[:BELONGS_TO]->(s:Section)
        RETURN s.number as section_number,
               s.title as section_title,
               node.text as text,
               score
        ORDER BY score DESC
        LIMIT $top_k
        """
        
        results = graph.query(vector_query, {
            "index_name": Config.VECTOR_INDEX_NAME,
            "top_k": 5,
            "query_embedding": query_embedding
        })
        
        if results:
            formatted = []
            for r in results:
                formatted.append(
                    f"Section {r['section_number']} ({r['section_title']}):\n{r['text']}\n(Similarity: {r['score']:.3f})"
                )
            return "\n\n".join(formatted)
        else:
            return "No relevant content found via vector search."
            
    except Exception as e:
        return f"Vector search error: {str(e)}"


def cypher_query(query: str) -> str:
    """
    Structured graph query using Cypher with Phase 2A Domain Relationships
    
    Best for:
    - Equipment PROHIBITIONS: "Where can't I install X?" (PROHIBITED_IN relationship)
    - Equipment CLEARANCES: "What clearances does X need?" (REQUIRES_CLEARANCE relationship)
    - Equipment STANDARDS: "What standards does X comply with?" (MUST_COMPLY_WITH relationship)
    - Equipment SAFETY DEVICES: "What safety devices does X require?" (REQUIRES_DEVICE relationship)
    - Equipment PERMITTED LOCATIONS: "Where can I install X?" (PERMITTED_IN relationship)
    - Section lookups: "What is Section 306?"
    - Equipment lists: "What equipment is in the database?"
    - Hierarchies: "What subsections are in Chapter 3?"
    
    All domain relationships include section references (code_ref property) for traceability.
    """
    try:
        
        # Generate Cypher from natural language
        cypher = generate_cypher(query)
        
        # AUTO-FIX: Ensure code_ref is included in RETURN clause for relationship queries
        if any(rel in cypher for rel in ['PROHIBITED_IN', 'REQUIRES_CLEARANCE', 'MUST_COMPLY_WITH', 
                                          'REQUIRES_DEVICE', 'PERMITTED_IN']):
            # Check if code_ref is in RETURN
            if 'code_ref' not in cypher and '-[r' in cypher:
                # Add code_ref to RETURN clause
                if 'RETURN ' in cypher:
                    # Find the RETURN clause and add r.code_ref
                    cypher = cypher.replace('RETURN ', 'RETURN r.code_ref as section, ')
        
        # Execute Cypher
        result = execute_cypher(cypher)
        
        if result["success"]:
            # Format results with emphasis on section references
            if result["results"]:
                result_count = len(result['results'])
                
                # Format results to highlight section numbers
                formatted_results = str(result['results'])
                
                # Add emphasis message for section references
                section_note = ""
                if 'code_ref' in formatted_results or 'section' in formatted_results:
                    section_note = "\nIMPORTANT: Results include code section references. ALWAYS cite these in your final answer!"
                
                return f"Query: {cypher}\n\nResults:\n{formatted_results}{section_note}"
            else:
                
                # Check if this might be a section from another code (900 series)
                section_match = re.search(r'\b([89]\d{2}(?:\.\d+)*)\b', query)
                helpful_note = ""
                if section_match:
                    section_num = section_match.group(1)
                    if section_num.startswith('9'):
                        helpful_note = f"\n\nNOTE: Section {section_num} is not in the HVAC code database. This section number is likely from the International Fire Code (IFC) or International Building Code (IBC), which are referenced by the HVAC code but not included in this database."
                    elif section_num.startswith('8'):
                        # Check if it's actually in the database or a cross-reference
                        helpful_note = f"\n\nNOTE: If Section {section_num} was not found, it may be a cross-reference to the International Fire Code (IFC) or International Building Code (IBC)."
                
                return f"Query executed successfully but returned no results.\nCypher: {cypher}{helpful_note}"
        else:
            return f"Cypher execution error: {result['error']}\nGenerated query: {cypher}"
    
    except Exception as e:
        return f"Cypher query error: {str(e)}"

# ==================== TOOL 3: Hybrid Search ====================

def hybrid_search(query: str) -> str:
    """
    Combined vector + graph search (vector semantic search + Cypher structured query)
    
    Best for:
    - Complex multi-part questions needing both domain relationships AND detailed text
    - Questions like "Why can't I install X in Y?" (needs prohibition relationship + explanation text)
    - Comprehensive equipment installation guides (needs all relationships + procedures)
    
    Strategy: First gets structured relationships via Cypher, then enriches with semantic text via Vector.
    """
    try:
        
        # Get vector results
        vector_results = vector_search(query)
        
        # Get Cypher results
        cypher_results = cypher_query(query)
        
        # Combine
        return f"=== SEMANTIC SEARCH ===\n{vector_results}\n\n=== GRAPH QUERY ===\n{cypher_results}"
    
    except Exception as e:
        return f"Hybrid search error: {str(e)}"

# ==================== TOOL DEFINITIONS ====================

tools = [
    Tool(
        name="CypherQuery",
        func=cypher_query,
        description=(
            "PRIMARY TOOL - Use this FIRST for all HVAC code questions. "
            "Structured graph queries about domain relationships, sections, equipment, and code requirements. "
            "BEST for: 'where can't I install X', 'what clearances does X need', "
            "'what standards does X comply with', 'what safety devices does X require', "
            "'what is Section 306', 'list all equipment', 'subsections of Chapter 3', "
            "'table data', 'piping requirements', 'gauge specifications'. "
            "Returns results with section references (code_ref). "
            "IMPORTANT: Pass the NATURAL LANGUAGE QUESTION as input, NOT Cypher code! "
            "Example: pass 'Where are furnaces prohibited?' not 'MATCH...RETURN'. "
            "If this returns no results, then try VectorSearch as fallback."
        )
    ),
    Tool(
        name="VectorSearch",
        func=vector_search,
        description=(
            "FALLBACK TOOL - Use ONLY when CypherQuery returns no results or for general explanations. "
            "Semantic search for unstructured content, procedures, and detailed explanations. "
            "Good for: 'how to install X', 'venting procedures', 'general HVAC concepts', "
            "'explain combustion air requirements'. "
            "STRATEGY: Always try CypherQuery first. Use VectorSearch as fallback."
        )
    ),
    Tool(
        name="HybridSearch",
        func=hybrid_search,
        description=(
            "Use this when you need both domain relationships AND detailed explanations. "
            "Good for: 'WHY can't I install X in Y' (needs prohibition + explanation text), "
            "'comprehensive installation guide for X' (needs all relationships + procedures)."
        )
    )
]

# ==================== AGENT PROMPT ====================

AGENT_PROMPT = """You are an HVAC Codes assistant with access to a Neo4j knowledge graph containing Phase 2A Domain Relationships.

DATABASE SCHEMA:
- Equipment nodes: Air Handler, Furnace, Boiler, Water Heater, etc. (34 types)
- Location nodes: Mechanical Room, Attic, Basement, Bedroom, etc. (27 types)
- Material nodes: Combustible Material, Steel, Concrete, Wood, etc. (21 types)
- Standard nodes: UL, NFPA, ASHRAE, ASME, etc. (16 types)
- SafetyDevice nodes: Relief Valve, Fire Damper, Disconnect Switch, etc. (12 types)
- Table nodes: TABLE_305.4, TABLE_803.9, etc. (tables from code document)
- Section nodes: 301, 303.3, 803.10.4, etc. (code sections with text)

DOMAIN RELATIONSHIPS (all include section references):
- PROHIBITED_IN: Where equipment CANNOT be installed
- REQUIRES_CLEARANCE: Minimum clearance distances for equipment
- MUST_COMPLY_WITH: Standards equipment must comply with
- REQUIRES_DEVICE: Safety devices required for equipment
- PERMITTED_IN: Where equipment CAN be installed
- CONTAINS_TABLE: Sections that contain tables
- HAS_ROW: Tables that have data rows

CRITICAL TOOL SELECTION STRATEGY - CYPHER-FIRST APPROACH:

1. PRIMARY: ALWAYS try CypherQuery FIRST
   - CypherQuery can access ALL data: sections, tables, relationships, equipment
   - It's faster and more accurate for structured queries
   
2. FALLBACK: Use VectorSearch ONLY when:
   - CypherQuery returns NO RESULTS
   - User asks for general procedures/explanations
   - Need to search unstructured text

3. NEVER skip CypherQuery for domain questions!

CRITICAL RULES FOR FINAL ANSWERS:
1. MANDATORY: ALWAYS cite code section numbers in your final answer
   - Format: "per Section 303.3", "Section 606.1 requires", "according to Section 1004"
2. MANDATORY: When tool results include 'code_ref', 'section', or 'r.code_ref', YOU MUST cite ALL section numbers
   - Example: If results show section "601", your answer MUST say "Section 601"
3. MANDATORY: Format multi-part answers with section citations for EACH part
   - Example: "Furnaces are prohibited in Bathrooms (Section 601), Bedrooms (Section 303.3), Closets (Section 601)"
4. FORBIDDEN: NEVER give answers without section references when the tool provides them
   - Bad: "Furnaces are prohibited in bedrooms" 
   - Good: "Furnaces are prohibited in bedrooms per Section 303.3"
5. MANDATORY: Use precise terminology from the code:
   - Use "prohibited" when equipment is not allowed (not just "cannot install")
   - Use "permitted" when equipment is allowed (not just "can install")
   - Use "combustion air" when discussing fuel-burning requirements
   - Use "ventilation" when discussing air circulation requirements
6. MANDATORY: When Section queries return 'summary' or 'text', include them in your answer
   - Example: "Section 303 covers Equipment and Appliance Location. It establishes general requirements..."
7. ALWAYS use tools to answer questions - never answer without using at least one tool
8. When calling CypherQuery tool, pass NATURAL LANGUAGE questions, NOT Cypher code!
   - CORRECT: Action Input: "Where are furnaces prohibited?"
   - WRONG: Action Input: "MATCH (e:Equipment)..."
9. POST-CHECK: Before giving Final Answer, verify you included:
   - Section citations from tool results
   - Key terminology (prohibited/permitted, combustion air, ventilation)
   - Section summaries if returned by query

Choose the right tool based on the question type:

USE CypherQuery FIRST for:
  - Equipment prohibitions: "Where can't I install X?"
  - Equipment clearances: "What clearances does X need?"
  - Safety devices: "What safety devices does X require?"
  - Permitted locations: "Where can I install X?"
  - Section lookups: "What is Section 306?"
  - Table data: "What gauge for 10-inch ducts?", "Show me TABLE 305.4"
  - Equipment lists: "List all equipment"
  - ANY structured domain question
  
  If CypherQuery returns no results, THEN try VectorSearch as fallback

USE VectorSearch ONLY when:
  - CypherQuery returned no results
  - General procedures: "How to install a furnace?"
  - Detailed explanations: "What are venting requirements?"
  - Exploratory questions: "Tell me about HVAC systems"
  
USE HybridSearch for:
  - Complex questions needing BOTH structure AND text
  - "Why can't I install X in Y?" (prohibition + explanation)

USE HybridSearch for:
  - Standards/compliance questions: "What standards must X comply with?" (needs specific standards + general requirements)
  - Complex multi-part questions needing BOTH relationships AND text
  - "WHY can't I install X in Y?" (needs prohibition + explanation)
  - Comprehensive requirements: "What are all requirements for X?"

You have access to these tools:
{tools}

Tool Names: {tool_names}

Use this format:

Question: the input question you must answer
Thought: think about which tool to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

REMEMBER: Always use tools! Don't answer without them.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

# Create prompt template
agent_prompt = PromptTemplate.from_template(AGENT_PROMPT)

# Create agent
agent = create_react_agent(llm, tools, agent_prompt)

# Create executor with verbose mode
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Shows tool calls in terminal
    handle_parsing_errors=True,
    max_iterations=25,  # Increased from 5 to handle complex aggregate queries
    max_execution_time=120  # Increased from 30 to allow thorough graph traversal
)

# ==================== MAIN QUERY FUNCTION ====================

def query_agent(question: str) -> str:
    """
    Main function to query the HVAC agent
    
    Args:
        question: Natural language question from user
    
    Returns:
        Agent's answer as string
    """
    try:
        # Show extracted entities for debugging
        entities = extract_entities(question)
        print(f"\n🔍 Extracted Entities: {entities}")
        
        # Invoke agent
        response = executor.invoke({"input": question})
        
        return response["output"]
    
    except Exception as e:
        return f"Error: {str(e)}"

# ==================== HELPER FUNCTIONS ====================

def get_statistics():
    """Get database statistics"""
    stats = {
        "Chapters": graph.query("MATCH (c:Chapter) RETURN count(c) as count")[0]["count"],
        "Sections": graph.query("MATCH (s:Section) RETURN count(s) as count")[0]["count"],
        "Equipment": graph.query("MATCH (e:Equipment) RETURN count(e) as count")[0]["count"],
        "Topics": graph.query("MATCH (t:Topic) RETURN count(t) as count")[0]["count"],
        "TextChunks": graph.query("MATCH (tc:TextChunk) RETURN count(tc) as count")[0]["count"]
    }
    return stats

# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print("="*60)
    print("HVAC GraphRAG Agent - Interactive Mode")
    print("="*60)
    print("\nType 'exit' to quit\n")
    
    while True:
        question = input("You: ")
        if question.lower() in ['exit', 'quit', 'q']:
            break
        
        answer = query_agent(question)
        print(f"\nAssistant: {answer}\n")
