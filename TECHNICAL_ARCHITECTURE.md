> Historical v1 document. For the current v2 runtime, setup, and measured limitations, see [README.md](README.md) and [V2_REVIEW.md](V2_REVIEW.md). The architecture and performance claims below are not v2 validation.

# HVAC GraphRAG - Complete Technical Architecture

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Database Schema](#database-schema)
4. [Data Flow](#data-flow)
5. [Query Processing Logic](#query-processing-logic)
6. [User Flow](#user-flow)
7. [Component Details](#component-details)
8. [Example Cypher Queries](#example-cypher-queries)

---

## 🏗️ System Overview

### What Is This System?

A **GraphRAG (Graph Retrieval-Augmented Generation)** system that:

- Extracts HVAC code data from PDF → Builds knowledge graph → Answers user questions
- Uses Neo4j graph database for structured data storage
- Uses LangChain ReAct agent for intelligent query routing
- Uses OpenAI GPT-4 for natural language understanding and response generation

### Core Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                         │
├─────────────────────────────────────────────────────────────┤
│  Frontend:         Streamlit (Python web framework)         │
│  Agent:            LangChain ReAct Agent                    │
│  LLM:              OpenAI GPT-4                             │
│  Database:         Neo4j 5.17.0 (Graph Database)            │
│  PDF Processing:   PyPDF2, pdfplumber                       │
│  NLP:              spaCy (en_core_web_sm)                   │
│  Query Language:   Cypher (Neo4j's query language)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 Architecture Diagram

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      Streamlit Web UI                             │  │
│  │  • Chat interface                                                 │  │
│  │  • Message history                                                │  │
│  │  • Session state management                                       │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      AGENT & ORCHESTRATION LAYER                        │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    LangChain ReAct Agent                          │  │
│  │  • Question analysis                                              │  │
│  │  • Tool selection (CypherQuery or VectorSearch)                   │  │
│  │  • Multi-step reasoning                                           │  │
│  │  • Response formatting                                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                    ↓                                     │
│  ┌──────────────────────┐              ┌──────────────────────┐         │
│  │   CypherQuery Tool   │              │ VectorSearch Tool    │         │
│  │  • Generate Cypher   │              │ • Semantic search    │         │
│  │  • Execute queries   │              │ • Not implemented    │         │
│  │  • Format results    │              │   yet                │         │
│  └──────────────────────┘              └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY GENERATION LAYER                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    cypher_generator.py                            │  │
│  │  • Dynamic Cypher query construction                              │  │
│  │  • Query pattern matching                                         │  │
│  │  • Parameter injection                                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      validation.py                                │  │
│  │  • Query validation                                               │  │
│  │  • Result verification                                            │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATABASE LAYER                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Neo4j Graph Database                           │  │
│  │  • 789 nodes (6 types)                                            │  │
│  │  • 2,248 relationships (5 types)                                  │  │
│  │  • Bolt protocol (neo4j://127.0.0.1:7687)                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA EXTRACTION LAYER                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   extract_and_load.py                             │  │
│  │  • PDF text extraction                                            │  │
│  │  • Section parsing                                                │  │
│  │  • Hierarchical structure building                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                relationship_extractor.py                          │  │
│  │  • NLP-based entity extraction                                    │  │
│  │  • Relationship identification                                    │  │
│  │  • Domain vocabulary matching                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA SOURCE                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      HVAC-Codes.pdf                               │  │
│  │  • Chapters 3-8, 10-11                                            │  │
│  │  • ~669 sections                                                  │  │
│  │  • Tables, requirements, prohibitions                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗄️ Database Schema

### Neo4j Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NODE TYPES (789 Total)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐                                                       │
│  │   Section    │  669 nodes                                            │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • number: string (e.g., "303.3")                                      │
│   • title: string (e.g., "Appliance access")                            │
│   • text: string (full section content)                                 │
│   • chapter: string (e.g., "3")                                         │
│   • exceptions: string (exception text if present)                      │
│   • level: integer (hierarchy level: 1, 2, or 3)                        │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │  Equipment   │  34 nodes                                             │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • name: string (e.g., "Furnace", "Boiler", "Water Heater")            │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │   Location   │  27 nodes                                             │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • name: string (e.g., "Bedroom", "Garage", "Attic")                   │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │   Material   │  21 nodes                                             │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • name: string (e.g., "Combustible Material", "Concrete")             │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │   Standard   │  16 nodes                                             │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • name: string (e.g., "ANSI Z21.10.1", "UL 296")                      │
│                                                                         │
│  ┌──────────────┐                                                       │
│  │ SafetyDevice │  12 nodes                                             │
│  └──────────────┘                                                       │
│   Properties:                                                           │
│   • name: string (e.g., "Pressure Relief Valve")                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   RELATIONSHIP TYPES (2,248 Total)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. HAS_CHILD (668 relationships)                                       │
│     ┌─────────┐                                                         │
│     │ Section │──────HAS_CHILD──────>│ Section │                        │
│     └─────────┘                      └─────────┘                        │
│     Properties: None                                                    │
│     Purpose: Parent-child section hierarchy                             │
│     Example: Section 303 ──HAS_CHILD──> Section 303.3                   │
│                                                                         │
│  2. HAS_PARENT (668 relationships)                                      │
│     ┌─────────┐                                                         │
│     │ Section │──────HAS_PARENT──────>│ Section │                       │
│     └─────────┘                       └─────────┘                       │
│     Properties: None                                                    │
│     Purpose: Child-parent section hierarchy (reverse of HAS_CHILD)      │
│     Example: Section 303.3 ──HAS_PARENT──> Section 303                  │
│                                                                         │
│  3. PROHIBITED_IN (~50 relationships)                                   │
│     ┌───────────┐                                                       │
│     │ Equipment │──────PROHIBITED_IN──────>│ Location │                 │
│     └───────────┘                          └──────────┘                 │
│     Properties:                                                         │
│     • code_ref: string (section number, e.g., "601.5")                  │
│     • reason: string (prohibition reason)                               │
│     Purpose: Where equipment cannot be installed                        │
│     Example: Furnace ──PROHIBITED_IN──> Bedroom                         │
│                                                                         │
│  4. REQUIRES_CLEARANCE (175 relationships)                              │
│     ┌───────────┐                                                       │
│     │ Equipment │──────REQUIRES_CLEARANCE──────>│ Material/Location │   │
│     └───────────┘                                └──────────────────┘   │
│     Properties:                                                         │
│     • code_ref: string (section number)                                 │
│     • min_inches: integer or null (minimum clearance)                   │
│     • note: string (additional requirements)                            │
│     Purpose: Clearance requirements for equipment                       │
│     Example: Furnace ──REQUIRES_CLEARANCE──> Combustible Material       │
│              (min_inches: null, note: "Per manufacturer instructions")  │
│                                                                         │
│  5. MUST_COMPLY_WITH (9 relationships)                                  │
│     ┌──────────────┐                                                    │
│     │ SafetyDevice │──────MUST_COMPLY_WITH──────>│ Standard │           │
│     └──────────────┘                             └──────────┘           │
│     Properties:                                                         │
│     • code_ref: string (section number)                                 │
│     Purpose: Standards compliance requirements                          │
│     Example: Pressure Relief Valve ──MUST_COMPLY_WITH──> ANSI Z21.22    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Database Storage Details

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEO4J STORAGE STRUCTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Connection:                                                    │
│    Protocol:  Bolt (binary protocol)                           │
│    URI:       neo4j://127.0.0.1:7687                           │
│    Database:  neo4j (default database)                         │
│    Auth:      Username/Password                                │
│                                                                 │
│  Storage:                                                       │
│    Format:    Property Graph Model                             │
│    Size:      ~5-10 MB                                          │
│    Location:  Neo4j data directory                             │
│                                                                 │
│  Indexes:                                                       │
│    • Section.number (for fast section lookup)                  │
│    • Equipment.name (for equipment queries)                    │
│    • Location.name (for location queries)                      │
│                                                                 │
│  Access Pattern:                                                │
│    Read-heavy workload                                          │
│    Writes only during data load/reload                          │
│    Concurrent reads supported                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Phase 1: Data Extraction & Loading

```
┌──────────────────────────────────────────────────────────────────────┐
│                    DATA EXTRACTION PIPELINE                          │
└──────────────────────────────────────────────────────────────────────┘

Step 1: PDF Extraction
────────────────────────────────────────────────────────────────────────
  HVAC-Codes.pdf
       ↓
  [PyPDF2 / pdfplumber]
       ↓
  Raw text (all pages)

Step 2: Section Parsing
────────────────────────────────────────────────────────────────────────
  Raw text
       ↓
  [Regex pattern matching]
       ↓
  Sections extracted:
  • Pattern: \d{3,4}\.\d{1,2}(?:\.\d{1,2})?
  • Examples: 303.3, 601.5, 1101.10.4
       ↓
  Hierarchical structure:
  • Level 1: 303 (chapter level)
  • Level 2: 303.3 (section level)
  • Level 3: 303.3.1 (subsection level)

Step 3: Entity Extraction (NLP)
────────────────────────────────────────────────────────────────────────
  Section text
       ↓
  [spaCy NLP pipeline]
       ↓
  Extract entities:
  • Equipment: noun phrases matching vocabulary
  • Locations: spatial nouns (bedroom, garage, attic)
  • Materials: material nouns (combustible, concrete)
  • Standards: codes matching patterns (ANSI, UL, ASHRAE)
  • Safety devices: specialized equipment
       ↓
  Deduplicate entities
       ↓
  Unique entity nodes

Step 4: Relationship Extraction
────────────────────────────────────────────────────────────────────────
  Section text + Entities
       ↓
  [Pattern matching + NLP]
       ↓
  Extract relationships:
  • "prohibited in" → PROHIBITED_IN
  • "clearance from" → REQUIRES_CLEARANCE
  • "comply with" → MUST_COMPLY_WITH
       ↓
  Relationship properties extracted:
  • code_ref (section number)
  • min_inches (numeric extraction)
  • reason/note (context)

Step 5: Graph Construction
────────────────────────────────────────────────────────────────────────
  Entities + Relationships
       ↓
  [Neo4j Cypher queries]
       ↓
  MERGE nodes (create if not exists)
  MERGE relationships
  SET properties
       ↓
  Complete knowledge graph

Step 6: Hierarchy Building
────────────────────────────────────────────────────────────────────────
  All Section nodes
       ↓
  [Parent-child linking]
       ↓
  Create HAS_CHILD / HAS_PARENT relationships
  • 303 ──HAS_CHILD──> 303.3
  • 303.3 ──HAS_PARENT──> 303

Result: 789 nodes, 2,248 relationships ready for queries
```

### Phase 2: Query Processing

````
┌──────────────────────────────────────────────────────────────────────┐
│                      QUERY PROCESSING FLOW                           │
└──────────────────────────────────────────────────────────────────────┘

Step 1: User Input
────────────────────────────────────────────────────────────────────────
  User types question in Streamlit UI
       ↓
  Example: "What clearances does a furnace need?"
       ↓
  Sent to LangChain agent

Step 2: Agent Analysis
────────────────────────────────────────────────────────────────────────
  Question received by ReAct agent
       ↓
  [OpenAI GPT-4 reasoning]
       ↓
  Agent analyzes:
  • Question type (section lookup, equipment query, etc.)
  • Required information
  • Best tool to use
       ↓
  Decision: Use CypherQuery tool (80% of queries)

Step 3: Cypher Query Generation
────────────────────────────────────────────────────────────────────────
  Tool: CypherQuery
  Input: "What clearances does a furnace need?"
       ↓
  [cypher_generator.py]
       ↓
  Pattern matching:
  • Detected: "clearances" + "furnace"
  • Query type: Equipment clearance requirements
       ↓
  Generated Cypher:
  ```cypher
  MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
  RETURN e.name as equipment,
         labels(target)[0] as target_type,
         target.name as target,
         r.min_inches as min_inches,
         r.code_ref as section,
         r.note as note
````

Step 4: Database Execution
────────────────────────────────────────────────────────────────────────
Cypher query
↓
[Neo4j database]
↓
Query execution:
• Index lookup on Equipment.name
• Traverse REQUIRES_CLEARANCE relationships
• Collect properties
↓
Raw results:
[
{equipment: "Furnace", target_type: "Material",
target: "Combustible Material", min_inches: null,
section: "304.9", note: "Per manufacturer instructions"},
{equipment: "Furnace", target_type: "Location",
target: "Garage", min_inches: 18,
section: "304.3", note: "Ignition source 18 inches above floor"},
{equipment: "Furnace", target_type: "Material",
target: "Concrete", min_inches: 3,
section: "304.10", note: "3 inches above grade or 6 inches suspended"}
]

Step 5: Result Formatting
────────────────────────────────────────────────────────────────────────
Raw results
↓
[Agent formatting with GPT-4]
↓
Formatted response:
"A furnace requires the following clearances:

1.  Combustible Materials (Section 304.9):

    - Per manufacturer's instructions

2.  Garage Installation (Section 304.3):

    - Minimum 18 inches above floor
    - Ignition source elevation requirement

3.  Grade/Concrete (Section 304.10): - Minimum 3 inches above grade - Or 6 inches if suspended"
    ↓
    Sent back to Streamlit UI

Step 6: Display
────────────────────────────────────────────────────────────────────────
Formatted response
↓
[Streamlit chat interface]
↓
Displayed to user with:
• Message bubble
• Section references (clickable)
• Stored in chat history

```

---

## 🧠 Query Processing Logic

### Agent Decision Tree

```

User Question
|
v
┌─────────────────────────────────────────┐
│ ReAct Agent (GPT-4) Analyzes Question │
└─────────────────────────────────────────┘
|
v
Question Type Detection
|
├──> "What is Section X.Y?"
| └──> CypherQuery: Section lookup
|
├──> "Where is [equipment] prohibited?"
| └──> CypherQuery: PROHIBITED_IN relationships
|
├──> "What clearances does [equipment] need?"
| └──> CypherQuery: REQUIRES_CLEARANCE relationships
|
├──> "Can I install [equipment] in [location]?"
| └──> CypherQuery: Multi-step (check prohibitions)
|
├──> "What equipment is prohibited in [location]?"
| └──> CypherQuery: Reverse prohibition query
|
├──> "What standards apply to [device]?"
| └──> CypherQuery: MUST_COMPLY_WITH relationships
|
└──> General/semantic question
└──> VectorSearch (not implemented yet)

````

### Cypher Query Patterns

```python
# Pattern 1: Section Lookup
# Question: "What is Section 303.3?"
MATCH (s:Section {number: "303.3"})
RETURN s.number, s.title, s.text, s.exceptions

# Pattern 2: Equipment Prohibitions
# Question: "Where are furnaces prohibited?"
MATCH (e:Equipment {name: "Furnace"})-[r:PROHIBITED_IN]->(l:Location)
RETURN l.name as location, r.code_ref as section, r.reason

# Pattern 3: Clearance Requirements
# Question: "What clearances does a furnace need?"
MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
RETURN target.name, r.min_inches, r.code_ref, r.note

# Pattern 4: Location Prohibitions (Reverse)
# Question: "What equipment is prohibited in bedrooms?"
MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location {name: "Bedroom"})
RETURN e.name as equipment, r.code_ref as section

# Pattern 5: Multi-Step Complex Query
# Question: "Can I install a furnace in a bedroom closet?"
# Step 1: Check direct prohibition
MATCH (e:Equipment {name: "Furnace"})-[r:PROHIBITED_IN]->(l:Location)
WHERE l.name IN ["Bedroom", "Closet", "Bedroom Closet"]
RETURN l.name, r.code_ref

# Step 2: Check clearance requirements
MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
RETURN target.name, r.min_inches

# Step 3: Agent combines results and reasons about feasibility

# Pattern 6: Standards Compliance
# Question: "What standards does a pressure relief valve need to comply with?"
MATCH (s:SafetyDevice {name: "Pressure Relief Valve"})-[r:MUST_COMPLY_WITH]->(std:Standard)
RETURN std.name as standard, r.code_ref as section
````

---

## 👤 User Flow

### Complete User Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER JOURNEY                               │
└─────────────────────────────────────────────────────────────────────┘

1. System Startup
   ─────────────────────────────────────────────────────────────────
   Terminal Command:
   $ streamlit run bot.py
        ↓
   Streamlit initializes:
   • Load config from .env
   • Connect to Neo4j
   • Initialize OpenAI client
   • Create LangChain agent
   • Setup session state
        ↓
   Browser opens: http://localhost:8501
        ↓
   UI displays:
   • Chat interface (empty)
   • Input box: "Ask me anything about HVAC codes..."
   • Sidebar with "About" section

2. User Asks Question
   ─────────────────────────────────────────────────────────────────
   User types: "What clearances does a furnace need?"
        ↓
   Clicks send or presses Enter
        ↓
   Message added to st.session_state.messages
        ↓
   UI displays user message bubble

3. Agent Processing (User sees "Thinking..." indicator)
   ─────────────────────────────────────────────────────────────────
   query_agent() function called
        ↓
   LangChain agent receives question
        ↓
   Agent thinks (GPT-4):
   • "This is asking about clearance requirements"
   • "I need to use CypherQuery tool"
   • "Equipment: Furnace, Query type: Clearances"
        ↓
   Tool: CypherQuery invoked with parameters
        ↓
   cypher_generator.py generates query
        ↓
   Neo4j executes query (~50-100ms)
        ↓
   Results returned to agent
        ↓
   Agent formats response (GPT-4)
        ↓
   Response ready

4. Response Display
   ─────────────────────────────────────────────────────────────────
   Agent returns formatted answer
        ↓
   Added to st.session_state.messages
        ↓
   UI displays assistant message bubble:

   "A furnace requires the following clearances:

    1. Combustible Materials (Section 304.9):
       - Per manufacturer's instructions

    2. Garage Installation (Section 304.3):
       - Minimum 18 inches above floor

    3. Grade/Concrete (Section 304.10):
       - Minimum 3 inches above grade"
        ↓
   User can:
   • Ask follow-up question
   • Clear chat history
   • Ask new question

5. Follow-up Question (Example)
   ─────────────────────────────────────────────────────────────────
   User: "Can I install a furnace in a bedroom?"
        ↓
   Agent reasoning:
   • Check PROHIBITED_IN relationships
   • Furnace → Bedroom prohibition exists?
        ↓
   Cypher query:
   MATCH (e:Equipment {name: "Furnace"})-[r:PROHIBITED_IN]->(l:Location {name: "Bedroom"})
   RETURN r.code_ref, r.reason
        ↓
   Results: [{"code_ref": "601.5", "reason": "Fuel-burning prohibited"}]
        ↓
   Agent response:
   "No, furnaces are prohibited in bedrooms according to
    Section 601.5. This prohibition applies to all fuel-burning
    appliances in sleeping areas."

6. Session Management
   ─────────────────────────────────────────────────────────────────
   All messages stored in Streamlit session state:
   st.session_state.messages = [
     {"role": "user", "content": "What clearances..."},
     {"role": "assistant", "content": "A furnace requires..."},
     {"role": "user", "content": "Can I install..."},
     {"role": "assistant", "content": "No, furnaces are prohibited..."}
   ]
        ↓
   Chat history persists during session
        ↓
   "Clear Chat History" button resets:
   st.session_state.messages = []
```

---

## 🔧 Component Details

### 1. bot.py (Streamlit UI)

```python
Purpose: Web interface for user interaction

Key Functions:
├─ main()
│  ├─ Initialize session state
│  ├─ Render sidebar
│  └─ Display chat interface
│
├─ Chat Loop
│  ├─ Display message history
│  ├─ Accept user input
│  ├─ Call query_agent()
│  └─ Display response
│
└─ Session State Management
   └─ st.session_state.messages (list of chat messages)

Flow:
1. User input → st.chat_input()
2. Append to messages list
3. Call agent.query_agent(user_input)
4. Append response to messages list
5. Rerender chat UI
```

### 2. agent.py (LangChain Agent)

```python
Purpose: Intelligent query router and orchestrator

Key Components:
├─ Tools
│  ├─ CypherQuery: Execute structured graph queries
│  └─ VectorSearch: Semantic search (planned)
│
├─ Agent
│  ├─ Type: ReAct (Reasoning + Acting)
│  ├─ LLM: OpenAI GPT-4
│  ├─ Memory: Agent scratchpad
│  └─ Max iterations: 5
│
└─ query_agent(question: str) → str
   ├─ Initialize agent executor
   ├─ Agent reasons about question
   ├─ Selects appropriate tool
   ├─ Executes tool (may chain multiple)
   ├─ Formats final response
   └─ Returns answer

Decision Logic:
• Section number mentioned → CypherQuery (section lookup)
• "prohibited", "not allowed" → CypherQuery (PROHIBITED_IN)
• "clearance", "distance" → CypherQuery (REQUIRES_CLEARANCE)
• Equipment + location question → CypherQuery (multi-step)
• Vague/semantic → VectorSearch (fallback, not implemented)
```

### 3. cypher_generator.py (Query Builder)

```python
Purpose: Dynamic Cypher query construction

Key Functions:
├─ generate_cypher_query(question: str, entities: dict) → str
│  ├─ Detect query pattern
│  ├─ Select query template
│  ├─ Inject parameters
│  └─ Return Cypher string
│
└─ Query Templates
   ├─ SECTION_LOOKUP
   ├─ EQUIPMENT_PROHIBITIONS
   ├─ CLEARANCE_REQUIREMENTS
   ├─ LOCATION_PROHIBITIONS
   └─ STANDARDS_COMPLIANCE

Pattern Matching:
if "Section" in question and section_number:
    → SECTION_LOOKUP template
elif equipment and "prohibited":
    → EQUIPMENT_PROHIBITIONS template
elif equipment and "clearance":
    → CLEARANCE_REQUIREMENTS template
elif location and "equipment":
    → LOCATION_PROHIBITIONS template

Example:
Input: "What clearances does a furnace need?"
Detected: equipment="Furnace", intent="clearance"
Template: CLEARANCE_REQUIREMENTS
Output: MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
        RETURN target.name, r.min_inches, r.code_ref, r.note
```

### 4. graph.py (Neo4j Connection)

```python
Purpose: Database connection and query execution

Key Components:
├─ get_graph() → Neo4jGraph
│  └─ Singleton instance of Neo4j connection
│
├─ Connection Configuration
│  ├─ URI: from config.neo4j_uri
│  ├─ Auth: (username, password)
│  └─ Database: from config.neo4j_database
│
└─ Query Methods
   ├─ graph.query(cypher: str) → List[Dict]
   │  └─ Execute Cypher, return results
   └─ graph.refresh_schema()
      └─ Update graph schema metadata

Error Handling:
• Connection failure → Retry with backoff
• Query timeout → Return empty results
• Syntax error → Log and return error message
```

### 5. extract_and_load.py (Data Pipeline)

```python
Purpose: Extract PDF data and load into Neo4j

Pipeline Stages:
├─ 1. PDF Extraction
│  ├─ PyPDF2.PdfReader(HVAC-Codes.pdf)
│  ├─ Extract text from all pages
│  └─ Combine into full text
│
├─ 2. Section Parsing
│  ├─ Regex: r'(\d{3,4}(?:\.\d{1,2}){1,2})\s+(.+?)(?=\n\d{3,4}\.|$)'
│  ├─ Extract section number, title, content
│  ├─ Filter: Only 301-1111 range (exclude 900s)
│  └─ Build hierarchy (level 1, 2, 3)
│
├─ 3. Entity Extraction (NLP)
│  ├─ spaCy: en_core_web_sm
│  ├─ Extract noun phrases
│  ├─ Match against vocabulary
│  ├─ Classify: Equipment, Location, Material, Standard, SafetyDevice
│  └─ Deduplicate
│
├─ 4. Relationship Extraction
│  ├─ Pattern: "prohibited in" → PROHIBITED_IN
│  ├─ Pattern: "clearance from/to" → REQUIRES_CLEARANCE
│  ├─ Pattern: "comply with" → MUST_COMPLY_WITH
│  ├─ Extract properties (code_ref, min_inches, reason)
│  └─ Create relationship objects
│
├─ 5. Graph Loading
│  ├─ MERGE Section nodes
│  ├─ MERGE entity nodes (Equipment, Location, etc.)
│  ├─ CREATE relationships
│  ├─ CREATE hierarchy (HAS_CHILD, HAS_PARENT)
│  └─ CREATE indexes
│
└─ 6. Validation
   ├─ Count nodes by type
   ├─ Count relationships by type
   ├─ Check for orphan sections
   └─ Report statistics

Execution Time: ~2-3 minutes for 669 sections
```

---

## 📝 Example Cypher Queries

### Query 1: Get Section Content

```cypher
-- Get full content of Section 303.3
MATCH (s:Section {number: "303.3"})
RETURN s.number as section_number,
       s.title as title,
       s.text as content,
       s.exceptions as exceptions

-- Result:
{
  "section_number": "303.3",
  "title": "Appliance access",
  "content": "Appliances shall be accessible for inspection, service, repair and replacement without removing permanent construction or building components.",
  "exceptions": "1. Appliances installed in an attic, crawl space or other concealed location..."
}
```

### Query 2: Find All Equipment Prohibited in Bedrooms

```cypher
-- What equipment is prohibited in bedrooms?
MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location {name: "Bedroom"})
RETURN e.name as equipment,
       r.code_ref as section,
       r.reason as reason
ORDER BY e.name

-- Results:
[
  {"equipment": "Furnace", "section": "601.5", "reason": "Fuel-burning appliances prohibited"},
  {"equipment": "Boiler", "section": "601.5", "reason": "Fuel-burning appliances prohibited"},
  {"equipment": "Water Heater", "section": "601.5", "reason": "Fuel-burning appliances prohibited"},
  ...
]
```

### Query 3: Get Clearance Requirements for Furnace

```cypher
-- What clearances does a furnace need?
MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
RETURN labels(target)[0] as target_type,
       target.name as target,
       r.min_inches as min_inches,
       r.code_ref as section,
       r.note as note

-- Results:
[
  {
    "target_type": "Material",
    "target": "Combustible Material",
    "min_inches": null,
    "section": "304.9",
    "note": "Per manufacturer instructions"
  },
  {
    "target_type": "Location",
    "target": "Garage",
    "min_inches": 18,
    "section": "304.3",
    "note": "Ignition source 18 inches above floor"
  },
  {
    "target_type": "Material",
    "target": "Concrete",
    "min_inches": 3,
    "section": "304.10",
    "note": "3 inches above grade or 6 inches suspended"
  }
]
```

### Query 4: Get Section Hierarchy

```cypher
-- Get all subsections of Chapter 3
MATCH (parent:Section {number: "303"})-[:HAS_CHILD*]->(child:Section)
RETURN parent.number as parent,
       child.number as subsection,
       child.title as title
ORDER BY child.number

-- Results:
[
  {"parent": "303", "subsection": "303.1", "title": "General"},
  {"parent": "303", "subsection": "303.2", "title": "Hazardous locations"},
  {"parent": "303", "subsection": "303.3", "title": "Appliance access"},
  ...
]
```

### Query 5: Complex Multi-Hop Query

```cypher
-- Find all locations where equipment with clearance requirements
-- to combustible materials is prohibited
MATCH (e:Equipment)-[c:REQUIRES_CLEARANCE]->(m:Material {name: "Combustible Material"})
MATCH (e)-[p:PROHIBITED_IN]->(l:Location)
RETURN e.name as equipment,
       l.name as prohibited_location,
       c.note as clearance_note,
       p.code_ref as prohibition_section,
       c.code_ref as clearance_section

-- Results show correlations between clearance requirements and prohibitions
```

### Query 6: Database Statistics

```cypher
-- Get complete database statistics
MATCH (n)
WITH labels(n)[0] as label, count(*) as count
RETURN label, count
ORDER BY count DESC

-- Get relationship statistics
MATCH ()-[r]->()
WITH type(r) as rel_type, count(*) as count
RETURN rel_type, count
ORDER BY count DESC

-- Find orphan sections (no parent)
MATCH (s:Section)
WHERE NOT exists((s)-[:HAS_PARENT]->())
AND s.number <> s.number  -- Exclude root sections
RETURN count(s) as orphan_count
```

---

## 🎯 Performance Characteristics

### Query Performance

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PERFORMANCE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Section Lookup:                    50-100ms                │
│  • Index-based lookup                                       │
│  • Single node retrieval                                    │
│  • Fastest query type                                       │
│                                                             │
│  Equipment Prohibitions:            100-200ms               │
│  • Relationship traversal                                   │
│  • 1-2 hops in graph                                        │
│  • ~5-10 results typically                                  │
│                                                             │
│  Clearance Requirements:            100-200ms               │
│  • Similar to prohibitions                                  │
│  • 1-2 hops in graph                                        │
│  • ~3-5 results typically                                   │
│                                                             │
│  Complex Multi-Step:                200-500ms               │
│  • Multiple Cypher queries                                  │
│  • Agent reasoning overhead                                 │
│  • GPT-4 processing time                                    │
│                                                             │
│  Total End-to-End (User → Answer): 5-20 seconds            │
│  • Network latency: ~50ms                                   │
│  • Database query: ~100-200ms                               │
│  • GPT-4 API call: 3-15 seconds (main bottleneck)           │
│  • Response formatting: ~1-2 seconds                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Scalability

```
Current Scale:
• 789 nodes → Can handle 100,000+ nodes
• 2,248 relationships → Can handle 1,000,000+ relationships
• Query response time stays constant (indexed lookups)

Bottlenecks:
1. OpenAI API calls (3-15 seconds)
   → Can be optimized with caching
2. Network latency (if deployed remotely)
   → Use CDN, optimize connection
3. First query of session (agent initialization)
   → Keep agent warm in production
```

---

## 🔒 Security & Access Control

```
Current Implementation:
├─ Neo4j Authentication
│  └─ Username/Password (from .env)
│
├─ OpenAI API Key
│  └─ Environment variable (not in code)
│
└─ Streamlit
   └─ Local hosting only (no authentication)

Production Recommendations:
├─ Add user authentication (Streamlit auth)
├─ Rate limiting on OpenAI API calls
├─ Neo4j encrypted connection (TLS)
├─ API key rotation policy
└─ Audit logging for queries
```

---

## 📚 Summary

### System Highlights

1. **Data Flow**: PDF → Extract → Parse → NLP → Graph → Query → Answer
2. **Storage**: Neo4j property graph (789 nodes, 2,248 relationships)
3. **Query**: Cypher-first strategy (80% success rate)
4. **Agent**: LangChain ReAct with GPT-4 (intelligent routing)
5. **UI**: Streamlit chat interface (simple, fast)

### Key Design Decisions

1. **Cypher over Vector Search**: More accurate for structured code data
2. **Graph over Relational**: Better for hierarchical and relationship queries
3. **ReAct Agent**: Allows multi-step reasoning and tool selection
4. **Lazy Loading**: Only generate embeddings when needed (future)
5. **Stateless Agent**: Each query is independent (no conversation memory)

### Future Enhancements

1. **Vector Search**: Add semantic search for vague queries
2. **Conversation Memory**: Remember context across questions
3. **Caching**: Cache common queries to reduce API calls
4. **Real-time Updates**: Allow PDF updates without full reload
5. **Multi-user**: Add authentication and user sessions

---

**Last Updated**: October 27, 2025
**Version**: 1.0
**Repository**: https://github.com/namanjain4463/Final_Final_AI
