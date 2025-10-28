# HVAC GraphRAG Knowledge Assistant

**A Graph Retrieval-Augmented Generation (GraphRAG) system for querying HVAC building codes using Neo4j knowledge graph, LangChang ReAct agents, and OpenAI GPT-4.**

This system extracts HVAC code sections from PDF documents, builds a structured knowledge graph in Neo4j, and provides an intelligent chatbot interface powered by a 3-tool AI agent to answer questions about HVAC installation requirements, equipment prohibitions, clearance specifications, and code compliance with section-referenced answers.

---

## 🎯 What This System Does

### Core Capabilities

**1. Intelligent Query Processing**

- **AI-Powered Agent**: Uses LangChain ReAct agent with 3 specialized tools (CypherQuery, VectorSearch, HybridSearch)
- **Cypher-First Strategy**: Prioritizes structured graph queries for precise domain-specific answers
- **LLM-Based Cypher Generation**: Converts natural language to Neo4j Cypher queries using GPT-4
- **Section-Referenced Answers**: Every response includes code section citations (e.g., "per Section 303.3")

**2. Domain Relationship Queries**

- Equipment prohibition lookups: "Where are furnaces prohibited?"
- Clearance requirements: "What clearances does a boiler need?"
- Safety device requirements: "What safety devices does a water heater require?"
- Standards compliance: "What standards must equipment comply with?"
- Permitted locations: "Where can I install an air handler?"

**3. Code Section Navigation**

- Direct section lookups: "What is Section 303.3?"
- Hierarchical browsing: "What subsections are in Chapter 6?"
- Table data queries: "Show me TABLE 305.4"
- Equipment and location lists: "List all equipment types"

**4. Real-World Scenario Queries**

- Installation feasibility: "Can I install a furnace in a bedroom closet?"
- Multi-requirement questions: "What are all requirements for installing a boiler?"
- Comparative queries: "What's the difference between Section 303.3 and 303.8?"
- Location-based queries: "What equipment is prohibited in bedrooms?"

**5. Advanced NLP Processing**

- **Entity Recognition**: Uses spaCy to extract equipment, locations, materials, standards, safety devices
- **Fuzzy Vocabulary Matching**: Normalizes variations ("AC unit" → "Air Conditioner", "HWH" → "Water Heater")
- **Relationship Extraction**: Pattern-based extraction of PROHIBITED_IN, REQUIRES_CLEARANCE, MUST_COMPLY_WITH relationships
- **Two-Stage Querying**: Entity recognition → Relationship query for optimal accuracy

### System Architecture

```
User Question
     ↓
Streamlit Bot (validation, sanitization)
     ↓
LangChain ReAct Agent (tool selection)
     ↓
    ┌─────────────┬─────────────┬─────────────┐
    │ CypherQuery │VectorSearch │HybridSearch │
    │  (PRIMARY)  │ (FALLBACK)  │  (COMPLEX)  │
    └─────────────┴─────────────┴─────────────┘
           ↓              ↓              ↓
    ┌──────────────────────────────────────┐
    │     Neo4j Knowledge Graph            │
    │  974 Nodes | 1,328 Relationships     │
    │  10 Node Types | 9 Relationship Types│
    └──────────────────────────────────────┘
           ↓
  Section-Referenced Answer
```

### Supported Query Types

| Query Category             | Example Questions                                     |
| -------------------------- | ----------------------------------------------------- |
| **Section Lookups**        | "What is Section 303.3?", "Show Chapter 6"            |
| **Equipment Prohibitions** | "Where are furnaces prohibited?"                      |
| **Clearance Requirements** | "What clearances does a boiler need?"                 |
| **Safety Devices**         | "What safety devices are required for water heaters?" |
| **Permitted Locations**    | "Where can I install equipment?"                      |
| **Standards Compliance**   | "What standards must boilers comply with?"            |
| **Table Data**             | "Show TABLE 305.4", "What gauge for 10-inch ducts?"   |
| **Equipment Lists**        | "List all equipment", "What locations exist?"         |
| **Multi-Part Questions**   | "Can I install a furnace in a bedroom closet?"        |
| **Comparative Questions**  | "Compare requirements for furnaces vs boilers"        |

### Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **AI Agent**: LangChain ReAct Agent with 3 specialized tools
- **LLM**: OpenAI GPT-4 (Cypher generation + answer synthesis)
- **Graph Database**: Neo4j 5.17.0+
- **NLP**: spaCy (en_core_web_sm) for entity extraction
- **PDF Processing**: PyMuPDF (fitz), pdfplumber
- **Vector Embeddings**: OpenAI text-embedding-ada-002
- **Query Language**: Cypher (Neo4j)

---

## 📂 Project Structure

```
final_ai/
├── agent.py                  # LangChain ReAct agent (3 tools)
├── bot.py                    # Streamlit chatbot UI
├── config.py                 # Environment configuration
├── graph.py                  # Neo4j connection
├── llm.py                    # OpenAI LLM wrapper
├── extract_and_load.py       # ETL pipeline
├── relationship_extractor.py # NLP relationship extraction
├── table_extractor.py        # PDF table parser
├── vocabulary.py             # Domain vocabulary
├── cypher_generator.py       # Cypher query generator
├── validation.py             # Query validation
├── utils.py                  # Utility functions
├── HVAC-Codes.pdf           # Source data
├── requirements.txt         # Dependencies
├── .env.example             # Config template
└── README.md                # This file
```

---

## 🔧 Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Neo4j 5.17.0+** (Desktop or Aura Cloud)
- **OpenAI API Key** (with GPT-4 access)

### Step 1: Install Neo4j

**Option A: Neo4j Desktop (Recommended)**

1. Download from [neo4j.com/download](https://neo4j.com/download/)
2. Install and create a new database named "neo4j"
3. Set a password
4. Start the database (runs on `bolt://localhost:7687`)

**Option B: Neo4j Aura (Cloud)**

1. Sign up at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Save connection credentials

### Step 2: Clone Repository

```powershell
git clone https://github.com/namanjain4463/Final_Final_AI.git
cd Final_Final_AI
```

### Step 3: Install Python Dependencies

```powershell
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Download spaCy NLP model
python -m spacy download en_core_web_sm
```

### Step 4: Configure Environment Variables

```powershell
# Copy template
copy .env.example .env

# Edit with your credentials
notepad .env
```

**Required `.env` settings:**

```env
# OpenAI
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_DATABASE=neo4j

# Data
PDF_PATH=HVAC-Codes.pdf
```

---

## 🚀 Usage Instructions

### Load Data into Neo4j (First Time Setup)

This extracts data from the PDF and builds the knowledge graph:

```powershell
python extract_and_load.py
```

**What this does:**

- Extracts 667 sections from HVAC-Codes.pdf using enhanced pattern matching
- Extracts 20 tables with 182 data rows
- Creates 10 node types: Section, Equipment, Location, Material, Standard, SafetyDevice, Chapter, Document, Table, TableRow
- Extracts 9 relationship types using NLP: CONTAINS, PROHIBITED_IN, REQUIRES_CLEARANCE, MUST_COMPLY_WITH, REQUIRES_DEVICE, PERMITTED_IN, MADE_OF, CONTAINS_TABLE, HAS_ROW
- Generates vector embeddings for semantic search (optional)
- Takes ~5-10 minutes

**Expected output:**

```
[1/7] Extracting text from HVAC-Codes.pdf...
✓ Extracted 1,234,567 characters

[2/7] Extracting sections...
✓ Found 667 sections

[3/7] Creating Document node...
✓ Document node created

[4/7] Creating Chapter and Section nodes...
✓ Created 6 Chapter nodes
✓ Created 667 Section nodes

[5/7] Extracting domain entities and relationships...
✓ Created 34 Equipment nodes
✓ Created 25 Location nodes
✓ Created 19 Material nodes
✓ Created 6 Standard nodes
✓ Created 9 SafetyDevice nodes
✓ Created 652 CONTAINS relationships
✓ Created 83 PROHIBITED_IN relationships
✓ Created 167 REQUIRES_CLEARANCE relationships
✓ Created 27 PERMITTED_IN relationships
✓ Created 9 MUST_COMPLY_WITH relationships
✓ Created 54 REQUIRES_DEVICE relationships
✓ Created 136 MADE_OF relationships

[6/7] Extracting tables...
✓ Created 20 Table nodes
✓ Created 182 TableRow nodes
✓ Created 18 CONTAINS_TABLE relationships
✓ Created 182 HAS_ROW relationships

[7/7] Database population complete!
Total: 974 nodes, 1,328 relationships
```

### Run the Chatbot

```powershell
streamlit run bot.py
```

Opens browser at `http://localhost:8501`

### Example Queries & Expected Responses

#### 1. Section Lookup

**Query:** "What is Section 303.3?"

**Expected Response:**

```
Section 303.3 - Appliance Access

Appliances shall be accessible for inspection, service, repair and replacement
without removing permanent construction or building finishes.

*Reference: Section 303.3*
```

#### 2. Equipment Prohibition

**Query:** "Where are furnaces prohibited?"

**Expected Response:**

```
Furnaces are prohibited in the following locations per HVAC codes:

1. Bedrooms - prohibited per Section 303.3
2. Bathrooms - prohibited per Section 601.5
3. Sleeping Rooms - prohibited per Section 303.3
4. Closets within bedrooms - prohibited per Section 303.3

These prohibitions ensure safety and proper ventilation for fuel-burning equipment.
```

#### 3. Clearance Requirements

**Query:** "What clearances does a furnace need?"

**Expected Response:**

```
Furnace clearance requirements per HVAC codes:

From Combustible Materials:
- Minimum 18 inches clearance (Section 304.1)
- Or as specified by manufacturer instructions

From Garage Floor:
- Minimum 18 inches above floor (Section 305.4)

From Concrete:
- Minimum 3 inches above grade (Section 304.9)

*Note: Always verify manufacturer specifications which may require greater clearances.*
```

#### 4. Safety Device Requirements

**Query:** "What safety devices does a water heater require?"

**Expected Response:**

```
Water heaters must be equipped with the following safety devices per Section 1004:

1. Pressure Relief Valve - Required per Section 1004.1, must comply with ANSI Z21.22
2. Temperature Relief Valve - Required per Section 1004.2
3. Vacuum Relief Valve - Required for certain installations per Section 1004.3

These devices protect against over-pressure and over-temperature conditions.
```

#### 5. Installation Feasibility

**Query:** "Can I install a furnace in a bedroom closet?"

**Expected Response:**

```
No, furnaces cannot be installed in bedroom closets.

Per Section 303.3, fuel-burning appliances including furnaces are prohibited in:
- Bedrooms
- Bathrooms
- Closets within bedrooms

Reasoning: Fuel-burning equipment requires adequate combustion air and poses carbon monoxide
risks in sleeping areas.

Alternative locations (permitted per Section 303.1):
- Dedicated mechanical rooms
- Basements (with proper clearances)
- Garages (elevated 18" from floor per Section 305.4)
- Attics (with proper access per Section 303.3)
```

#### 6. Table Data Query

**Query:** "Show me TABLE 305.4"

**Expected Response:**

```
TABLE 305.4 - Minimum Clearances for Appliances in Garages

Equipment Type | Clearance from Floor
---------------|---------------------
Furnace        | 18 inches
Water Heater   | 18 inches
Boiler         | 18 inches

*Reference: Section 305.4 - Protection from Vehicle Impact*

This elevation protects equipment from vehicle impact and fuel vapor ignition risks.
```

### Query Best Practices

**✅ DO:**

- Ask specific questions about equipment, locations, or code sections
- Reference section numbers if you know them: "What is Section 303.3?"
- Ask about relationships: "Where are furnaces prohibited?"
- Request clearances, devices, or standards: "What clearances does X need?"

**⚠️ For Best Results:**

- Use specific equipment names: "furnace" instead of "heating unit"
- Be clear about what you're asking: "prohibited" vs "permitted" vs "clearances"
- One question at a time works better than compound questions

### Reset Database (If Needed)

If you need to reload data or fix issues:

```powershell
# Clear all nodes and relationships
python -c "from graph import graph; graph.query('MATCH (n) DETACH DELETE n'); print('Database cleared')"

# Reload data
python extract_and_load.py
```

---

## 💾 Data Description

### Source Dataset: HVAC-Codes.pdf

**Overview:**

- **Format**: PDF document
- **Size**: ~15-20 MB
- **Pages**: ~200 pages
- **Source**: International Mechanical Code (IMC) - HVAC chapters
- **Coverage**: Chapters 3-8, 10-11 (Chapter 9 is in separate International Fire Code)

**Content:**

| Chapter    | Sections  | Topics                                             |
| ---------- | --------- | -------------------------------------------------- |
| Chapter 3  | 301-399   | General Regulations (location, access, clearances) |
| Chapter 4  | 401-499   | Ventilation (natural/mechanical)                   |
| Chapter 5  | 501-599   | Exhaust Systems                                    |
| Chapter 6  | 601-699   | Duct Systems (construction, installation)          |
| Chapter 7  | 701-799   | Combustion Air                                     |
| Chapter 8  | 801-899   | Chimneys and Vents                                 |
| Chapter 10 | 1001-1099 | Boilers, Water Heaters                             |
| Chapter 11 | 1101-1199 | Refrigeration Systems                              |

**Extraction Statistics:**

- **Total Sections**: 669
- **Section Range**: 301.1 - 1111.1
- **Hierarchical Levels**: 3 (e.g., 303 → 303.3 → 303.3.1)
- **Average Section Length**: 200-500 characters
- **Tables Extracted**: ~15-20
- **Cross-References**: ~100+ to other codes

**Important Note:**  
Chapter 9 (Sections 901-999) is NOT included in this PDF - those sections are part of the International Fire Code (IFC).

---

## 🗄️ Neo4j Database Details

### Database Configuration

```
Protocol:  Bolt (neo4j://)
Host:      127.0.0.1
Port:      7687
Database:  neo4j
Storage:   5-10 MB
```

### Graph Structure

**Total:**

- **Nodes**: 974
- **Relationships**: 1,328
- **Indexes**: 3 (Section.number, Equipment.name, Location.name)

### Node Types (10 types)

#### 1. Section (667 nodes)

```cypher
(:Section {
  number: "303.3",              # Section identifier
  title: "Appliance access",    # Section title
  text: "...",                  # Full content (200-500 chars)
  chapter: "3",                 # Chapter number
  exceptions: "...",            # Exception text (if any)
  level: 2                      # Hierarchy level (1, 2, or 3)
})
```

#### 2. Equipment (34 nodes)

```cypher
(:Equipment {name: "Furnace"})
```

**Complete list:** Furnace, Boiler, Water Heater, Air Conditioner, Heat Pump, Evaporative Cooler, Duct Heater, Unit Heater, Infrared Heater, Cooking Appliance, Clothes Dryer, Pool Heater, Spa Heater, Refrigeration Unit, Fireplace, Stove, Range, Oven, Incinerator, Crematory, Kiln, Dryer, Generator, Compressor, Fan, Blower, Exhaust Fan, Ventilator, Air Handler, Condenser, Evaporator, Radiator, Baseboard Heater, Radiant Floor Heater

#### 3. Location (25 nodes)

```cypher
(:Location {name: "Bedroom"})
```

**Complete list:** Bedroom, Bathroom, Closet, Bedroom Closet, Garage, Attic, Crawl Space, Basement, Mechanical Room, Utility Room, Hallway, Kitchen, Living Room, Sleeping Room, Storage Room, Alcove, Under-floor Space, Roof, Exterior Wall, Interior Wall, Ceiling Space, Floor Space, Enclosed Space, Concealed Location, Outdoor Location

#### 4. Material (19 nodes)

```cypher
(:Material {name: "Combustible Material"})
```

**Complete list:** Combustible Material, Concrete, Wood, Metal, Plastic, Drywall, Masonry, Brick, Stone, Glass, Insulation, Fiberglass, Foam, Steel, Aluminum, Copper, Galvanized Metal, Stainless Steel, Cast Iron

#### 5. Standard (6 nodes)

```cypher
(:Standard {name: "ANSI Z21.10.1"})
```

**Complete list:** ANSI Z21.10.1, ANSI Z21.86, ANSI Z21.47, ANSI Z21.22, UL 296, UL 127

#### 6. SafetyDevice (9 nodes)

```cypher
(:SafetyDevice {name: "Pressure Relief Valve"})
```

**Complete list:** Pressure Relief Valve, Temperature Relief Valve, Vacuum Relief Valve, Backflow Preventer, Flame Rollout Switch, High Limit Switch, Low Water Cutoff, Combustion Air Proving Switch, Draft Hood

#### 7. Chapter (6 nodes)

```cypher
(:Chapter {
  number: "3",
  title: "General Regulations"
})
```

**Complete list:** Chapter 3, 4, 5, 6, 7, 8 (Note: Chapters 10-11 may be present but not shown in current view)

#### 8. Document (6 nodes)

```cypher
(:Document {
  title: "International Mechanical Code",
  source: "HVAC-Codes.pdf"
})
```

Represents the source documents from which sections were extracted.

#### 9. Table (20 nodes)

```cypher
(:Table {
  number: "Table 304.1",
  title: "Minimum Clearances"
})
```

Represents tables extracted from the PDF that contain structured data (clearances, specifications, etc.)

#### 10. TableRow (182 nodes)

```cypher
(:TableRow {
  equipment: "Furnace",
  location: "Combustible Wall",
  clearance: "18"
})
```

Individual rows from extracted tables containing specific requirements.

### Relationship Types (9 types)

#### 1. CONTAINS (652 relationships)

```cypher
(chapter:Chapter)-[:CONTAINS]->(section:Section)
(document:Document)-[:CONTAINS]->(chapter:Chapter)
```

Represents hierarchical document structure. Chapters contain Sections, Documents contain Chapters.

**Example:** Chapter 3 → CONTAINS → Section 303.3

#### 2. PROHIBITED_IN (83 relationships)

```cypher
(equipment:Equipment)-[:PROHIBITED_IN {
  code_ref: "303.3",
  reason: "Fuel-burning appliances prohibited in sleeping areas"
}]->(location:Location)
```

**Example:** Furnace → PROHIBITED_IN → Bedroom

#### 3. PERMITTED_IN (27 relationships)

```cypher
(equipment:Equipment)-[:PERMITTED_IN {
  code_ref: "303.1",
  conditions: "With adequate ventilation"
}]->(location:Location)
```

Indicates locations where equipment IS allowed to be installed (often with specific conditions).

**Example:** Furnace → PERMITTED_IN → Mechanical Room

#### 4. REQUIRES_CLEARANCE (167 relationships)

```cypher
(equipment:Equipment)-[:REQUIRES_CLEARANCE {
  code_ref: "304.1",
  min_inches: 18,
  note: "Per manufacturer instructions"
}]->(target:Material|Location)
```

**Examples:**

- Furnace → REQUIRES_CLEARANCE → Combustible Material (18 inches minimum)
- Furnace → REQUIRES_CLEARANCE → Garage (18 inches from floor)
- Furnace → REQUIRES_CLEARANCE → Concrete (3 inches above grade)

#### 5. MUST_COMPLY_WITH (9 relationships)

```cypher
(device:SafetyDevice)-[:MUST_COMPLY_WITH {
  code_ref: "1004.1"
}]->(standard:Standard)
```

**Example:** Pressure Relief Valve → MUST_COMPLY_WITH → ANSI Z21.22

#### 6. REQUIRES_DEVICE (54 relationships)

```cypher
(equipment:Equipment)-[:REQUIRES_DEVICE {
  code_ref: "1004.2",
  mandatory: true
}]->(device:SafetyDevice)
```

Indicates which safety devices are required for specific equipment.

**Example:** Water Heater → REQUIRES_DEVICE → Pressure Relief Valve

#### 7. MADE_OF (136 relationships)

```cypher
(equipment:Equipment)-[:MADE_OF {
  component: "Heat exchanger"
}]->(material:Material)
```

Indicates materials used in equipment construction or installation.

**Example:** Duct → MADE_OF → Galvanized Metal

#### 8. CONTAINS_TABLE (18 relationships)

```cypher
(section:Section)-[:CONTAINS_TABLE]->(table:Table)
```

Links sections to tables they reference.

**Example:** Section 304.1 → CONTAINS_TABLE → Table 304.1

#### 9. HAS_ROW (182 relationships)

```cypher
(table:Table)-[:HAS_ROW]->(row:TableRow)
```

Links tables to their individual data rows.

**Example:** Table 304.1 → HAS_ROW → TableRow (Furnace clearance from combustible wall)

---

## 🔍 How to Query Neo4j

### Access Methods

**1. Neo4j Browser (Web Interface)**

```
URL: http://localhost:7474
Login: neo4j / <your-password>
```

**2. Python (Using this project)**

```python
from graph import get_graph

graph = get_graph()
results = graph.query("MATCH (s:Section) RETURN s LIMIT 10")
print(results)
```

**3. Cypher Shell (Command Line)**

```bash
cypher-shell -u neo4j -p <your-password>
```

### Essential Cypher Queries

#### Database Exploration

```cypher
// Count nodes by type
MATCH (n)
RETURN labels(n)[0] as NodeType, count(*) as Count
ORDER BY Count DESC

// Count relationships by type
MATCH ()-[r]->()
RETURN type(r) as RelType, count(*) as Count
ORDER BY Count DESC

// View schema
CALL db.schema.visualization()
```

#### Section Queries

```cypher
// Get specific section
MATCH (s:Section {number: "303.3"})
RETURN s.number, s.title, s.text, s.exceptions

// Get all subsections of a chapter
MATCH (parent:Section {number: "303"})-[:HAS_CHILD*]->(child:Section)
RETURN child.number, child.title
ORDER BY child.number

// Search sections by keyword
MATCH (s:Section)
WHERE toLower(s.title) CONTAINS 'clearance'
RETURN s.number, s.title
ORDER BY s.number
```

#### Equipment & Location Queries

```cypher
// Find where equipment is prohibited
MATCH (e:Equipment {name: "Furnace"})-[r:PROHIBITED_IN]->(l:Location)
RETURN l.name as Location, r.code_ref as Section, r.reason

// Find equipment prohibited in a location
MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location {name: "Bedroom"})
RETURN e.name as Equipment, r.code_ref as Section
ORDER BY e.name

// Get clearance requirements for equipment
MATCH (e:Equipment {name: "Furnace"})-[r:REQUIRES_CLEARANCE]->(target)
RETURN labels(target)[0] as TargetType,
       target.name as Target,
       r.min_inches as MinInches,
       r.code_ref as Section,
       r.note as Note
```

#### Complex Queries

```cypher
// Get complete information for equipment
MATCH (e:Equipment {name: "Furnace"})
OPTIONAL MATCH (e)-[p:PROHIBITED_IN]->(l:Location)
OPTIONAL MATCH (e)-[c:REQUIRES_CLEARANCE]->(target)
RETURN e.name as Equipment,
       collect(DISTINCT {location: l.name, section: p.code_ref}) as Prohibitions,
       collect(DISTINCT {target: target.name, clearance: c.min_inches, section: c.code_ref}) as Clearances

// Find sections mentioning equipment
MATCH (s:Section)
WHERE toLower(s.text) CONTAINS 'furnace'
RETURN s.number, s.title
ORDER BY s.number
LIMIT 20
```

#### Database Maintenance

```cypher
// Delete all data
MATCH (n) DETACH DELETE n

// Create indexes (done automatically)
CREATE INDEX section_number IF NOT EXISTS FOR (s:Section) ON (s.number);
CREATE INDEX equipment_name IF NOT EXISTS FOR (e:Equipment) ON (e.name);
CREATE INDEX location_name IF NOT EXISTS FOR (l:Location) ON (l.name);
```

---

## 📊 Database Statistics

```
Node Distribution:
├── Section:      667 (68.5%)
├── TableRow:     182 (18.7%)
├── Equipment:     34 (3.5%)
├── Location:      25 (2.6%)
├── Table:         20 (2.1%)
├── Material:      19 (2.0%)
├── SafetyDevice:   9 (0.9%)
├── Chapter:        6 (0.6%)
├── Document:       6 (0.6%)
└── Standard:       6 (0.6%)
Total Nodes:      974

Relationship Distribution:
├── CONTAINS:           652 (49.1%)
├── HAS_ROW:            182 (13.7%)
├── REQUIRES_CLEARANCE: 167 (12.6%)
├── MADE_OF:            136 (10.2%)
├── PROHIBITED_IN:       83 (6.2%)
├── REQUIRES_DEVICE:     54 (4.1%)
├── PERMITTED_IN:        27 (2.0%)
├── CONTAINS_TABLE:      18 (1.4%)
└── MUST_COMPLY_WITH:     9 (0.7%)
Total Relationships: 1,328

Coverage:
├── Chapters:     6 extracted (3-8)
├── Sections:     667 total
├── Tables:       20 extracted
├── Table Rows:   182 data points
└── Avg Size:     ~300 characters per section
```

---

## 📚 Additional Documentation

- **[TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md)** - Complete system architecture, component details, data flow diagrams, and implementation guide
- **[QUERY_REFERENCE.md](QUERY_REFERENCE.md)** - Comprehensive query guide with 40+ examples, performance ratings, and best practices
- **[TECHNICAL_ROADMAP.md](TECHNICAL_ROADMAP.md)** - Development approach, AI techniques, and system workflow diagrams
- **[GRAPH_MODELING_ANALYSIS.md](GRAPH_MODELING_ANALYSIS.md)** - Graph data modeling analysis against Neo4j best practices

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
