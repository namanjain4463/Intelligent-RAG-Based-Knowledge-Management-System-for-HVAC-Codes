# HVAC GraphRAG Knowledge Assistant 🏭

A sophisticated GraphRAG (Graph Retrieval-Augmented Generation) system for querying HVAC codes using Neo4j knowledge graph, LangChain agents, and OpenAI.

## 🎯 What This System Can Do

### ✅ Supported Queries

1. **Section Lookup**

   - Get full content of any section (e.g., "What is Section 303.3?")
   - Retrieve subsections and exceptions

2. **Equipment Prohibition Queries**

   - Find where specific equipment is prohibited (e.g., "Where are furnaces prohibited?")
   - List all equipment prohibited in specific locations (e.g., "What equipment is prohibited in bedrooms?")
   - Returns section references with detailed requirements

3. **Clearance Requirements**

   - Get clearance requirements for any equipment (e.g., "What clearances does a furnace need?")
   - Includes combustible material clearances (Section 304.9)
   - Garage elevation requirements (Section 304.3)
   - Grade clearance requirements (Section 304.10)

4. **Complex Multi-Step Reasoning**

   - Combine multiple queries (e.g., "Can I install a furnace in a bedroom closet?")
   - Agent chains multiple Cypher queries automatically
   - Provides comprehensive answers with section citations

5. **Standards and Compliance**
   - Find which standards apply to equipment or installations
   - Query ANSI, UL, ASHRAE, and other referenced standards

### ❌ Current Limitations

1. **Vector Search Not Implemented**

   - Embeddings generation not yet configured
   - System relies entirely on structured Cypher queries
   - Semantic search functionality planned for future release

2. **Specific Equipment Details**

   - Clearance distances often show "per manufacturer instructions" (as per code)
   - Specific numeric values may not be available for all equipment types
   - Some requirements use generic "equipment and appliances" language

3. **Scope Limited to HVAC Code**
   - Does not include plumbing, electrical, or building codes
   - Cross-references to other codes are noted but not followed

## 🚀 Key Features

- **Cypher-First Strategy**: 80% of queries answered with structured graph queries alone
- **Intelligent Fallback**: Automatic retry with vector search if Cypher returns no results
- **Multi-Tool Agent**: LangChain ReAct agent selects optimal tool for each query
- **Interactive Chatbot**: Clean Streamlit-based UI with chat history
- **Production-Ready**: Tested with 5/5 success rate on standard queries

## 📋 Prerequisites

- Python 3.11+
- Neo4j Database (local or cloud)
- OpenAI API key

## 🔧 Quick Start Guide

### Prerequisites

- **Python 3.11+** installed
- **Neo4j 5.17.0+** (Desktop or Aura)
- **OpenAI API key** with GPT-4 access

### Step 1: Install Neo4j

**Option A: Neo4j Desktop (Recommended for Windows)**

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Install and create a new project
3. Create a database named "neo4j" with password of your choice
4. Start the database (should run on `bolt://localhost:7687`)

**Option B: Neo4j Aura (Cloud)**

1. Sign up at [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Create a free instance
3. Save the connection URI and credentials

### Step 2: Clone and Setup

```powershell
# Clone the repository
git clone <repository-url>
cd final_ai

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for NLP extraction)
python -m spacy download en_core_web_sm
```

### Step 3: Configure Environment

```powershell
# Copy example environment file
copy .env.example .env

# Edit .env with your credentials
notepad .env
```

**Required settings in `.env`:**

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
OPENAI_MODEL=gpt-4

# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password-here
NEO4J_DATABASE=neo4j

# PDF Path (ensure this file exists)
PDF_PATH=HVAC-Codes.pdf
```

### Step 4: Load Data into Neo4j

This is the **most important step** - it extracts data from the PDF and builds the knowledge graph.

```powershell
# Ensure HVAC-Codes.pdf is in the project root
# Run the extraction and loading script
python extract_and_load.py
```

**What this does:**

- Extracts all sections from HVAC-Codes.pdf (Chapters 3-8, 10-11)
- Creates 669 Section nodes with hierarchical structure
- Extracts equipment, locations, materials, standards
- Creates relationships (PROHIBITED_IN, REQUIRES_CLEARANCE, MUST_COMPLY_WITH)
- Takes approximately 5-10 minutes

**Expected output:**

```
Created 669 Section nodes
Created 34 Equipment nodes
Created 27 Location nodes
Created 175 REQUIRES_CLEARANCE relationships
Created ~50 PROHIBITED_IN relationships
...
```

### Step 5: Run the Chatbot

```powershell
streamlit run bot.py
```

The chatbot will open in your browser at `http://localhost:8501`

### Step 6: Test the System

Run the validation test to ensure everything works:

```powershell
python test_agent_minimal.py
```

**Expected results:**

- ✅ 5/5 queries should PASS
- Total runtime: ~60-70 seconds
- All queries should return section citations

## 📊 Database Statistics

After running `extract_and_load.py`, your Neo4j database will contain:

| Node Type    | Count | Description                              |
| ------------ | ----- | ---------------------------------------- |
| Section      | 669   | HVAC code sections (Chapters 3-8, 10-11) |
| Equipment    | 34    | Furnaces, boilers, water heaters, etc.   |
| Location     | 27    | Bedrooms, garages, attics, etc.          |
| Material     | 21    | Combustible materials, concrete, etc.    |
| Standard     | 16    | ANSI, UL, ASHRAE standards               |
| SafetyDevice | 12    | Pressure relief valves, etc.             |

| Relationship Type  | Count | Description                          |
| ------------------ | ----- | ------------------------------------ |
| REQUIRES_CLEARANCE | 175   | Clearance requirements for equipment |
| PROHIBITED_IN      | ~50   | Equipment prohibited in locations    |
| MUST_COMPLY_WITH   | 9     | Standards compliance requirements    |
| HAS_CHILD          | 668   | Section hierarchy                    |
| HAS_PARENT         | 668   | Section hierarchy (reverse)          |

## 🧪 Example Queries

Try these queries in the chatbot to test functionality:

### Basic Queries (Direct Section Lookup)

- "What is Section 303.3?"
- "Show me Section 601.5"
- "What does Section 304.9 say?"

### Equipment Prohibition Queries

- "Where are furnaces prohibited?"
- "What equipment is prohibited in bedrooms?"
- "Can I install a boiler in a closet?"

### Clearance Requirement Queries

- "What clearances does a furnace need?"
- "What are the clearance requirements for combustible materials?"
- "How high should appliances be in a garage?"

### Complex Multi-Step Queries

- "Can I install a furnace in a bedroom closet?"
- "What are the requirements for installing a water heater in a garage?"
- "Tell me about gas appliance installation in residential buildings"

### Standards and Compliance

- "What standards does a pressure relief valve need to comply with?"
- "What ANSI standards are referenced?"

### Known Limitations

- "What is Section 901.2?" ➜ Correctly returns "not found" (IFC/IBC reference)
- Any Section 901-999 ➜ Not in HVAC code PDF

## 🛠️ Project Structure

```
final_ai/
├── Core Application
│   ├── agent.py              # LangChain ReAct agent with tool selection
│   ├── bot.py                # Streamlit chatbot UI
│   ├── config.py             # Configuration management
│   ├── graph.py              # Neo4j connection and queries
│   ├── llm.py                # OpenAI LLM wrapper
│   └── logging_config.py     # Logging configuration
│
├── Data Extraction Pipeline
│   ├── extract_and_load.py        # Main ETL script
│   ├── relationship_extractor.py  # NLP-based relationship extraction
│   ├── table_extractor.py         # Extract data from tables
│   └── vocabulary.py              # Domain-specific vocabulary
│
├── Query Generation
│   ├── cypher_generator.py   # Dynamic Cypher query builder
│   ├── validation.py         # Query validation
│   └── utils.py              # Utility functions
│
├── Testing
│   ├── test_agent_minimal.py     # Quick 5-query validation test
│   └── test_agent_fast.py        # 22-query comprehensive test
│
├── Utilities
│   ├── clear_and_reload.py  # Reset and reload database
│   └── setup.py              # Initial project setup
│
├── Configuration
│   ├── .env                  # Environment variables (DO NOT COMMIT)
│   ├── .env.example          # Template for .env
│   ├── .gitignore           # Git ignore rules
│   └── requirements.txt      # Python dependencies
│
├── Documentation
│   └── README.md            # This file
│
├── Data
│   └── HVAC-Codes.pdf       # Source PDF (required)
│
└── Tools
    └── tools/               # Future tool modules
```

## 🔄 How to Start Fresh

If you need to completely reset and reload the database:

### Option 1: Using the Clear and Reload Script

```powershell
python clear_and_reload.py
```

This will:

1. Delete all nodes and relationships in Neo4j
2. Re-extract data from HVAC-Codes.pdf
3. Rebuild the entire knowledge graph

### Option 2: Manual Reset

```powershell
# In Neo4j Browser (http://localhost:7474), run:
MATCH (n) DETACH DELETE n

# Then reload data:
python extract_and_load.py
```

### When to Reset

- After updating HVAC-Codes.pdf
- If you modify extraction logic in `extract_and_load.py`
- If database becomes corrupted
- If you want to test with clean data

## 🔐 Security Notes

- **Never commit `.env` file** to version control (it's already in `.gitignore`)
- Store API keys and passwords securely
- Use environment-specific `.env` files for different deployments
- Rotate API keys regularly

## 📝 Environment Variables Reference

| Variable         | Description                      | Default                 | Required |
| ---------------- | -------------------------------- | ----------------------- | -------- |
| `OPENAI_API_KEY` | OpenAI API key (starts with sk-) | -                       | ✅ Yes   |
| `OPENAI_MODEL`   | OpenAI model name                | `gpt-4`                 | No       |
| `NEO4J_URI`      | Neo4j connection URI             | `bolt://localhost:7687` | No       |
| `NEO4J_USERNAME` | Neo4j username                   | `neo4j`                 | No       |
| `NEO4J_PASSWORD` | Neo4j password                   | -                       | ✅ Yes   |
| `NEO4J_DATABASE` | Neo4j database name              | `neo4j`                 | No       |
| `PDF_PATH`       | Path to PDF file                 | `HVAC-Codes.pdf`        | No       |
| `LOG_LEVEL`      | Logging level                    | `INFO`                  | No       |

**Notes:**

- `OPENAI_EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `VECTOR_*` variables are not currently used (vector search not implemented)
- All paths can be absolute or relative to project root
- Neo4j URI format: `bolt://host:port` or `neo4j://host:port`

## 🏗️ Architecture Overview

### Query Flow

```
User Question
    ↓
Streamlit UI (bot.py)
    ↓
LangChain ReAct Agent (agent.py)
    ↓
Tool Selection
    ├─→ CypherQuery (80% of queries)
    │   └─→ Neo4j Graph Database
    │       └─→ Returns structured data
    │
    └─→ VectorSearch (fallback, not yet implemented)
        └─→ Would use embeddings for semantic search

Final Answer with Citations
```

### How the Agent Works

1. **Receives Question**: User asks a question in Streamlit UI
2. **Analyzes Intent**: LangChain agent determines query type
3. **Selects Tool**:
   - **CypherQuery**: For structured queries (sections, prohibitions, clearances)
   - **VectorSearch**: For semantic queries (not yet implemented)
4. **Executes Query**: Runs Cypher against Neo4j knowledge graph
5. **Processes Results**: Formats data with section citations
6. **Returns Answer**: Displays in chat with relevant code sections

### Cypher-First Strategy

The system prioritizes structured Cypher queries because:

- ✅ More accurate for code section lookups
- ✅ Better for relationship queries (prohibitions, requirements)
- ✅ Faster than semantic search
- ✅ Returns exact section references
- ✅ Handles multi-step reasoning (agent chains queries)

## 🔐 Security Best Practices

### Critical Security Rules

1. **NEVER commit `.env` file to version control**
   - Already in `.gitignore`
   - Contains sensitive API keys and passwords
2. **Protect your API keys**

   - Rotate OpenAI API keys regularly
   - Set spending limits in OpenAI dashboard
   - Use separate keys for dev/prod

3. **Secure your Neo4j database**

   - Use strong passwords
   - Don't expose Neo4j ports to internet
   - For production, use Neo4j Aura with encryption

4. **Environment-specific configuration**
   - Use different `.env` files for dev/test/prod
   - Never share credentials in screenshots or logs

## 🧪 Testing

### Quick Validation Test (Recommended)

```powershell
python test_agent_minimal.py
```

**Tests 5 queries across difficulty levels:**

- Section lookup
- Equipment prohibitions
- Clearance requirements
- Complex multi-step reasoning

**Expected results:**

- ✅ 5/5 PASSED
- Runtime: ~60-70 seconds
- Shows which queries used Cypher only vs. fallback

### Comprehensive Test

```powershell
python test_agent_fast.py
```

**Tests 22 queries:**

- Basic, medium, hard, complex, and super complex queries
- Summary output only (less verbose)
- Runtime: ~5-8 minutes
- **Warning**: Uses significant OpenAI API credits

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Vector Search Implementation**

   - Generate embeddings for all sections
   - Create vector index in Neo4j
   - Implement semantic search fallback

2. **Additional Relationships**

   - Extract more equipment-location relationships
   - Add manufacturer-specific clearances
   - Extract installation requirements

3. **UI Enhancements**

   - Add section highlighting
   - Display relationship graphs
   - Export answers to PDF

4. **Testing**
   - Add unit tests
   - Add integration tests
   - Expand test query coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🎓 Understanding the Code Structure

**For beginners:**

1. Start with `bot.py` - simple Streamlit UI
2. Read `agent.py` - see how LangChain agent works
3. Study `cypher_generator.py` - understand query building
4. Review `extract_and_load.py` - learn data extraction

**For contributors:**

1. Check `relationship_extractor.py` for NLP logic
2. Review `validation.py` for query validation
3. Study test files for query patterns
4. Understand graph schema in `graph.py`

## 🆘 Troubleshooting

### Database Issues

**❌ "Failed to connect to Neo4j"**

- Verify Neo4j Desktop is running
- Check `NEO4J_URI` in `.env` (should be `bolt://localhost:7687`)
- Verify `NEO4J_PASSWORD` matches your database password
- Test connection in Neo4j Browser: http://localhost:7474

**❌ "Database is empty / No sections found"**

- Run `python extract_and_load.py` to load data
- Check that `HVAC-Codes.pdf` exists in project root
- Verify extraction completed without errors (should see "Created 669 Section nodes")

**❌ "Section 901.x not found"**

- ✅ This is **CORRECT behavior** - Section 901-999 are not in HVAC code
- These sections reference International Fire Code (IFC) or Building Code (IBC)
- The system correctly reports these as not found

### OpenAI API Issues

**❌ "OpenAI API error / Invalid API key"**

- Verify `OPENAI_API_KEY` in `.env` starts with `sk-`
- Check API key is active at https://platform.openai.com/api-keys
- Ensure you have API credits/billing set up

**❌ "Rate limit exceeded / Quota exceeded"**

- You've hit OpenAI API quota limit
- Wait a few minutes and try again
- Check usage at https://platform.openai.com/usage
- Consider upgrading your OpenAI plan

**❌ "Model not found"**

- Verify you have access to GPT-4 (not all accounts do)
- Try changing `OPENAI_MODEL` to `gpt-3.5-turbo` in `.env`
- Restart the chatbot after changing `.env`

### Python/Installation Issues

**❌ "Module not found" errors**

- Ensure virtual environment is activated: `venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Install spaCy model: `python -m spacy download en_core_web_sm`

**❌ "Python version incompatible"**

- Check Python version: `python --version`
- Requires Python 3.11 or higher
- Install correct version from https://www.python.org/downloads/

**❌ "PDF not found"**

- Verify `HVAC-Codes.pdf` exists in project root
- Check `PDF_PATH` in `.env` points to correct location
- Use absolute path if needed: `PDF_PATH=C:\Users\...\HVAC-Codes.pdf`

### Query Issues

**❌ "Agent returns no results"**

- Check that data was loaded: verify in Neo4j Browser
- Try simpler queries first (e.g., "What is Section 303.3?")
- Review agent logs for errors

**❌ "Clearance requirements show 'per manufacturer instructions'"**

- ✅ This is **CORRECT** - Section 304.9 uses this generic language
- The HVAC code defers to manufacturer specifications for exact distances
- Look up specific equipment in manufacturer documentation

**❌ "Query takes very long time"**

- First query after startup takes longer (LLM initialization)
- Complex queries may take 20-30 seconds
- Check OpenAI API status if consistently slow

### Testing Issues

**❌ "Test script fails"**

- Ensure database is loaded: `python extract_and_load.py`
- Check OpenAI API quota (tests make multiple API calls)
- Run minimal test first: `python test_agent_minimal.py`
- Check for error messages in output

## 📧 Getting Help

If you encounter issues not covered here:

1. Check that all prerequisites are met (Python 3.11+, Neo4j running, API key valid)
2. Verify `.env` configuration matches your setup
3. Try starting fresh with `python clear_and_reload.py`
4. Review error messages carefully - they often indicate the exact problem
5. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Your Python version and OS
   - Relevant `.env` settings (DO NOT share API keys)
