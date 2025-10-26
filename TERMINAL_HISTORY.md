# Terminal Command History - HVAC GraphRAG Project

This file contains the complete terminal command history for setting up and managing the HVAC GraphRAG project.

## Project Setup Commands

### Initial Setup
```powershell
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Environment Configuration
```powershell
# Create .env from example
copy .env.example .env

# Edit .env (use notepad or your preferred editor)
notepad .env
```

## Database Operations

### Load Data into Neo4j
```powershell
# Extract data from PDF and load into Neo4j
python extract_and_load.py
```

### Reset and Reload Database
```powershell
# Clear database and reload fresh data
python clear_and_reload.py
```

## Running the Application

### Start Streamlit Chatbot
```powershell
# Run the interactive chatbot UI
streamlit run bot.py
```

### Quick Agent Test (Direct Query)
```powershell
# Test a single query
python -c "from agent import query_agent; print(query_agent('What is Section 303.3?'))"
```

## Testing Commands

### Minimal Test (5 queries - Recommended)
```powershell
# Quick validation test
python test_agent_minimal.py
```

### Fast Test (22 queries - Comprehensive)
```powershell
# More extensive testing
python test_agent_fast.py
```

### Previous Test Scripts (Now Deleted)
```powershell
# These were used during development but have been removed
python test_agent_quick.py          # DELETED
python test_agent_simple.py         # DELETED
python test_agent_comprehensive.py  # DELETED
```

## File Management Commands

### Cleanup Operations (Already Done)
```powershell
# Remove __pycache__ directories
Remove-Item -Path "__pycache__" -Recurse -Force

# Delete old test files
del test_agent_simple.py -ErrorAction SilentlyContinue
del test_agent_quick.py -ErrorAction SilentlyContinue
del test_agent_comprehensive.py -ErrorAction SilentlyContinue

# Delete check/diagnostic scripts (batch cleanup)
del add_general_clearances.py, analyze_*.py, check_*.py, verify_*.py -ErrorAction SilentlyContinue

# Delete old implementation files
del phase*.py, implement_*.py, fix_*.py, find_*.py -ErrorAction SilentlyContinue

# Delete old schema files
del COMPLETE_GRAPH_SCHEMA.py, ESSENTIAL_SCHEMA.py -ErrorAction SilentlyContinue

# Delete documentation files (kept README.md only)
del *_SUMMARY.md, TABLE_*.md -ErrorAction SilentlyContinue

# Delete logs and results
del comprehensive_test_results.txt, test_results.txt, hvac_graphrag.log -ErrorAction SilentlyContinue
```

## Git Operations

### Check Git Status
```powershell
git status
```

### View Remotes
```powershell
git remote -v
```

### Stage All Changes
```powershell
git add .
```

### Commit Changes
```powershell
git commit -m "Production-ready HVAC GraphRAG system with comprehensive testing and documentation"
```

### Remove Old Remote
```powershell
git remote remove origin
```

### Push to New GitHub Repository
```powershell
# After creating repository on GitHub, run:
git remote add origin https://github.com/namanjain4463/Final_Final_AI.git
git branch -M main
git push -u origin main

# OR simply run the helper script:
.\push_to_github.ps1
```

## Neo4j Browser Commands

### Access Neo4j Browser
```
http://localhost:7474
```

### Useful Cypher Queries in Neo4j Browser

#### View All Node Types and Counts
```cypher
MATCH (n) 
RETURN labels(n) as NodeType, count(*) as Count 
ORDER BY Count DESC
```

#### View All Relationship Types
```cypher
MATCH ()-[r]->() 
RETURN type(r) as RelationshipType, count(*) as Count 
ORDER BY Count DESC
```

#### Find All Sections
```cypher
MATCH (s:Section) 
RETURN s.number, s.title 
ORDER BY s.number 
LIMIT 50
```

#### Find Equipment and Their Prohibitions
```cypher
MATCH (e:Equipment)-[r:PROHIBITED_IN]->(l:Location)
RETURN e.name as Equipment, l.name as ProhibitedLocation, r.code_ref as Section
ORDER BY e.name
```

#### Find Equipment Clearance Requirements
```cypher
MATCH (e:Equipment)-[r:REQUIRES_CLEARANCE]->(target)
RETURN e.name as Equipment, 
       labels(target)[0] as TargetType, 
       target.name as Target, 
       r.min_inches as MinInches, 
       r.code_ref as Section,
       r.note as Note
ORDER BY e.name
```

#### Clear Entire Database (CAUTION!)
```cypher
MATCH (n) DETACH DELETE n
```

## Python Interactive Commands

### Test Individual Components
```powershell
# Test Neo4j connection
python -c "from graph import graph; print('Connection successful!' if graph else 'Connection failed')"

# Test LLM initialization
python -c "from llm import llm; print(llm.model_name)"

# Test config loading
python -c "from config import config; print(f'PDF Path: {config.pdf_path}')"

# Query agent directly
python -c "from agent import query_agent; result = query_agent('What is Section 303.3?'); print(result)"
```

### Check Python Environment
```powershell
# Verify Python version
python --version

# List installed packages
pip list

# Check if Neo4j is accessible
python -c "from neo4j import GraphDatabase; print('Neo4j driver installed')"

# Verify spaCy model
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model loaded')"
```

## Development Workflow

### Typical Development Session
```powershell
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Start Neo4j (via Neo4j Desktop or service)
# (Open Neo4j Desktop and start the database)

# 3. Load data if needed
python extract_and_load.py

# 4. Test the system
python test_agent_minimal.py

# 5. Run the chatbot
streamlit run bot.py

# 6. When done, commit changes
git add .
git commit -m "Your commit message"
git push
```

## Troubleshooting Commands

### Check if Neo4j is Running
```powershell
# Try to connect
python -c "from graph import graph; print('Neo4j is running!' if graph else 'Neo4j not accessible')"
```

### Verify Environment Variables
```powershell
# Check if .env file exists
Test-Path .env

# View .env content (be careful with API keys!)
Get-Content .env
```

### Check for Port Conflicts
```powershell
# Check if port 7687 (Neo4j) is in use
netstat -an | findstr "7687"

# Check if port 8501 (Streamlit) is in use
netstat -an | findstr "8501"
```

### Clear Python Cache
```powershell
# Remove all __pycache__ directories
Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force

# Remove .pyc files
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
```

## Performance Monitoring

### Monitor OpenAI API Usage
```powershell
# Check usage at: https://platform.openai.com/usage
# View API keys at: https://platform.openai.com/api-keys
```

### Monitor Neo4j Memory Usage
```powershell
# Open Neo4j Desktop
# View database statistics in the UI
# Or access metrics at: http://localhost:7474
```

## Backup Commands

### Backup Neo4j Database
```powershell
# Export all data to Cypher
# In Neo4j Browser, run:
# CALL apoc.export.cypher.all("backup.cypher", {})

# Note: Requires APOC plugin installed
```

### Backup Project Files
```powershell
# Create backup of entire project
Copy-Item -Path "C:\Users\Naman\Desktop\final_ai" -Destination "C:\Users\Naman\Desktop\final_ai_backup" -Recurse

# Backup just the code (no venv)
$exclude = @('venv', '__pycache__', '.env')
Get-ChildItem -Path . -Exclude $exclude | Copy-Item -Destination "C:\Users\Naman\Desktop\final_ai_code_backup" -Recurse
```

## Project Statistics

### Count Lines of Code
```powershell
# Count Python files
(Get-ChildItem -Filter "*.py" -Recurse | Get-Content | Measure-Object -Line).Lines

# Count all code files
(Get-ChildItem -Include "*.py","*.md","*.json" -Recurse | Get-Content | Measure-Object -Line).Lines
```

### List All Python Files
```powershell
Get-ChildItem -Filter "*.py" | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
```

---

## Quick Reference

### Most Common Commands
```powershell
# Start working
venv\Scripts\activate

# Test quickly
python test_agent_minimal.py

# Run chatbot
streamlit run bot.py

# Reload database
python clear_and_reload.py

# Check status
git status
```

### Emergency Reset
```powershell
# If something goes wrong, start fresh:
python clear_and_reload.py
python test_agent_minimal.py
```

---

**Last Updated:** October 26, 2025
**Project:** HVAC GraphRAG Knowledge Assistant
**Repository:** https://github.com/namanjain4463/Final_Final_AI
