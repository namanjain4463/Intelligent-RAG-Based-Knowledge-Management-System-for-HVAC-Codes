# HVAC Codes Chatbot

An AI-powered chatbot for querying HVAC codes, equipment specifications, installation requirements, and regulatory standards using Neo4j graph database and OpenAI.

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Neo4j database instance
- OpenAI API key

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd final_ai

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your actual credentials:

   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-actual-openai-api-key
   OPENAI_MODEL=gpt-4

   # Neo4j Database Configuration
   NEO4J_URI=neo4j+s://your-actual-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-actual-neo4j-password
   ```

### 4. Load HVAC Data into Neo4j (First Time Setup)

Before running the chatbot, you need to load the HVAC codes data into your Neo4j database:

```bash
python property_rich_hvac.py
```

This script will:

- Extract text from `HVAC-Codes.pdf`
- Create semantic chunks for better search
- Extract property-rich entities (code sections, standards, equipment)
- Load everything into your Neo4j database with vector embeddings

### 5. Running the Application

```bash
streamlit run hvac_bot.py
```

The application will start and open in your default web browser at `http://localhost:8501`.

## Environment Variables

| Variable         | Description         | Example                            |
| ---------------- | ------------------- | ---------------------------------- |
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...`                           |
| `OPENAI_MODEL`   | OpenAI model to use | `gpt-4` or `gpt-3.5-turbo`         |
| `NEO4J_URI`      | Neo4j database URI  | `neo4j+s://xxx.databases.neo4j.io` |
| `NEO4J_USERNAME` | Neo4j username      | `neo4j`                            |
| `NEO4J_PASSWORD` | Neo4j password      | `your-password`                    |

## Project Structure

```
final_ai/
├── .env                    # Environment variables (not committed)
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── hvac_bot.py            # Main Streamlit application
├── agent.py               # Agent module (imports from hvac_agent)
├── hvac_agent.py          # Agent logic and tool definitions
├── llm.py                 # OpenAI LLM and embeddings configuration
├── graph.py               # Neo4j graph database connection
├── utils.py               # Utility functions
├── property_rich_hvac.py  # HVAC property extraction script
└── tools/                 # Tools package
    ├── __init__.py        # Package initializer
    ├── vector.py          # Vector search functionality
    └── cypher.py          # Cypher query generation
```

## Security Notes

- **Never commit your `.env` file** to version control
- The `.env` file is included in `.gitignore` to prevent accidental commits
- Use `.env.example` as a template for sharing configuration structure
- Keep your API keys and passwords secure

## Features

- Query HVAC codes and regulatory standards
- Search equipment specifications and installation requirements
- Natural language interface powered by OpenAI
- Graph database queries using Neo4j
- Vector similarity search for knowledge retrieval

## License

[Your License Here]
