"""
Configuration module for HVAC GraphRAG System
Centralizes all environment variable access
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration from environment variables"""
    
    # ==================== OpenAI Configuration ====================
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # ==================== Neo4j Configuration ====================
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # SECURITY: No default password
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # ==================== Application Configuration ====================
    PDF_PATH = os.getenv("PDF_PATH", "HVAC-Codes.pdf")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # ==================== Vector Search Configuration ====================
    VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "hvac_chunk_embeddings")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1536"))
    VECTOR_SIMILARITY = os.getenv("VECTOR_SIMILARITY", "cosine")
    
    # ==================== Logging Configuration ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "hvac_graphrag.log")
    
    @classmethod
    def validate(cls):
        """Validate required configuration with enhanced checks"""
        errors = []
        warnings = []
        
        # OpenAI API Key validation
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required in .env file")
        elif not cls.OPENAI_API_KEY.startswith("sk-"):
            warnings.append("OPENAI_API_KEY format looks incorrect (should start with 'sk-')")
        
        # Neo4j Password validation
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD is required in .env file")
        
        # PDF file existence check
        import os
        if not os.path.exists(cls.PDF_PATH):
            warnings.append(f"PDF file not found: {cls.PDF_PATH} (required for data extraction)")
        
        # Print warnings
        if warnings:
            print("\n⚠️  Configuration Warnings:")
            for w in warnings:
                print(f"  - {w}")
        
        # Raise error if critical issues found
        if errors:
            raise ValueError(
                "\n❌ Configuration Errors - Please fix the following:\n  - " + 
                "\n  - ".join(errors) +
                "\n\nSet these variables in your .env file"
            )
        
        return True
    
    @classmethod
    def display(cls):
        """Display current configuration (without sensitive data)"""
        print("="*60)
        print("Current Configuration")
        print("="*60)
        print(f"OpenAI Model:          {cls.OPENAI_MODEL}")
        print(f"Embedding Model:       {cls.OPENAI_EMBEDDING_MODEL}")
        print(f"Neo4j URI:             {cls.NEO4J_URI}")
        print(f"Neo4j Database:        {cls.NEO4J_DATABASE}")
        print(f"PDF Path:              {cls.PDF_PATH}")
        print(f"Chunk Size:            {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap:         {cls.CHUNK_OVERLAP}")
        print(f"Vector Index:          {cls.VECTOR_INDEX_NAME}")
        print(f"Vector Dimensions:     {cls.VECTOR_DIMENSION}")
        print(f"Vector Similarity:     {cls.VECTOR_SIMILARITY}")
        print(f"Log Level:             {cls.LOG_LEVEL}")
        print(f"Log File:              {cls.LOG_FILE}")
        print("="*60)

# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"\n❌ Configuration Error:\n{e}\n")
    raise
