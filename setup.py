"""
Setup Script for HVAC GraphRAG System
Helps configure the environment and verify setup
"""
import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("✓ .env file already exists")
        return True
    
    if not env_example_path.exists():
        print("❌ .env.example file not found")
        return False
    
    # Copy .env.example to .env
    with open(env_example_path, 'r') as f:
        content = f.read()
    
    with open(env_path, 'w') as f:
        f.write(content)
    
    print("✓ Created .env file from .env.example")
    print("\n⚠️  IMPORTANT: Please edit .env and add your credentials:")
    print("   - OPENAI_API_KEY")
    print("   - NEO4J_PASSWORD")
    return True

def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 11:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.11+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required = [
        'neo4j',
        'langchain',
        'openai',
        'streamlit',
        'dotenv',
        'spacy',
        'fitz'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package if package != 'dotenv' else 'dotenv')
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")
            missing.append(package)
    
    return len(missing) == 0

def check_env_variables():
    """Check if required environment variables are set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required = {
        'OPENAI_API_KEY': 'OpenAI API Key',
        'NEO4J_PASSWORD': 'Neo4j Password'
    }
    
    missing = []
    for key, description in required.items():
        value = os.getenv(key)
        if value and value.strip() and not value.startswith('your-'):
            print(f"✓ {description}")
        else:
            print(f"❌ {description} (not set in .env)")
            missing.append(key)
    
    return len(missing) == 0

def check_pdf_file():
    """Check if PDF file exists"""
    from dotenv import load_dotenv
    load_dotenv()
    
    pdf_path = os.getenv('PDF_PATH', 'HVAC-Codes.pdf')
    if Path(pdf_path).exists():
        print(f"✓ PDF file found: {pdf_path}")
        return True
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        print(f"   Please place your PDF at: {pdf_path}")
        print(f"   Or update PDF_PATH in .env")
        return False

def main():
    """Run setup checks"""
    print("="*60)
    print("HVAC GraphRAG System - Setup Verification")
    print("="*60)
    
    print("\n[1] Checking Python Version...")
    python_ok = check_python_version()
    
    print("\n[2] Checking/Creating .env File...")
    env_created = create_env_file()
    
    print("\n[3] Checking Dependencies...")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Install missing dependencies with:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")
    
    print("\n[4] Checking Environment Variables...")
    env_ok = check_env_variables()
    
    print("\n[5] Checking PDF File...")
    pdf_ok = check_pdf_file()
    
    print("\n" + "="*60)
    
    if python_ok and env_created and deps_ok and env_ok and pdf_ok:
        print("✅ Setup Complete! You're ready to go.")
        print("\nNext steps:")
        print("  1. Run: python extract_and_load.py")
        print("  2. Then: streamlit run bot.py")
    else:
        print("❌ Setup Incomplete. Please fix the issues above.")
        print("\nFor help, see README.md")
    
    print("="*60)

if __name__ == "__main__":
    main()
