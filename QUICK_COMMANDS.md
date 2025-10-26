# Quick Command Reference Card

## 🚀 Essential Commands (Use These Daily)

### Activate Environment
```powershell
venv\Scripts\activate
```

### Run Chatbot
```powershell
streamlit run bot.py
```

### Quick Test
```powershell
python test_agent_minimal.py
```

### Reload Database
```powershell
python clear_and_reload.py
```

---

## 📊 Project Setup (One-Time)

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate

# 3. Install packages
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Configure environment
copy .env.example .env
notepad .env

# 6. Load data into Neo4j
python extract_and_load.py
```

---

## 🔧 Git Commands

```powershell
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "Your message"

# Push
git push

# View history
git log --oneline
```

---

## 🐛 Troubleshooting

```powershell
# Clear Python cache
Remove-Item -Path "__pycache__" -Recurse -Force

# Check Neo4j connection
python -c "from graph import graph; print('Connected!' if graph else 'Failed')"

# Check Python version
python --version

# List packages
pip list
```

---

## 🌐 Access Points

- **Streamlit UI:** http://localhost:8501
- **Neo4j Browser:** http://localhost:7474
- **OpenAI Dashboard:** https://platform.openai.com/usage
- **GitHub Repo:** https://github.com/namanjain4463/Final_Final_AI

---

For complete command reference, see: `TERMINAL_HISTORY.md`
