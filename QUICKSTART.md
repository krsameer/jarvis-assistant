# Jarvis AI Assistant - Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Prerequisites
```bash
# Check Python version (need 3.9+)
python3 --version

# Check if Ollama is installed
ollama --version

# If Ollama not installed, visit: https://ollama.ai
```

### 2. Setup
```bash
# Clone/navigate to project
cd jarvis-assistant

# Run setup script
chmod +x setup.sh
./setup.sh

# Edit .env with your Pinecone API key
nano .env
```

### 3. Start Ollama (Terminal 1)
```bash
# Pull and run the model
ollama run ollama3

# Keep this terminal open
```

### 4. Start Backend (Terminal 2)
```bash
# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
cd backend
uvicorn main:app --reload --port 8000

# Keep this terminal open
```

### 5. Start Frontend (Terminal 3)
```bash
# Activate virtual environment
source venv/bin/activate

# Start Streamlit app
cd frontend
streamlit run streamlit_app.py

# Your browser should auto-open to http://localhost:8501
```

### 6. Use the App
1. **Add Knowledge**: In the sidebar, paste a document and click "Ingest Document"
2. **Ask Questions**: Type questions in the chat interface
3. **View Sources**: Click on "Sources" to see where answers came from

## 📚 Example Documents to Ingest

### Employee Handbook Example
```
Company Vacation Policy

All full-time employees are entitled to:
- 15 days of paid vacation per year for employees with 0-2 years of service
- 20 days of paid vacation per year for employees with 3-5 years of service
- 25 days of paid vacation per year for employees with 6+ years of service

Vacation requests must be submitted at least 2 weeks in advance through the HR portal.
Unused vacation days can be carried over up to a maximum of 5 days.
```

### Product Manual Example
```
SaaS Platform - User Authentication

Our platform supports multiple authentication methods:
1. Username/Password with 2FA
2. OAuth 2.0 (Google, Microsoft, GitHub)
3. SAML 2.0 for enterprise SSO

Password Requirements:
- Minimum 12 characters
- At least one uppercase letter
- At least one number
- At least one special character

Session timeout is set to 8 hours of inactivity.
```

## 🧪 Test Queries

After ingesting documents, try these queries:
- "How many vacation days do new employees get?"
- "What are the password requirements?"
- "How do I authenticate with SSO?"
- "Can I carry over unused vacation days?"

## 🔧 Troubleshooting

### Backend won't start
```bash
# Check if all dependencies installed
pip install -r requirements.txt

# Check environment variables
cat .env

# Verify Pinecone credentials
```

### LLM errors
```bash
# Verify Ollama is running
ollama list

# Ensure ollama3 model is pulled
ollama pull ollama3

# Test Ollama directly
curl http://localhost:11434/api/tags
```

### Empty retrieval results
```bash
# Check if documents were ingested
# Use /stats endpoint: http://localhost:8000/stats

# Verify Pinecone index exists
# Check Pinecone dashboard
```

## 📖 API Testing (without UI)

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Employee benefits include health insurance, 401k matching, and wellness programs.",
    "metadata": {"source": "benefits.pdf", "section": "overview"}
  }'
```

### Ask a question
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What benefits are available?"
  }'
```

### Check health
```bash
curl http://localhost:8000/health
```

## 🎯 What Next?

1. **Add more documents**: Build up your knowledge base
2. **Customize system prompt**: Edit `llm_service.py` to change behavior
3. **Adjust chunking**: Modify chunk size in `.env` for better results
4. **Try different models**: Change `OLLAMA_MODEL` in `.env`
5. **Add authentication**: Implement OAuth2 for production use

## 📊 Monitoring

Access these URLs while running:
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Stats**: http://localhost:8000/stats

## 🛑 Shutdown

```bash
# Stop Streamlit: Ctrl+C in terminal 3
# Stop FastAPI: Ctrl+C in terminal 2
# Stop Ollama: Ctrl+C in terminal 1
```

## 💡 Tips

1. **Better Retrieval**: Use descriptive metadata when ingesting
2. **Conversation Context**: The system remembers last 5 conversation turns
3. **Chunk Size**: Smaller chunks = more precise, larger = more context
4. **Model Selection**: Try different Ollama models for different use cases
5. **Similarity Threshold**: Adjust in RAG service for stricter/looser matching

Enjoy using Jarvis! 🤖
