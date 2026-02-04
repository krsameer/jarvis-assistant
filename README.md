# Jarvis - Personal AI Assistant for Enterprise SaaS

A demo implementation of a RAG-powered AI assistant using self-hosted LLM, FastAPI, Pinecone, and Streamlit.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Streamlit Chat App)                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                │ HTTP REST API
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         API Endpoints                                   │  │
│  │  • /ingest    - Document ingestion                      │  │
│  │  • /chat      - User query processing                   │  │
│  │  • /health    - Health check                            │  │
│  └────────┬──────────────────────────────────┬─────────────┘  │
│           │                                    │                │
│  ┌────────▼────────┐                 ┌────────▼────────────┐  │
│  │  RAG Service    │                 │   LLM Service       │  │
│  │  (Orchestrator) │◄────────────────│   (Ollama)          │  │
│  └────────┬────────┘                 └─────────────────────┘  │
│           │                                                     │
│  ┌────────▼────────┐                 ┌──────────────────────┐ │
│  │  Embedding      │                 │   Vector Store       │ │
│  │  Service        │◄────────────────│   (Pinecone)         │ │
│  └─────────────────┘                 └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 End-to-End Workflow

### Document Ingestion Flow
```
1. User uploads document via /ingest endpoint
   ↓
2. Text Processor splits document into chunks (with overlap)
   ↓
3. Embedding Service generates vector embeddings for each chunk
   ↓
4. Vector Store (Pinecone) indexes embeddings with metadata
   ↓
5. Return success confirmation
```

### Query Processing Flow (RAG)
```
1. User sends query via /chat endpoint
   ↓
2. Embedding Service converts query to vector
   ↓
3. Vector Store performs similarity search (retrieves top-k chunks)
   ↓
4. RAG Service constructs prompt with:
   - Retrieved context chunks
   - User query
   - System instructions
   ↓
5. LLM Service (Ollama) generates response
   ↓
6. Return response to user with sources
```

## 📋 Prerequisites

- Python 3.9+
- Ollama installed with ollama3 model
- Pinecone account (free tier works)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Pinecone API key and environment
```

### 3. Start Ollama (in separate terminal)
```bash
ollama run ollama3
```

### 4. Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Start Frontend
```bash
cd frontend
streamlit run streamlit_app.py
```

### 6. Access Application
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
jarvis-assistant/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── models.py            # Pydantic data models
│   └── services/
│       ├── llm_service.py         # Ollama LLM integration
│       ├── embedding_service.py   # Text embedding logic
│       ├── vector_store.py        # Pinecone operations
│       └── rag_service.py         # RAG orchestration
├── frontend/
│   └── streamlit_app.py     # Chat UI
└── utils/
    └── text_processor.py    # Document chunking utilities
```

## 🔑 Key Components

### 1. **LLM Service** (`llm_service.py`)
- Interfaces with Ollama for text generation
- Handles streaming and non-streaming responses
- Temperature and parameter control

### 2. **Embedding Service** (`embedding_service.py`)
- Generates vector embeddings for text
- Uses sentence-transformers or Ollama embeddings
- Consistent embedding dimension management

### 3. **Vector Store** (`vector_store.py`)
- Pinecone client wrapper
- Index management (create, upsert, query)
- Metadata filtering and similarity search

### 4. **RAG Service** (`rag_service.py`)
- Orchestrates the RAG pipeline
- Context retrieval and prompt engineering
- Response generation with source attribution

### 5. **Text Processor** (`text_processor.py`)
- Document chunking with overlap
- Text cleaning and normalization
- Metadata extraction

## 🎯 API Endpoints

### POST /ingest
Ingest and index documents for retrieval.

**Request:**
```json
{
  "text": "Your document content here...",
  "metadata": {
    "source": "document.pdf",
    "type": "policy"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "chunks_processed": 15,
  "document_id": "doc_123"
}
```

### POST /chat
Query the AI assistant with RAG.

**Request:**
```json
{
  "query": "What is our vacation policy?",
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "Based on the company policy...",
  "sources": [
    {
      "text": "Employees are entitled to...",
      "metadata": {"source": "hr_policy.pdf", "page": 5}
    }
  ],
  "tokens_used": 250
}
```

## 🔧 Configuration

Key environment variables in `.env`:

- `PINECONE_API_KEY`: Your Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment (e.g., "us-east-1-aws")
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `OLLAMA_BASE_URL`: Ollama API endpoint (default: http://localhost:11434)
- `OLLAMA_MODEL`: Model name (default: ollama3)
- `EMBEDDING_MODEL`: Embedding model name

## 🎓 Enterprise Considerations

### For Production Deployment:

1. **Security**
   - Add authentication (OAuth2, JWT)
   - Rate limiting
   - Input validation and sanitization
   - API key rotation

2. **Scalability**
   - Use async processing for document ingestion
   - Implement caching (Redis)
   - Load balancing for API
   - Batch embedding generation

3. **Monitoring**
   - Logging (structured logs)
   - Metrics (Prometheus)
   - Tracing (OpenTelemetry)
   - Error tracking (Sentry)

4. **Data Management**
   - Document versioning
   - Metadata enrichment
   - Data retention policies
   - GDPR compliance

5. **Performance**
   - Connection pooling
   - Query optimization
   - Embedding caching
   - Model quantization

## 📝 Usage Examples

### Ingest a Document
```python
import requests

response = requests.post("http://localhost:8000/ingest", json={
    "text": "Company vacation policy: All full-time employees...",
    "metadata": {"source": "hr_handbook.pdf", "section": "benefits"}
})
print(response.json())
```

### Query the Assistant
```python
response = requests.post("http://localhost:8000/chat", json={
    "query": "How many vacation days do I get?",
    "conversation_history": []
})
print(response.json()["response"])
```

## 🐛 Troubleshooting

**Ollama connection error:**
- Ensure Ollama is running: `ollama list`
- Check the model is pulled: `ollama pull ollama3`

**Pinecone authentication error:**
- Verify API key in `.env`
- Check Pinecone dashboard for correct environment

**Empty retrieval results:**
- Ingest documents first using `/ingest` endpoint
- Verify index name matches configuration

## 📚 Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Ollama Documentation](https://ollama.ai/docs)
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## 📄 License

MIT License - Demo/Educational purposes
