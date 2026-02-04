"""
FastAPI Backend - Main Application
Entry point for the Jarvis AI Assistant API.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Import models
from models import (
    IngestRequest, IngestResponse,
    ChatRequest, ChatResponse,
    HealthResponse, StatsResponse
)

# Import services
from services.embedding_service import create_embedding_service
from services.vector_store import create_vector_store
from services.llm_service import create_llm_service
from services.rag_service import create_rag_service

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processor import create_text_processor

# Load environment variables
load_dotenv()

# Global service instances (initialized at startup)
rag_service = None
text_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize all services
    - Shutdown: Clean up resources
    """
    global rag_service, text_processor
    
    print("=" * 60)
    print("🚀 Starting Jarvis AI Assistant Backend...")
    print("=" * 60)
    
    # Load configuration from environment
    try:
        # Pinecone configuration
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge-base")
        
        # Ollama configuration
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "ollama3")
        
        # Embedding configuration
        embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        
        # RAG configuration
        top_k = int(os.getenv("TOP_K_RESULTS", "5"))
        temperature = float(os.getenv("TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("MAX_TOKENS", "500"))
        
        # Chunking configuration
        chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        print("\n📋 Configuration:")
        print(f"  • Ollama Model: {ollama_model}")
        print(f"  • Embedding Model: {embedding_model}")
        print(f"  • Pinecone Index: {pinecone_index}")
        print(f"  • Top-K Results: {top_k}")
        
        # Initialize services
        print("\n🔧 Initializing services...")
        
        # 1. Embedding Service
        print("  1️⃣  Loading embedding model...")
        embedding_service = create_embedding_service(model_name=embedding_model)
        
        # 2. Vector Store
        print("  2️⃣  Connecting to Pinecone...")
        vector_store = create_vector_store(
            api_key=pinecone_api_key,
            environment=pinecone_env,
            index_name=pinecone_index,
            dimension=embedding_dim
        )
        
        # 3. LLM Service
        print("  3️⃣  Configuring Ollama LLM...")
        llm_service = create_llm_service(
            base_url=ollama_url,
            model=ollama_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 4. RAG Service
        print("  4️⃣  Initializing RAG pipeline...")
        rag_service = create_rag_service(
            embedding_service=embedding_service,
            vector_store=vector_store,
            llm_service=llm_service,
            top_k=top_k
        )
        
        # 5. Text Processor
        print("  5️⃣  Setting up text processor...")
        text_processor = create_text_processor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        print("\n✅ All services initialized successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Startup failed: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    print("\n🛑 Shutting down Jarvis AI Assistant...")


# Create FastAPI application
app = FastAPI(
    title="Jarvis AI Assistant API",
    description="Enterprise RAG-powered AI assistant using Ollama, Pinecone, and FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "Jarvis AI Assistant API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "chat": "/chat",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Verifies that all services are operational:
    - LLM (Ollama) is accessible
    - Vector store (Pinecone) is connected
    - Embedding service is loaded
    """
    services_status = {}
    overall_healthy = True
    
    try:
        # Check LLM service
        llm_healthy = await rag_service.llm_service.check_health()
        services_status["llm"] = llm_healthy
        overall_healthy &= llm_healthy
        
        # Check vector store
        try:
            rag_service.vector_store.get_index_stats()
            services_status["vector_store"] = True
        except:
            services_status["vector_store"] = False
            overall_healthy = False
        
        # Check embedding service (using Ollama API)
        services_status["embedding"] = rag_service.embedding_service._dimension is not None
        overall_healthy &= services_status["embedding"]
        
        return HealthResponse(
            status="healthy" if overall_healthy else "degraded",
            services=services_status,
            message="All systems operational" if overall_healthy else "Some services unavailable"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@app.post("/ingest", response_model=IngestResponse, tags=["Document Management"])
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the knowledge base.
    
    Process:
    1. Split document into chunks
    2. Generate embeddings
    3. Store in Pinecone
    
    This endpoint is used to add new documents to the RAG system.
    Documents can be policies, manuals, FAQs, etc.
    """
    try:
        # Validate input
        if not request.text or len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is too short or empty"
            )
        
        # Process text into chunks
        print(f"📄 Processing document ({len(request.text)} chars)...")
        chunks = text_processor.chunk_text(
            text=request.text,
            metadata=request.metadata
        )
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks generated from text"
            )
        
        print(f"  ✂️  Created {len(chunks)} chunks")
        
        # Ingest into RAG system
        result = await rag_service.ingest_documents(chunks)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Ingestion failed")
            )
        
        print(f"  ✅ Stored {result['vectors_stored']} vectors in Pinecone")
        
        return IngestResponse(
            status="success",
            chunks_processed=result["chunks_processed"],
            vectors_stored=result["vectors_stored"],
            document_id=result["ids"][0] if result.get("ids") else None,
            message="Document ingested successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat with the AI assistant using RAG.
    
    Process:
    1. Retrieve relevant context from knowledge base
    2. Generate response using LLM with context
    3. Return response with source citations
    
    This is the main endpoint for user interaction.
    """
    try:
        # Validate query
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is too short or empty"
            )
        
        print(f"\n💬 Processing query: {request.query[:100]}...")
        
        # Convert conversation history to dict format
        history = None
        if request.conversation_history:
            history = [
                {"user": turn.user, "assistant": turn.assistant}
                for turn in request.conversation_history
            ]
        
        # Execute RAG pipeline
        result = await rag_service.chat(
            query=request.query,
            conversation_history=history,
            filter=request.filter
        )
        
        if result["status"] == "error":
            print(f"  ❌ Error: {result.get('message')}")
            return ChatResponse(
                status="error",
                response=result["response"],
                sources=[],
                message=result.get("message")
            )
        
        print(f"  ✅ Generated response with {result['context_count']} context chunks")
        print(f"  📚 {len(result['sources'])} sources cited")
        
        return ChatResponse(
            status="success",
            response=result["response"],
            sources=result["sources"],
            context_count=result["context_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}"
        )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """
    Get system statistics and configuration.
    
    Returns information about:
    - Embedding model
    - LLM model
    - Vector store status
    - System configuration
    """
    try:
        stats = rag_service.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (development only)
        log_level="info"
    )
