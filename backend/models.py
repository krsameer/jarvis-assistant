"""
Pydantic Models for Request/Response Validation
Defines data structures for API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class IngestRequest(BaseModel):
    """
    Request model for document ingestion endpoint.
    
    Example:
    {
        "text": "Company policy document...",
        "metadata": {
            "source": "hr_handbook.pdf",
            "section": "benefits",
            "page": 5
        }
    }
    """
    text: str = Field(..., description="Document text to ingest")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata (source, type, date, etc.)"
    )
    chunk_size: Optional[int] = Field(
        default=500,
        description="Size of text chunks"
    )
    chunk_overlap: Optional[int] = Field(
        default=50,
        description="Overlap between chunks"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Our company offers comprehensive health insurance...",
                "metadata": {
                    "source": "benefits_guide.pdf",
                    "type": "policy",
                    "department": "HR"
                },
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }


class IngestResponse(BaseModel):
    """
    Response model for document ingestion.
    """
    status: str = Field(..., description="Status of the operation")
    chunks_processed: int = Field(..., description="Number of chunks created")
    vectors_stored: int = Field(..., description="Number of vectors stored in Pinecone")
    document_id: Optional[str] = Field(None, description="Generated document ID")
    message: Optional[str] = Field(None, description="Additional message")


class ConversationTurn(BaseModel):
    """
    Single turn in a conversation.
    """
    user: str = Field(..., description="User message")
    assistant: str = Field(..., description="Assistant response")


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint.
    
    Example:
    {
        "query": "What is our vacation policy?",
        "conversation_history": [
            {
                "user": "Hello",
                "assistant": "Hi! How can I help you?"
            }
        ],
        "filter": {
            "source": "hr_handbook.pdf"
        }
    }
    """
    query: str = Field(..., description="User's question")
    conversation_history: Optional[List[ConversationTurn]] = Field(
        default=None,
        description="Previous conversation turns"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter for scoped retrieval"
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of context chunks to retrieve"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the vacation policy for new employees?",
                "conversation_history": [],
                "top_k": 5
            }
        }


class Source(BaseModel):
    """
    Source citation for a response.
    """
    text: str = Field(..., description="Text snippet from source")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default={}, description="Source metadata")


class ChatResponse(BaseModel):
    """
    Response model for chat endpoint.
    """
    status: str = Field(..., description="Status of the operation")
    response: str = Field(..., description="Generated response")
    sources: List[Source] = Field(default=[], description="Source citations")
    context_count: Optional[int] = Field(None, description="Number of context chunks used")
    message: Optional[str] = Field(None, description="Error message if status is error")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Overall system status")
    services: Dict[str, bool] = Field(..., description="Status of individual services")
    message: Optional[str] = Field(None, description="Additional information")


class StatsResponse(BaseModel):
    """
    Response model for system statistics endpoint.
    """
    embedding_model: Dict[str, Any] = Field(..., description="Embedding model info")
    llm_model: Dict[str, Any] = Field(..., description="LLM model info")
    vector_store: Dict[str, Any] = Field(..., description="Vector store stats")
    config: Dict[str, Any] = Field(..., description="System configuration")
