"""
RAG Service - Retrieval-Augmented Generation Orchestrator
Coordinates the entire RAG pipeline: embedding, retrieval, and generation.
"""

from typing import List, Dict, Optional
from .embedding_service import EmbeddingService
from .vector_store import VectorStore
from .llm_service import LLMService


class RAGService:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    
    RAG Flow:
    1. INGESTION: Document → Chunks → Embeddings → Vector DB
    2. RETRIEVAL: Query → Embedding → Similarity Search → Context
    3. GENERATION: Context + Query → LLM → Response
    
    This service brings together all the components to provide
    a seamless question-answering experience.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the RAG service.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for storage/retrieval
            llm_service: LLM for text generation
            top_k: Number of context chunks to retrieve
            similarity_threshold: Minimum similarity score for results
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    async def ingest_documents(
        self,
        chunks: List[Dict],
        batch_size: int = 32
    ) -> Dict:
        """
        Ingest document chunks into the vector database.
        
        This is the INGESTION phase of RAG:
        - Takes pre-chunked documents
        - Generates embeddings for each chunk
        - Stores in Pinecone with metadata
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            batch_size: Number of chunks to process at once
            
        Returns:
            Ingestion statistics
        """
        if not chunks:
            return {
                "status": "error",
                "message": "No chunks provided"
            }
        
        try:
            # Extract text from chunks
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk.get("metadata", {}) for chunk in chunks]
            
            # Generate embeddings in batches for efficiency
            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_service.embed_documents(texts)
            
            # Store in vector database
            print(f"Storing embeddings in Pinecone...")
            result = self.vector_store.upsert_vectors(
                vectors=embeddings,
                texts=texts,
                metadatas=metadatas
            )
            
            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "vectors_stored": result["upserted_count"],
                "ids": result["ids"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Ingestion failed: {str(e)}"
            }
    
    async def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant context for a query.
        
        This is the RETRIEVAL phase of RAG:
        - Convert query to embedding
        - Search vector database for similar chunks
        - Return ranked results
        
        Args:
            query: User's question
            top_k: Number of results (overrides default)
            filter: Metadata filter for scoped search
            
        Returns:
            List of relevant context chunks with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Search vector database
            results = self.vector_store.query(
                query_vector=query_embedding,
                top_k=top_k or self.top_k,
                filter=filter
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in results 
                if r["score"] >= self.similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []
    
    async def generate_response(
        self,
        query: str,
        context: Optional[List[Dict]] = None,
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict:
        """
        Generate a response using the LLM with retrieved context.
        
        This is the GENERATION phase of RAG:
        - Takes query and retrieved context
        - Constructs a prompt
        - Calls LLM to generate response
        - Returns response with sources
        
        Args:
            query: User's question
            context: Retrieved context chunks (if not provided, will retrieve)
            conversation_history: Previous conversation turns
            system_prompt: Custom system instructions
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        try:
            # Retrieve context if not provided
            if context is None:
                print(f"Retrieving context for query: {query}")
                context = await self.retrieve_context(query)
            
            # Extract text from context
            context_texts = [c["text"] for c in context] if context else []
            
            # Generate response using LLM
            print(f"Generating response with {len(context_texts)} context chunks...")
            response = await self.llm_service.generate_with_context(
                query=query,
                context=context_texts,
                conversation_history=conversation_history,
                system_prompt=system_prompt
            )
            
            # Prepare sources for citation
            sources = []
            if context:
                for ctx in context:
                    source = {
                        "text": ctx["text"][:200] + "...",  # Truncate for display
                        "score": ctx["score"],
                        "metadata": ctx.get("metadata", {})
                    }
                    sources.append(source)
            
            return {
                "status": "success",
                "response": response,
                "sources": sources,
                "context_count": len(context_texts)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Generation failed: {str(e)}",
                "response": "I apologize, but I encountered an error processing your request.",
                "sources": []
            }
    
    async def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        filter: Optional[Dict] = None
    ) -> Dict:
        """
        Complete RAG chat workflow.
        
        This is the end-to-end RAG pipeline:
        1. Retrieve relevant context
        2. Generate response with context
        3. Return response with citations
        
        This is the main method for user-facing chat.
        
        Args:
            query: User's question
            conversation_history: Previous conversation
            filter: Metadata filter for scoped retrieval
            
        Returns:
            Complete response with sources and metadata
        """
        # Retrieve relevant context
        context = await self.retrieve_context(query, filter=filter)
        
        # Generate response
        result = await self.generate_response(
            query=query,
            context=context,
            conversation_history=conversation_history
        )
        
        return result
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the RAG system.
        
        Returns:
            System statistics and configuration
        """
        return {
            "embedding_model": self.embedding_service.get_model_info(),
            "llm_model": self.llm_service.get_model_info(),
            "vector_store": self.vector_store.get_index_stats(),
            "config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            }
        }


def create_rag_service(
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    llm_service: LLMService,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> RAGService:
    """
    Factory function to create a RAG service instance.
    
    Args:
        embedding_service: Embedding service
        vector_store: Vector store
        llm_service: LLM service
        top_k: Number of results to retrieve
        similarity_threshold: Minimum similarity score
        
    Returns:
        Configured RAGService instance
    """
    return RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
