"""
Vector Store Service - Pinecone Integration
Handles storage and retrieval of embeddings using Pinecone vector database.
"""

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from typing import List, Dict, Optional, Tuple
import uuid
import time


class VectorStore:
    """
    Service for managing vector embeddings in Pinecone.
    
    Pinecone is a managed vector database that excels at:
    - Fast similarity search at scale
    - Automatic indexing and sharding
    - Metadata filtering
    - High availability
    
    In a RAG system, this is where we store document embeddings
    and retrieve relevant context for user queries.
    """
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 384,
        metric: str = "cosine"
    ):
        """
        Initialize the vector store.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east-1-aws')
            index_name: Name of the Pinecone index
            dimension: Embedding dimension (must match embedding model)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
                   'cosine' is best for normalized embeddings
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
        # Ensure index exists
        self._ensure_index()
    
    def _ensure_index(self):
        """
        Create Pinecone index if it doesn't exist, or connect to existing one.
        
        Index creation is idempotent - safe to call multiple times.
        """
        try:
            # List existing indexes
            existing_indexes = [idx['name'] for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless spec (recommended for most use cases)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(5)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            raise Exception(f"Failed to initialize Pinecone index: {str(e)}")
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> Dict:
        """
        Insert or update vectors in the index.
        
        This is the core operation for storing embeddings.
        "Upsert" = Update + Insert (overwrites if ID exists)
        
        Args:
            vectors: List of embedding vectors
            texts: Original text chunks (stored as metadata)
            metadatas: Optional metadata for each vector
            ids: Optional custom IDs (auto-generated if not provided)
            
        Returns:
            Dictionary with upsert statistics
        """
        if not self.index:
            raise Exception("Index not initialized")
        
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors and texts must match")
        
        # Generate IDs if not provided
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Prepare metadata
        if not metadatas:
            metadatas = [{} for _ in range(len(vectors))]
        
        # Add text to metadata (for retrieval)
        for i, text in enumerate(texts):
            metadatas[i]["text"] = text
            metadatas[i]["id"] = ids[i]
        
        try:
            # Prepare vectors for upsert
            # Format: [(id, vector, metadata), ...]
            vectors_to_upsert = [
                (ids[i], vectors[i], metadatas[i])
                for i in range(len(vectors))
            ]
            
            # Upsert in batches (Pinecone limit: 100 vectors per request)
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                response = self.index.upsert(vectors=batch)
                total_upserted += response.upserted_count
            
            return {
                "status": "success",
                "upserted_count": total_upserted,
                "ids": ids
            }
            
        except Exception as e:
            raise Exception(f"Failed to upsert vectors: {str(e)}")
    
    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Perform similarity search to find most relevant vectors.
        
        This is the retrieval step in RAG:
        1. Convert user query to vector
        2. Find top-k most similar document chunks
        3. Return the original text with metadata
        
        Args:
            query_vector: Query embedding
            top_k: Number of results to return
            filter: Metadata filter (e.g., {"source": "handbook.pdf"})
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching results with text and metadata
        """
        if not self.index:
            raise Exception("Index not initialized")
        
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=include_metadata
            )
            
            # Format results for consumption
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),  # Similarity score
                    "text": match.metadata.get("text", "") if include_metadata else "",
                    "metadata": {k: v for k, v in match.metadata.items() if k != "text"} if include_metadata else {}
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Query failed: {str(e)}")
    
    def delete_vectors(self, ids: List[str]) -> Dict:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Deletion status
        """
        if not self.index:
            raise Exception("Index not initialized")
        
        try:
            self.index.delete(ids=ids)
            return {
                "status": "success",
                "deleted_count": len(ids)
            }
        except Exception as e:
            raise Exception(f"Failed to delete vectors: {str(e)}")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index.
        
        Useful for:
        - Monitoring index size
        - Debugging
        - Usage tracking
        
        Returns:
            Dictionary with index statistics
        """
        if not self.index:
            raise Exception("Index not initialized")
        
        try:
            stats = self.index.describe_index_stats()
            # Convert namespaces to a plain JSON-serializable dict
            namespaces = {}
            if stats.namespaces:
                for ns_name, ns_summary in stats.namespaces.items():
                    try:
                        namespaces[ns_name] = {
                            "vector_count": getattr(ns_summary, "vector_count", 0)
                        }
                    except Exception:
                        namespaces[ns_name] = {}

            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": float(stats.index_fullness) if stats.index_fullness is not None else 0.0,
                "namespaces": namespaces
            }
        except Exception as e:
            raise Exception(f"Failed to get index stats: {str(e)}")
    
    def clear_index(self) -> Dict:
        """
        Delete all vectors from the index.
        
        WARNING: This is destructive and cannot be undone.
        Use with caution!
        
        Returns:
            Status of the operation
        """
        if not self.index:
            raise Exception("Index not initialized")
        
        try:
            self.index.delete(delete_all=True)
            return {
                "status": "success",
                "message": "All vectors deleted"
            }
        except Exception as e:
            raise Exception(f"Failed to clear index: {str(e)}")


def create_vector_store(
    api_key: str,
    environment: str,
    index_name: str,
    dimension: int = 384,
    metric: str = "cosine"
) -> VectorStore:
    """
    Factory function to create a VectorStore instance.
    
    Args:
        api_key: Pinecone API key
        environment: Pinecone environment
        index_name: Name of the index
        dimension: Embedding dimension
        metric: Distance metric
        
    Returns:
        Configured VectorStore instance
    """
    return VectorStore(
        api_key=api_key,
        environment=environment,
        index_name=index_name,
        dimension=dimension,
        metric=metric
    )
