"""
Embedding Service
Generates vector embeddings for text using Ollama embeddings API (lightweight alternative).
"""

import httpx
from typing import List, Union
import numpy as np


class EmbeddingService:
    """
    Service for generating text embeddings using Ollama.
    
    Embeddings are vector representations of text that capture semantic meaning.
    Similar texts will have similar embeddings (closer in vector space).
    
    Why Ollama embeddings?
    - No need for PyTorch (lightweight)
    - Uses the same model infrastructure
    - Works well with Pinecone
    - Can run locally (no API calls to external services)
    """
    
    def __init__(
        self, 
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Ollama embedding model
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self._dimension = 768  # nomic-embed-text dimension
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        
        print(f"Using Ollama embeddings: {self.model_name}")
        print(f"Embedding dimension: {self._dimension}")
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding from Ollama API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["embedding"]
        except Exception as e:
            raise Exception(f"Embedding API error: {str(e)}")
    
    async def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.
        
        This converts text into a vector representation that can be:
        - Stored in Pinecone
        - Compared for similarity
        - Used for semantic search
        
        Args:
            text: Single text string or list of strings
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        try:
            # Handle single string
            if isinstance(text, str):
                return await self._get_embedding(text)
            
            # Handle list of strings (batch processing)
            elif isinstance(text, list):
                embeddings = []
                for t in text:
                    embedding = await self._get_embedding(t)
                    embeddings.append(embedding)
                return embeddings
            
            else:
                raise ValueError("Text must be a string or list of strings")
                
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        This is a convenience method specifically for queries.
        Uses the same underlying model but makes intent clear.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return await self.embed_text(query)
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (batch operation).
        
        This is optimized for processing many documents at once,
        which is common during document ingestion.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of embedding vectors
        """
        if not documents:
            return []
        
        return await self.embed_text(documents)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        This is needed when:
        - Creating Pinecone index
        - Validating embedding compatibility
        
        Returns:
            Embedding dimension size
        """
        return self._dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Similarity score ranges from -1 to 1:
        - 1: Identical semantic meaning
        - 0: No relationship
        - -1: Opposite meaning
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        # If vectors are normalized (which ours are), dot product = cosine similarity
        similarity = np.dot(vec1, vec2)
        
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model details
        """
        return {
            "model_name": self.model_name,
            "dimension": self._dimension,
            "base_url": self.base_url
        }


def create_embedding_service(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434"
) -> EmbeddingService:
    """
    Factory function to create an embedding service instance.
    
    Args:
        model_name: Ollama embedding model
        base_url: Ollama API endpoint
        
    Returns:
        Configured EmbeddingService instance
    """
    return EmbeddingService(model_name=model_name, base_url=base_url)
