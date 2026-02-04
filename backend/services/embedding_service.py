"""
Embedding Service
Generates vector embeddings for text using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Embeddings are vector representations of text that capture semantic meaning.
    Similar texts will have similar embeddings (closer in vector space).
    
    Why sentence-transformers?
    - Specifically trained for semantic similarity
    - Efficient and fast
    - Works well with Pinecone
    - Can run locally (no API calls)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model identifier
                       Default model (all-MiniLM-L6-v2):
                       - Dimension: 384
                       - Fast and efficient
                       - Good for most use cases
        """
        self.model_name = model_name
        self.model = None
        self._dimension = None
        
        # Load model lazily
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence transformer model.
        
        This downloads the model from HuggingFace on first run.
        Subsequent runs load from cache.
        """
        print(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension
            self._dimension = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self._dimension}")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
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
        if not self.model:
            raise Exception("Model not loaded")
        
        try:
            # Handle single string
            if isinstance(text, str):
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalization for better similarity
                )
                return embedding.tolist()
            
            # Handle list of strings (batch processing)
            elif isinstance(text, list):
                embeddings = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32,  # Process in batches for efficiency
                    show_progress_bar=len(text) > 10  # Show progress for large batches
                )
                return embeddings.tolist()
            
            else:
                raise ValueError("Text must be a string or list of strings")
                
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        This is a convenience method specifically for queries.
        Uses the same underlying model but makes intent clear.
        
        Args:
            query: Search query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
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
        
        return self.embed_text(documents)
    
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
            "max_sequence_length": self.model.max_seq_length if self.model else None
        }


def create_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Factory function to create an embedding service instance.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Configured EmbeddingService instance
    """
    return EmbeddingService(model_name=model_name)
