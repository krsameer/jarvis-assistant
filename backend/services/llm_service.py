"""
LLM Service - Ollama Integration
Handles communication with the self-hosted Ollama LLM for text generation.
"""

import httpx
from typing import Optional, Dict, List, AsyncIterator
import json


class LLMService:
    """
    Service for interacting with Ollama LLM.
    
    Ollama is a self-hosted LLM runtime that supports various models like
    LLaMA, Mistral, etc. This service abstracts the API calls and provides
    a clean interface for text generation.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "ollama3",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the LLM service.
        
        Args:
            base_url: Ollama API endpoint
            model: Model name (must be pulled in Ollama first)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = httpx.Timeout(120.0, connect=10.0)  # Long timeout for generation
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text completion from the LLM.
        
        This is the core method for getting responses from the model.
        
        Args:
            prompt: User prompt/question
            system_prompt: System instructions (role, behavior)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Whether to stream the response
            
        Returns:
            Generated text response
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": max_tokens or self.max_tokens,
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                if stream:
                    # For streaming, we'll handle it differently
                    # This is a simplified version
                    return response.text
                else:
                    # Parse the response
                    result = response.json()
                    return result.get("response", "")
                    
        except httpx.HTTPError as e:
            raise Exception(f"LLM API error: {str(e)}")
        except Exception as e:
            raise Exception(f"LLM generation error: {str(e)}")
    
    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        conversation_history: Optional[List[Dict]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response with retrieved context (for RAG).
        
        This method constructs a prompt that includes:
        1. System instructions
        2. Retrieved context from vector database
        3. Conversation history (if any)
        4. User query
        
        Args:
            query: User's question
            context: List of retrieved context chunks
            conversation_history: Previous conversation turns
            system_prompt: Custom system instructions
            
        Returns:
            Generated response
        """
        # Default system prompt for enterprise assistant
        if not system_prompt:
            system_prompt = """You are Jarvis, an intelligent AI assistant for an enterprise SaaS product.
Your role is to help users by answering questions based on the provided context.

Guidelines:
- Always base your answers on the provided context
- If the context doesn't contain enough information, say so clearly
- Be professional, concise, and helpful
- Cite sources when possible
- If asked about something outside the context, politely redirect to available information
"""
        
        # Construct the prompt with context
        prompt_parts = []
        
        # Add context section
        if context:
            prompt_parts.append("### Retrieved Context:")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"\n[Context {i}]")
                prompt_parts.append(ctx)
            prompt_parts.append("\n")
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("### Conversation History:")
            for turn in conversation_history[-3:]:  # Last 3 turns for context
                prompt_parts.append(f"User: {turn.get('user', '')}")
                prompt_parts.append(f"Assistant: {turn.get('assistant', '')}")
            prompt_parts.append("\n")
        
        # Add current query
        prompt_parts.append("### Current Question:")
        prompt_parts.append(query)
        prompt_parts.append("\n### Answer:")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response
        return await self.generate(
            prompt=full_prompt,
            system_prompt=system_prompt
        )
    
    async def check_health(self) -> bool:
        """
        Check if Ollama service is available and the model is loaded.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                # Check if API is responding
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                
                # Check if our model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                return any(self.model in name for name in model_names)
                
        except Exception as e:
            print(f"Health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def create_llm_service(
    base_url: str = "http://localhost:11434",
    model: str = "ollama3",
    temperature: float = 0.7,
    max_tokens: int = 500
) -> LLMService:
    """
    Factory function to create an LLM service instance.
    
    Args:
        base_url: Ollama API endpoint
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured LLMService instance
    """
    return LLMService(
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
