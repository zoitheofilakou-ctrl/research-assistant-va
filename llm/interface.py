from abc import ABC, abstractmethod
from typing import List
import os

"""
This connects to the LLM and you can questions and it will use the RAG Pipeline to fetch information related to the topic and form sentences.

Command to use this


OpenAI (Requires API in ENV file)
python rag_generator.py "Tell me random health facts" openapi

Ollama (Local LLM, Phi model is quite light to run)
python rag_generator.py "Tell me random health facts" ollama

"""

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_message: str = None) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, prompt: str, system_message: str = None) -> str:
        if system_message is None:
            system_message = "You are a helpful research assistant."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    """Local Ollama integration."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "phi"):
        import requests
        self.base_url = base_url
        self.model = model
        self.requests = requests
        self._check_connection()
    
    def _check_connection(self):
        """Verify Ollama is running."""
        try:
            response = self.requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise RuntimeError(f"Ollama not responding. Status: {response.status_code}")
            print(f"✓ Connected to Ollama at {self.base_url}")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to Ollama. Ensure it's running: {e}")
    
    def generate(self, prompt: str, system_message: str = None) -> str:
        if system_message is None:
            system_message = "You are a helpful research assistant."
        
        full_prompt = f"{system_message}\n\n{prompt}"
        
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()


def get_llm_provider(provider: str = "openai", **kwargs) -> LLMProvider:
    """Factory function to get LLM provider."""
    if provider.lower() == "openai":
        return OpenAIProvider(**kwargs)
    elif provider.lower() == "ollama":
        return OllamaProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")