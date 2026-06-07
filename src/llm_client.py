"""
LLM Client - Unified wrapper for OpenAI and Anthropic APIs.
"""

import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import json

# Try imports, handle gracefully if not installed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class Message:
    """A chat message."""
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Standardized response from LLM APIs."""
    content: str
    model: str
    usage: Dict[str, int]
    raw: Dict[str, Any]
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)
    
    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)
    
    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


class OpenAIClient:
    """Client for OpenAI API (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
    
    def chat(
        self,
        messages: List[Union[Dict[str, str], Message]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts or Message objects
            temperature: Sampling temperature (0-2)
            max_tokens: Max completion tokens
            **kwargs: Additional OpenAI parameters
        
        Returns:
            LLMResponse with standardized fields
        """
        # Normalize messages
        normalized = []
        for m in messages:
            if isinstance(m, Message):
                normalized.append(m.to_dict())
            else:
                normalized.append(m)
        
        params = {
            "model": self.model,
            "messages": normalized,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**params)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            raw=response.model_dump()
        )
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Simple prompt completion (wraps chat)."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


class AnthropicClient:
    """Client for Anthropic API (Claude 3, etc.)."""
    
    MODELS = {
        "claude-opus": "claude-opus-4-20240229",
        "claude-sonnet": "claude-sonnet-4-20240229",
        "claude-haiku": "claude-haiku-3-20240307",
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = self.MODELS.get(model, model)
    
    def chat(
        self,
        messages: List[Union[Dict[str, str], Message]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Send a message to Claude.
        
        Args:
            messages: List of message dicts or Message objects
            temperature: Sampling temperature (0-1)
            max_tokens: Max tokens to generate
            **kwargs: Additional Anthropic parameters
        
        Returns:
            LLMResponse with standardized fields
        """
        # Normalize and filter system messages
        system = None
        normalized = []
        
        for m in messages:
            if isinstance(m, Message):
                d = m.to_dict()
            else:
                d = m
            
            if d["role"] == "system":
                system = d["content"]
            else:
                normalized.append(d)
        
        params = {
            "model": self.model,
            "messages": normalized,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if system:
            params["system"] = system
        
        response = self.client.messages.create(**params)
        
        return LLMResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            raw=response.model_dump()
        )
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Simple prompt completion."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)


# Factory function
def get_client(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Union[OpenAIClient, AnthropicClient]:
    """
    Get an LLM client by provider name.
    
    Args:
        provider: "openai" or "anthropic"
        api_key: API key (or set via env var)
        model: Model name/alias
        **kwargs: Additional client params
    
    Returns:
        LLM client instance
    """
    if provider.lower() == "openai":
        return OpenAIClient(api_key=api_key, model=model or "gpt-4")
    elif provider.lower() in ("anthropic", "claude"):
        return AnthropicClient(api_key=api_key, model=model or "claude-sonnet")
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")