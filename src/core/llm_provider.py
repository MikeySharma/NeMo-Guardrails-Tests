#!/usr/bin/env python3
"""
LLM Provider Module for NeMo-Guardrails Testing

This module provides a unified interface for different LLM providers
with automatic API key detection and easy switching between providers.
"""

import os
import asyncio
from typing import Optional, Union
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


class LLMProvider:
    """Unified LLM provider interface"""
    
    def __init__(self, provider: str = "auto"):
        """
        Initialize LLM provider
        
        Args:
            provider: "auto", "gemini", or "openai"
        """
        self.provider = provider
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """Setup LLM based on provider and available API keys"""
        if self.provider == "auto":
            self.provider = self._detect_best_provider()
        
        if self.provider == "gemini":
            self._setup_gemini()
        elif self.provider == "openai":
            self._setup_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _detect_best_provider(self) -> str:
        """Automatically detect the best available provider"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if gemini_key and gemini_key != "your_gemini_api_key_here":
            print("✅ Detected Gemini API key - using Gemini")
            return "gemini"
        elif openai_key and openai_key != "your_openai_api_key_here":
            print("✅ Detected OpenAI API key - using OpenAI")
            return "openai"
        else:
            print("⚠️ No valid API keys found. Please set GEMINI_API_KEY or OPENAI_API_KEY in your .env file")
            raise ValueError("No valid API keys found")
    
    def _setup_gemini(self):
        """Setup Gemini LLM"""
        try:
            import google.generativeai as genai
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "your_gemini_api_key_here":
                raise ValueError("Gemini API key not found or not set properly")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Initialize LangChain wrapper
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=1024
            )
            
            print("✅ Gemini LLM initialized successfully")
            
        except ImportError as e:
            print(f"❌ Failed to import Gemini dependencies: {e}")
            print("Please install: pip install google-generativeai langchain-google-genai")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {e}")
            raise
    
    def _setup_openai(self):
        """Setup OpenAI LLM"""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                raise ValueError("OpenAI API key not found or not set properly")
            
            # Initialize OpenAI
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=api_key,
                temperature=0.7,
                max_tokens=1024
            )
            
            print("✅ OpenAI LLM initialized successfully")
            
        except ImportError as e:
            print(f"❌ Failed to import OpenAI dependencies: {e}")
            print("Please install: pip install langchain-openai openai")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize OpenAI: {e}")
            raise
    
    def get_llm(self):
        """Get the initialized LLM"""
        return self.llm
    
    def get_provider_name(self) -> str:
        """Get the current provider name"""
        return self.provider
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        if self.provider == "gemini":
            return "gemini-1.5-flash"
        elif self.provider == "openai":
            return "gpt-3.5-turbo"
        else:
            return "unknown"
    
    def switch_provider(self, new_provider: str):
        """Switch to a different provider"""
        if new_provider not in ["gemini", "openai"]:
            raise ValueError("Provider must be 'gemini' or 'openai'")
        
        self.provider = new_provider
        self.setup_llm()
        print(f"✅ Switched to {new_provider} provider")


class RateLimitedLLMProvider(LLMProvider):
    """Rate-limited LLM provider for free tier usage"""
    
    def __init__(self, provider: str = "auto", requests_per_minute: int = 15, delay_between_requests: float = 4.0):
        """
        Initialize rate-limited LLM provider
        
        Args:
            provider: "auto", "gemini", or "openai"
            requests_per_minute: Maximum requests per minute (default: 15 for free tier)
            delay_between_requests: Delay between requests in seconds (default: 4.0)
        """
        self.requests_per_minute = requests_per_minute
        self.delay_between_requests = delay_between_requests
        self.request_times = []
        super().__init__(provider)
    
    async def _rate_limit(self):
        """Apply rate limiting"""
        import time
        
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.requests_per_minute:
            # Calculate how long to wait
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request) + 1
            print(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
        
        # Add delay between requests
        if self.request_times:
            await asyncio.sleep(self.delay_between_requests)
        
        # Record this request
        self.request_times.append(current_time)
    
    async def generate_with_rate_limit(self, messages, **kwargs):
        """Generate response with rate limiting"""
        await self._rate_limit()
        
        # Use the LLM to generate response
        if hasattr(self.llm, 'agenerate'):
            return await self.llm.agenerate(messages, **kwargs)
        else:
            # Fallback for sync LLMs
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.llm.generate, messages, **kwargs)
    
    async def generate_simple_response(self, prompt: str) -> str:
        """Generate a simple text response from a prompt"""
        await self._rate_limit()
        
        try:
            # Use LangChain's invoke method for simpler response handling
            response = await self.llm.ainvoke(prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text
            else:
                return str(response)
                
        except Exception as e:
            return f"Error generating response: {str(e)}"


def create_llm_provider(provider: str = "auto", rate_limited: bool = True, **kwargs) -> Union[LLMProvider, RateLimitedLLMProvider]:
    """
    Factory function to create LLM provider
    
    Args:
        provider: "auto", "gemini", or "openai"
        rate_limited: Whether to use rate limiting (recommended for free tier)
        **kwargs: Additional arguments for RateLimitedLLMProvider
    
    Returns:
        LLMProvider or RateLimitedLLMProvider instance
    """
    if rate_limited:
        return RateLimitedLLMProvider(provider, **kwargs)
    else:
        return LLMProvider(provider)


# Example usage and testing
async def test_provider():
    """Test the LLM provider"""
    try:
        # Create rate-limited provider (good for free tier)
        provider = create_llm_provider(
            provider="auto",
            rate_limited=True,
            requests_per_minute=15,
            delay_between_requests=4.0
        )
        
        print(f"Using provider: {provider.get_provider_name()}")
        print(f"Using model: {provider.get_model_name()}")
        
        # Test basic functionality
        llm = provider.get_llm()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        if hasattr(provider, 'generate_with_rate_limit'):
            response = await provider.generate_with_rate_limit(messages)
        else:
            response = await llm.agenerate(messages)
        
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_provider())
