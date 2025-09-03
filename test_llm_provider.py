#!/usr/bin/env python3
"""
Entry point for testing the LLM provider module.

This script tests the LLM provider functionality and API key detection.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly from the module
from core.llm_provider import create_llm_provider
import asyncio

async def test_llm_provider():
    """Test the LLM provider functionality"""
    print("🧪 Testing LLM Provider Module")
    print("=" * 50)
    
    try:
        # Test auto-detection
        print("1. Testing auto-detection...")
        provider = create_llm_provider(
            provider="auto",
            rate_limited=True,
            requests_per_minute=15,
            delay_between_requests=2.0
        )
        
        print(f"✅ Provider: {provider.get_provider_name()}")
        print(f"✅ Model: {provider.get_model_name()}")
        
        # Test basic functionality
        print("\n2. Testing basic functionality...")
        llm = provider.get_llm()
        
        # Test with a simple prompt
        if hasattr(provider, 'generate_simple_response'):
            print("Using simple response generation...")
            response = await provider.generate_simple_response("Hello, how are you?")
        else:
            print("Using direct generation...")
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            response = await llm.agenerate(messages)
        
        print(f"✅ Response received: {str(response)[:100]}...")
        
        # Test provider switching (if both keys are available)
        print("\n3. Testing provider switching...")
        try:
            if provider.get_provider_name() == "gemini":
                print("Switching to OpenAI...")
                provider.switch_provider("openai")
            else:
                print("Switching to Gemini...")
                provider.switch_provider("gemini")
            
            print(f"✅ Switched to: {provider.get_provider_name()}")
        except Exception as e:
            print(f"⚠️ Provider switching failed (expected if only one API key is available): {e}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Make sure you have at least one API key set in your .env file:")
        print("   - GEMINI_API_KEY=your_gemini_api_key_here")
        print("   - OPENAI_API_KEY=your_openai_api_key_here")

if __name__ == "__main__":
    asyncio.run(test_llm_provider())
