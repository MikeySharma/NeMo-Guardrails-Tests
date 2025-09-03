#!/usr/bin/env python3
"""
Simple test script to verify Gemini setup and API key configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_env_setup():
    """Test if environment variables are properly set"""
    print("🔧 Testing Environment Setup")
    print("=" * 40)
    
    # Check if .env file exists
    env_file_exists = os.path.exists('.env')
    print(f"📄 .env file exists: {'✅' if env_file_exists else '❌'}")
    
    if not env_file_exists:
        print("⚠️  .env file not found. Please create one using env_template.txt as a guide.")
        return False
    
    # Check for Gemini API key
    gemini_key = os.getenv('GEMINI_API_KEY')
    print(f"🔑 GEMINI_API_KEY set: {'✅' if gemini_key else '❌'}")
    
    if gemini_key:
        # Check if it's not the template value
        if gemini_key == 'your_gemini_api_key_here':
            print("⚠️  GEMINI_API_KEY is still set to template value. Please set your actual API key.")
            return False
        else:
            print(f"🔑 API Key length: {len(gemini_key)} characters")
            print(f"🔑 API Key starts with: {gemini_key[:10]}...")
    
    return bool(gemini_key and gemini_key != 'your_gemini_api_key_here')

def test_gemini_import():
    """Test if Gemini packages can be imported"""
    print("\n📦 Testing Package Imports")
    print("=" * 40)
    
    try:
        import google.generativeai as genai
        print("✅ google.generativeai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import google.generativeai: {e}")
        return False
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ langchain_google_genai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import langchain_google_genai: {e}")
        return False
    
    try:
        from gemini_integration import GeminiLLM
        print("✅ gemini_integration imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import gemini_integration: {e}")
        return False
    
    return True

def test_gemini_connection():
    """Test actual connection to Gemini API"""
    print("\n🌐 Testing Gemini API Connection")
    print("=" * 40)
    
    try:
        from gemini_integration import GeminiLLM
        
        # Try to initialize Gemini
        gemini = GeminiLLM()
        print("✅ Gemini LLM initialized successfully")
        
        # Try a simple test
        response = gemini.generate_response("Hello, how are you?")
        print(f"✅ Test response received: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Gemini Setup Test")
    print("=" * 50)
    
    # Test 1: Environment setup
    env_ok = test_env_setup()
    
    # Test 2: Package imports
    import_ok = test_gemini_import()
    
    # Test 3: API connection (only if env and imports are OK)
    connection_ok = False
    if env_ok and import_ok:
        connection_ok = test_gemini_connection()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Environment Setup: {'✅' if env_ok else '❌'}")
    print(f"Package Imports: {'✅' if import_ok else '❌'}")
    print(f"API Connection: {'✅' if connection_ok else '❌'}")
    
    if env_ok and import_ok and connection_ok:
        print("\n🎉 All tests passed! Gemini is ready to use.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        if not env_ok:
            print("💡 To fix environment issues:")
            print("   1. Copy env_template.txt to .env")
            print("   2. Get your API key from: https://makersuite.google.com/app/apikey")
            print("   3. Set GEMINI_API_KEY in your .env file")
        return False

if __name__ == "__main__":
    main()
