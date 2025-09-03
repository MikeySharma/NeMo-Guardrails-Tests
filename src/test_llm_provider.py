#!/usr/bin/env python3
"""
Entry point for testing the LLM provider module.

This script tests the LLM provider functionality and API key detection.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the test function from the tests folder
from tests.test_llm_provider import test_llm_provider

if __name__ == "__main__":
    asyncio.run(test_llm_provider())
