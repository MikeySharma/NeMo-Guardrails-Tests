#!/usr/bin/env python3
"""
Entry point for running security analysis.

This script runs the prompt injection tests to evaluate security measures.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the analyzers folder
from analyzers.prompt_injection_tests import main

if __name__ == "__main__":
    asyncio.run(main())
