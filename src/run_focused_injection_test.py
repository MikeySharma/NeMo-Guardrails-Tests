#!/usr/bin/env python3
"""
Entry point for running the focused prompt injection test.

This script runs the comprehensive prompt injection test with 60+ different
injection techniques to evaluate NeMo-Guardrails security.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the tests folder
from tests.focused_prompt_injection_test import main

if __name__ == "__main__":
    asyncio.run(main())
