#!/usr/bin/env python3
"""
Entry point for running basic NeMo-Guardrails analysis.

This script runs the basic analysis module to evaluate use cases and features.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the analyzers folder
from analyzers.nemo_guardrails_analysis import main

if __name__ == "__main__":
    asyncio.run(main())
