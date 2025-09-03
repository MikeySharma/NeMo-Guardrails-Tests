#!/usr/bin/env python3
"""
Entry point for running performance analysis.

This script runs the performance benchmarks to evaluate system performance.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the analyzers folder
from analyzers.performance_benchmarks import main

if __name__ == "__main__":
    asyncio.run(main())
