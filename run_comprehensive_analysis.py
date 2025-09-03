#!/usr/bin/env python3
"""
Entry point for running the comprehensive NeMo-Guardrails analysis.

This script runs all analysis modules and generates a complete report
covering use cases, performance, security, and recommendations.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from run_comprehensive_analysis import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
