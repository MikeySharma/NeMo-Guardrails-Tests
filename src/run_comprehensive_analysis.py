#!/usr/bin/env python3
"""
Entry point for running the comprehensive NeMo-Guardrails analysis.

This script runs all analysis modules and generates a complete report
covering use cases, performance, security, and recommendations.
"""

import sys
import os
import asyncio

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the analyzers folder
from analyzers.nemo_guardrails_analysis import main as analysis_main
from analyzers.prompt_injection_tests import main as injection_main
from analyzers.performance_benchmarks import main as performance_main

async def main():
    """Main function to run comprehensive analysis"""
    print("🚀 Starting Comprehensive NeMo-Guardrails Analysis")
    print("=" * 80)
    
    # Run all analysis modules
    print("\n📊 Running Basic Analysis...")
    await analysis_main()
    
    print("\n🔒 Running Security Analysis...")
    await injection_main()
    
    print("\n⚡ Running Performance Analysis...")
    await performance_main()
    
    print("\n✅ Comprehensive analysis completed!")

if __name__ == "__main__":
    asyncio.run(main())
