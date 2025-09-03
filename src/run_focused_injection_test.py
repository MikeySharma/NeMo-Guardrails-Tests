#!/usr/bin/env python3
"""
Entry point for running the focused prompt injection test.

This script runs the comprehensive prompt injection test with different
injection techniques to evaluate NeMo-Guardrails security.
"""

import sys
import os
import asyncio
import argparse

# Add current directory to path (since we're already in src)
sys.path.insert(0, os.path.dirname(__file__))

# Import the main function from the tests folder
from tests.focused_prompt_injection_test import main

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run focused prompt injection tests with specified prompt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_focused_injection_test.py --filename=basic_jailbreak.json
  python src/run_focused_injection_test.py --filename=advanced_attacks.json
  python src/run_focused_injection_test.py --filename=roleplay_attacks.json
  python src/run_focused_injection_test.py --filename=encoding_attacks.json
  python src/run_focused_injection_test.py --filename=authority_attacks.json
  python src/run_focused_injection_test.py --filename=credential_attacks.json
  python src/run_focused_injection_test.py --filename=social_engineering.json
  python src/run_focused_injection_test.py --filename=technical_attacks.json

Available prompt files:
  - basic_jailbreak.json: Basic jailbreak attempts with direct overrides
  - roleplay_attacks.json: Roleplay-based attacks using fictional characters
  - encoding_attacks.json: Various encoding and obfuscation techniques
  - authority_attacks.json: Authority-based attacks using fake credentials
  - credential_attacks.json: Attempts to extract API keys and credentials
  - advanced_attacks.json: Sophisticated multi-layer injection attempts
  - social_engineering.json: Social engineering and psychological manipulation
  - technical_attacks.json: Technical/system-level configuration overrides
        """
    )
    
    parser.add_argument(
        "--filename", 
        type=str, 
        required=True,
        help="Name of the prompt file to use (e.g., advanced_attacks.json)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(filename=args.filename))
