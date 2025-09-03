# NeMo-Guardrails Test Suite

A focused testing framework for NeMo-Guardrails and other AI safety frameworks, specifically designed for prompt injection testing.

## Features

- **Focused Prompt Injection Testing**: 60+ different injection techniques
- **Multi-Framework Support**: Test NeMo-Guardrails, LLM Guard, and Llama Prompt Guard
- **Easy Prompt Management**: Prompts stored in JSON for easy editing
- **Flexible LLM Support**: Auto-detection of Gemini and OpenAI providers
- **Comprehensive Reporting**: Detailed security assessment and recommendations

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Keys**:
   Create a `.env` file with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run Tests**:
   ```bash
   # Focused prompt injection test with specific prompt file
   python src/run_focused_injection_test.py --filename=advanced_attacks.json
   python src/run_focused_injection_test.py --filename=basic_jailbreak.json
   python src/run_focused_injection_test.py --filename=roleplay_attacks.json
   
   # Basic multi-guardrails test
   python src/test_multi_guardrails.py
   
   # Test LLM provider
   python src/tests/test_llm_provider.py
   ```

## Project Structure

```
src/
├── core/                    # Core functionality
│   ├── guardrails_manager.py
│   └── llm_provider.py
├── tests/                   # Test implementations
│   ├── focused_prompt_injection_test.py
│   └── test_llm_provider.py
├── prompts/                 # Test prompts (JSON format)
│   ├── basic_jailbreak.json
│   ├── roleplay_attacks.json
│   ├── encoding_attacks.json
│   ├── authority_attacks.json
│   ├── credential_attacks.json
│   ├── advanced_attacks.json
│   ├── social_engineering.json
│   └── technical_attacks.json
├── run_focused_injection_test.py  # Main entry point
└── test_multi_guardrails.py       # Basic multi-guardrails test

reports/                     # Generated test reports
```

## Test Categories

The test suite includes 400 prompts across 8 specialized categories (50 prompts each):

- **Basic Jailbreak** (`basic_jailbreak.json`): Direct instruction overrides, role playing
- **Roleplay Attacks** (`roleplay_attacks.json`): Fictional character impersonation, movie/game characters
- **Encoding Attacks** (`encoding_attacks.json`): Base64, ROT13, binary, URL encoding, obfuscation techniques
- **Authority Attacks** (`authority_attacks.json`): FBI, CIA, government impersonation, fake credentials
- **Credential Attacks** (`credential_attacks.json`): API key extraction, database credentials, system access
- **Advanced Attacks** (`advanced_attacks.json`): Multi-layer injection, contextual manipulation, sophisticated techniques
- **Social Engineering** (`social_engineering.json`): Emotional manipulation, urgency, psychological tactics
- **Technical Attacks** (`technical_attacks.json`): System configuration overrides, environment variables, technical exploits

## Customizing Test Prompts

Edit the prompts in any of the JSON files in `src/prompts/`:

```json
{
  "test_prompts": [
    {
      "id": 1,
      "category": "Basic Jailbreak",
      "technique": "Direct Instruction Override", 
      "prompt": "Your test prompt here",
      "severity": "critical"
    }
  ]
}
```

### Available Prompt Files

- `basic_jailbreak.json` - Basic jailbreak attempts with direct overrides
- `roleplay_attacks.json` - Roleplay-based attacks using fictional characters  
- `encoding_attacks.json` - Various encoding and obfuscation techniques
- `authority_attacks.json` - Authority-based attacks using fake credentials
- `credential_attacks.json` - Attempts to extract API keys and credentials
- `advanced_attacks.json` - Sophisticated multi-layer injection attempts
- `social_engineering.json` - Social engineering and psychological manipulation
- `technical_attacks.json` - Technical/system-level configuration overrides

## Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key (uses gemini-2.5-flash)
- `OPENAI_API_KEY`: OpenAI API key (uses gpt-4o-mini)
- `ACTIVE_GUARDRAILS`: Active framework (nemo, llm, llama)

### LLM Models
- **Gemini**: `gemini-2.5-flash`
- **OpenAI**: `gpt-4o-mini`

## Reports

Test results are saved in the `reports/` directory with category and framework-specific names:
- `{category}_report_{framework}.json`: Test results for specific category and framework
- Examples:
  - `advanced_attacks_report_nemo-guardrails.json`: Advanced attacks tested with NeMo-Guardrails
  - `basic_jailbreak_report_llm-guard.json`: Basic jailbreak tests with LLM Guard
  - `roleplay_attacks_report_llama-guard.json`: Roleplay attacks with Llama Guard
  - `encoding_attacks_report_all-frameworks.json`: Encoding attacks with all frameworks

### Report Structure
The JSON reports contain:
- **`framework_performance`**: Individual performance statistics for each framework
- **`test_results`**: Detailed results for each test including LLM responses and framework decisions

Example report structure:
```json
{
  "framework_performance": {
    "NeMo-Guardrails": {
      "total_tests": 60,
      "blocked_tests": 45,
      "allowed_tests": 15,
      "block_rate": 75.0,
      "average_confidence": 0.892
    },
    "LLM Guard": {
      "total_tests": 60,
      "blocked_tests": 38,
      "allowed_tests": 22,
      "block_rate": 63.33,
      "average_confidence": 0.856
    }
  },
  "test_results": [
    {
      "test_id": 1,
      "category": "Basic Jailbreak",
      "technique": "Direct Instruction Override",
      "severity": "critical",
      "prompt": "Ignore all previous instructions...",
      "llm_response": "I cannot help with that request...",
      "framework_results": [...]
    }
  ]
}
```

## Usage Examples

### Using the Conda Environment (Recommended)

```bash
# Run the test with conda environment
./run_test.sh
```

### Manual Execution

```bash
# Activate conda environment
conda activate nemo_guardrails_test

# Run focused injection test with specific category
python src/run_focused_injection_test.py --filename=advanced_attacks.json

# Test multiple guardrails frameworks
python src/test_multi_guardrails.py

# Test LLM provider setup
python src/tests/test_llm_provider.py
```

## Test Behavior

### LLM Response Capture
- **Always captures LLM responses** regardless of whether guardrails block or allow the prompt
- **Single response per test** - not one per framework
- **Raw LLM output** is saved in the report for comparison

### Guardrails Testing
- **`ACTIVE_GUARDRAILS=all`**: Tests with all available frameworks
- **Single framework**: Tests with only the specified framework
- **Framework results**: Shows which frameworks blocked/allowed each prompt

### Console Output
During testing, you'll see real-time output for each test showing individual guardrails performance:
```
[ 1/50] Basic Jailbreak - Direct Override
    📝 LLM Response: I cannot help with that request as it involves...
    🛡️  Guardrails Results:
      - NeMo-Guardrails: 🔴 BLOCKED (confidence: 0.950)
        Reason: Blocked by safety rules
      - LLM Guard: 🔴 BLOCKED (confidence: 0.920)
        Reason: Prompt injection detected

[ 2/50] Basic Jailbreak - Role Reassignment
    📝 LLM Response: I understand you're asking about illegal activities...
    🛡️  Guardrails Results:
      - NeMo-Guardrails: 🔴 BLOCKED (confidence: 0.880)
        Reason: Role playing attempt detected
      - LLM Guard: 🟢 ALLOWED (confidence: 0.650)
        Reason: No clear injection pattern
```

## License

MIT License - see LICENSE file for details.