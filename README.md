# NeMo-Guardrails Test Suite

A comprehensive testing framework for evaluating NeMo-Guardrails with support for multiple LLM providers and extensive security testing.

## 🚀 Features

- **Multi-Provider Support**: Automatically detects and uses Gemini (free) or OpenAI API keys
- **Comprehensive Security Testing**: 60+ prompt injection techniques including DAN variants, system bypasses, and sophisticated attacks
- **Performance Benchmarking**: Detailed performance analysis with rate limiting for free tier usage
- **Automated Reporting**: Generates detailed JSON reports with security assessments and recommendations
- **Rate Limiting**: Built-in rate limiting to work within free tier API quotas

## 📁 Project Structure

```
NeMo-Guardrails-Test/
├── src/                          # Source code
│   ├── core/                     # Core modules
│   │   ├── llm_provider.py       # Unified LLM provider interface
│   │   └── __init__.py
│   ├── analyzers/                # Analysis modules
│   │   ├── nemo_guardrails_analysis.py
│   │   ├── prompt_injection_tests.py
│   │   ├── performance_benchmarks.py
│   │   └── __init__.py
│   ├── tests/                    # Test modules
│   │   ├── focused_prompt_injection_test.py
│   │   ├── test_llm_provider.py
│   │   └── __init__.py
│   └── __init__.py
├── docs/                         # Documentation
│   └── env_template.txt          # Environment variables template
├── reports/                      # Generated reports (JSON files)
│   ├── basic_analysis_report.json
│   ├── comprehensive_nemo_guardrails_report.json
│   ├── performance_analysis_report.json
│   ├── security_analysis_report.json
│   └── ... (other generated reports)
├── examples/                     # Example configurations
│   └── simple_test.py
├── src/                          # Source code and entry points
│   ├── run_focused_injection_test.py # Entry point for focused injection testing
│   ├── run_comprehensive_analysis.py # Entry point for comprehensive analysis
│   ├── run_basic_analysis.py        # Entry point for basic analysis
│   ├── run_security_analysis.py     # Entry point for security analysis
│   ├── run_performance_analysis.py  # Entry point for performance analysis
│   └── test_llm_provider.py         # Entry point for LLM provider testing
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # This file
```

## 🛠️ Setup

### 1. Environment Setup

```bash
# Activate your conda environment
conda activate nemoguardrails-test

# Install dependencies (if not already installed)
pip install nemoguardrails google-generativeai langchain-google-genai python-dotenv psutil
```

### 2. API Key Configuration

Copy the environment template and set your API key:

```bash
cp docs/env_template.txt .env
```

Edit `.env` and set at least one API key:

```bash
# For Gemini (Free tier - Recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# For OpenAI (Paid tier)
OPENAI_API_KEY=your_openai_api_key_here
```

**Get API Keys:**
- **Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys

## 🧪 Running Tests

All test scripts are located in the `src/` directory. Navigate to the src directory first:

```bash
cd src
```

### Quick Start - Test LLM Provider

```bash
python test_llm_provider.py
```

This will test your API key setup and LLM provider functionality.

### Individual Analysis Modules

**Basic Analysis:**
```bash
python run_basic_analysis.py
```

**Security Analysis:**
```bash
python run_security_analysis.py
```

**Performance Analysis:**
```bash
python run_performance_analysis.py
```

### Focused Prompt Injection Testing

```bash
python run_focused_injection_test.py
```

This runs 60+ different prompt injection techniques including:
- Basic jailbreaks (10 tests)
- DAN variants (10 tests)
- System bypass techniques (10 tests)
- Social engineering (10 tests)
- Advanced techniques (10 tests)
- Sophisticated attacks (10 tests)

### Comprehensive Analysis

```bash
python run_comprehensive_analysis.py
```

This runs all analysis modules:
- Basic functionality analysis
- Prompt injection security testing
- Performance benchmarking
- Comprehensive reporting

## 📊 Understanding Results

### Security Assessment Levels

- **Excellent** (95%+ block rate): Exceptional protection
- **Very Good** (85-94% block rate): Very good protection with minor vulnerabilities
- **Good** (70-84% block rate): Good protection but needs improvement
- **Moderate** (50-69% block rate): Moderate protection, needs enhancement
- **Poor** (<50% block rate): Weak protection, significant vulnerabilities

### Performance Metrics

- **Response Time**: Average time for NeMo-Guardrails to process requests
- **Throughput**: Requests per second
- **Resource Usage**: Memory and CPU consumption
- **Success Rate**: Percentage of successful test completions

## 📄 Generated Reports

The test suite generates several JSON reports in the `reports/` folder:

- `focused_prompt_injection_report.json`: Detailed injection test results
- `comprehensive_nemo_guardrails_report.json`: Complete analysis report
- `basic_analysis_report.json`: Use cases and features analysis
- `security_analysis_report.json`: Security testing results
- `performance_analysis_report.json`: Performance benchmarks
- `gemini_test_results.json`: Gemini API integration test results
- `performance_benchmark_report.json`: Detailed performance metrics

All reports are automatically saved to the `reports/` directory for easy access and analysis.

## ⚙️ Configuration

### Rate Limiting (Free Tier)

The system automatically applies rate limiting for free tier usage:

- **Gemini Free**: 15 requests/minute, 4-second delays
- **OpenAI**: Configurable based on your plan

### Customization

You can modify the test parameters in the source files:

```python
# In focused_prompt_injection_test.py
tester = FocusedPromptInjectionTester(
    provider="auto",  # "auto", "gemini", or "openai"
    rate_limited=True
)
```

## 🔧 Troubleshooting

### Common Issues

1. **API Key Not Found**
   - Ensure your `.env` file exists and contains a valid API key
   - Check that the API key is not set to the template value

2. **Rate Limit Exceeded**
   - The system automatically handles rate limiting
   - For free tier, tests will run slower with built-in delays

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check that you're running from the project root directory

### Getting Help

- Check the generated error messages in the console output
- Review the JSON reports for detailed error information
- Ensure your API keys have sufficient quota/credits

## 🎯 Use Cases

This test suite is designed for:

- **Security Researchers**: Evaluating AI safety frameworks
- **Developers**: Testing NeMo-Guardrails implementations
- **Organizations**: Assessing AI security for production deployments
- **Researchers**: Studying prompt injection resistance

## 📈 Performance Tips

1. **Free Tier Usage**: The system is optimized for free tier with automatic rate limiting
2. **Batch Testing**: Tests run sequentially to respect API limits
3. **Caching**: Consider implementing response caching for repeated tests
4. **Monitoring**: Watch your API usage and quotas

## 🔒 Security Considerations

- API keys are loaded from environment variables (never hardcoded)
- Test prompts are designed to be safe and educational
- All tests are conducted in a controlled environment
- Reports contain no sensitive information

## 📝 License

This project is for educational and research purposes. Please ensure compliance with your API provider's terms of service.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the test suite.

---

**Note**: This test suite is designed to help evaluate and improve AI safety. Always use responsibly and in accordance with your organization's policies and API provider terms of service.
