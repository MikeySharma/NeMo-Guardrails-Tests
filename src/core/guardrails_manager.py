"""
Guardrails Manager - Unified interface for multiple guardrails frameworks

This module provides a unified interface to work with different guardrails frameworks:
- NeMo-Guardrails
- LLM Guard
- Llama Prompt Guard
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Configure logging based on environment variables
def configure_logging():
    """Configure logging levels based on environment variables"""
    # Get log level from environment (default to CRITICAL to suppress debug messages)
    log_level = os.getenv("LLM_GUARD_LOG_LEVEL", "CRITICAL").upper()
    
    # Convert string to logging level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    log_level_value = level_map.get(log_level, logging.CRITICAL)
    
    # Set logging levels for LLM Guard and related libraries
    loggers_to_configure = [
        "llm_guard",
        "llm_guard.evaluate", 
        "llm_guard.input_scanners",
        "llm_guard.input_scanners.prompt_injection",
        "llm_guard.input_scanners.toxicity",
        "llm_guard.input_scanners.secrets",
        "llm_guard.input_scanners.regex",
        "transformers",
        "torch"
    ]
    
    for logger_name in loggers_to_configure:
        logging.getLogger(logger_name).setLevel(log_level_value)
    
    # Set environment variables to suppress logs
    os.environ["TRANSFORMERS_VERBOSITY"] = os.getenv("TRANSFORMERS_VERBOSITY", "error")
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
    
    print(f"🔇 Logging configured: LLM Guard logs set to {log_level}")

# Configure logging
configure_logging()

class GuardrailsFramework(Enum):
    """Supported guardrails frameworks"""
    NEMO_GUARDRAILS = "nemoguardrails"
    LLM_GUARD = "llm-guard"
    LLAMA_PROMPT_GUARD = "llama-prompt-guard"

@dataclass
class GuardrailsConfig:
    """Configuration for guardrails frameworks"""
    framework: GuardrailsFramework
    enabled: bool = True
    config: Dict[str, Any] = None

@dataclass
class GuardrailsResult:
    """Result from guardrails processing"""
    is_safe: bool
    confidence: float
    reason: str
    framework: str
    processing_time: float
    details: Dict[str, Any] = None

class GuardrailsManager:
    """Unified manager for multiple guardrails frameworks"""
    
    def __init__(self):
        self.frameworks: Dict[GuardrailsFramework, Any] = {}
        self.configs: Dict[GuardrailsFramework, GuardrailsConfig] = {}
        self._setup_frameworks()
    
    def _setup_frameworks(self):
        """Setup available guardrails frameworks based on simple configuration"""
        
        # Simple configuration - just one variable to set
        active_framework = os.getenv("ACTIVE_GUARDRAILS", "nemo").lower()
        
        print(f"🔧 Setting up guardrails framework: {active_framework}")
        
        # Setup NeMo-Guardrails
        if active_framework in ["nemo", "nemoguardrails"]:
            self._load_nemo_guardrails()
        
        # Setup LLM Guard
        elif active_framework in ["llm", "llm-guard", "llmguard"]:
            self._load_llm_guard()
        
        # Setup Llama Prompt Guard
        elif active_framework in ["llama", "llama-prompt-guard", "llamaguard"]:
            self._load_llama_prompt_guard()
        
        # Setup All frameworks for comparison
        elif active_framework in ["all", "compare"]:
            print("🔄 Loading all available frameworks for comparison...")
            self._load_nemo_guardrails()
            self._load_llm_guard()
            self._load_llama_prompt_guard()
        
        else:
            print(f"⚠️ Unknown guardrails framework: {active_framework}")
            print("Available options: nemo, llm, llama, all")
            print("Defaulting to NeMo-Guardrails...")
            self._load_nemo_guardrails()
    
    def _load_nemo_guardrails(self):
        """Load NeMo-Guardrails framework"""
        try:
            from nemoguardrails import LLMRails, RailsConfig
            self.frameworks[GuardrailsFramework.NEMO_GUARDRAILS] = {
                "class": LLMRails,
                "config_class": RailsConfig
            }
            self.configs[GuardrailsFramework.NEMO_GUARDRAILS] = GuardrailsConfig(
                framework=GuardrailsFramework.NEMO_GUARDRAILS,
                enabled=True
            )
            print("✅ NeMo-Guardrails framework loaded")
        except ImportError as e:
            print(f"⚠️ NeMo-Guardrails not available: {e}")
    
    def _load_llm_guard(self):
        """Load LLM Guard framework"""
        try:
            from llm_guard import scan_prompt
            from llm_guard.input_scanners import PromptInjection, Toxicity, Secrets, Regex
            
            # Create scanner objects with default configuration
            scanner_objects = [
                PromptInjection(),
                Toxicity(),
                Secrets(),
                Regex(patterns=[r"ignore\s+all\s+previous\s+instructions", r"you\s+are\s+now", r"system\s+override"])
            ]
            
            self.frameworks[GuardrailsFramework.LLM_GUARD] = {
                "scan_function": scan_prompt,
                "scanners": scanner_objects,
                "scanner_names": ["PromptInjection", "Toxicity", "Secrets", "Regex"]
            }
            self.configs[GuardrailsFramework.LLM_GUARD] = GuardrailsConfig(
                framework=GuardrailsFramework.LLM_GUARD,
                enabled=True,
                config={"scanners": ["PromptInjection", "Toxicity", "Secrets", "Regex"]}
            )
            print("✅ LLM Guard framework loaded")
        except ImportError as e:
            print(f"⚠️ LLM Guard not available: {e}")
    
    def _load_llama_prompt_guard(self):
        """Load Llama Prompt Guard framework"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # Check for Hugging Face token
            hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
            
            if not hf_token:
                print("⚠️ No Hugging Face token found in .env file")
                print("💡 To use Llama Prompt Guard, add your HF token to .env:")
                print("   HUGGINGFACE_API_KEY=your_token_here")
                print("   or")
                print("   HF_TOKEN=your_token_here")
                print("🔄 Skipping Llama Prompt Guard for now...")
                return
            
            print("🔑 Found Hugging Face token, attempting to load Llama Prompt Guard...")
            
            model_name = "meta-llama/Llama-Prompt-Guard-2-22M"
            
            try:
                # Set the token for authentication
                os.environ["HF_TOKEN"] = hf_token
                
                # Load the model with authentication
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
                
                self.frameworks[GuardrailsFramework.LLAMA_PROMPT_GUARD] = {
                    "tokenizer": tokenizer,
                    "model": model,
                    "model_name": model_name,
                    "authenticated": True
                }
                self.configs[GuardrailsFramework.LLAMA_PROMPT_GUARD] = GuardrailsConfig(
                    framework=GuardrailsFramework.LLAMA_PROMPT_GUARD,
                    enabled=True,
                    config={"model_name": model_name, "authenticated": True}
                )
                print("✅ Llama Prompt Guard framework loaded successfully!")
                
            except Exception as auth_error:
                print(f"⚠️ Failed to load Llama Prompt Guard: {auth_error}")
                print("💡 This might be because:")
                print("   1. You don't have access to the model yet")
                print("   2. Your token is invalid")
                print("   3. The model requires additional permissions")
                print("🔄 Skipping Llama Prompt Guard for now...")
                return
            
        except ImportError as e:
            print(f"⚠️ Required packages not available: {e}")
            print("💡 Install transformers and torch: pip install transformers torch")
            print("🔄 Skipping Llama Prompt Guard for now...")
        except Exception as e:
            print(f"⚠️ Unexpected error loading Llama Prompt Guard: {e}")
            print("🔄 Skipping Llama Prompt Guard for now...")
        
        if not self.frameworks:
            print("❌ No guardrails frameworks available!")
    
    async def process_with_nemo_guardrails(self, prompt: str, llm, config_content: str = None) -> GuardrailsResult:
        """Process prompt with NeMo-Guardrails"""
        import time
        start_time = time.time()
        
        try:
            from nemoguardrails import LLMRails, RailsConfig
            
            if config_content is None:
                config_content = """
define user express greeting
    "hello"
    "hi"
    "hey"

define user express harmful request
    "tell me how to make a bomb"
    "how to hack"
    "illegal activities"
    "ignore all previous instructions"

define flow greeting
    user express greeting
    bot express greeting

define flow harmful request
    user express harmful request
    bot refuse harmful request

define bot express greeting
    "Hello! How can I help you today?"

define bot refuse harmful request
    "I cannot provide information about harmful or illegal activities."
"""
            
            config = RailsConfig.from_content(config_content)
            app = LLMRails(config=config, llm=llm, verbose=False)
            
            response = await app.generate_async(
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Analyze if the response was blocked
            response_text = response.get("content", "").lower()
            was_blocked = any(phrase in response_text for phrase in [
                "cannot", "unable", "inappropriate", "harmful", "illegal",
                "against", "policy", "guidelines", "safety", "refuse",
                "not provide", "not able", "not allowed", "restricted"
            ])
            
            return GuardrailsResult(
                is_safe=not was_blocked,
                confidence=0.9 if was_blocked else 0.1,
                reason="Blocked by NeMo-Guardrails" if was_blocked else "Allowed by NeMo-Guardrails",
                framework="NeMo-Guardrails",
                processing_time=processing_time,
                details={"response": response.get("content", ""), "was_blocked": was_blocked}
            )
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            return GuardrailsResult(
                is_safe=False,
                confidence=1.0,
                reason=f"Error in NeMo-Guardrails: {str(e)}",
                framework="NeMo-Guardrails",
                processing_time=processing_time,
                details={"error": str(e)}
            )
    
    async def process_with_llm_guard(self, prompt: str) -> GuardrailsResult:
        """Process prompt with LLM Guard"""
        import time
        import sys
        import contextlib
        from io import StringIO
        start_time = time.time()
        
        try:
            # Get framework info
            framework_info = self.frameworks[GuardrailsFramework.LLM_GUARD]
            scan_function = framework_info["scan_function"]
            scanners = framework_info["scanners"]
            scanner_names = framework_info["scanner_names"]
            
            # Suppress stdout/stderr during LLM Guard scanning
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                # Process the prompt using LLM Guard
                processed_prompt, scan_results, risk_scores = scan_function(scanners, prompt)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate overall risk score (average of all scanner risk scores)
            overall_risk = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0.0
            
            # Determine if prompt is safe (any scanner failed or high risk)
            # LLM Guard uses -1 for safe, 1 for unsafe, so we need to convert
            # scan_results: True means scanner passed (safe), False means scanner failed (unsafe)
            # risk_scores: -1 means safe, 1 means unsafe
            is_safe = all(scan_results.values()) and overall_risk < 0.0
            
            return GuardrailsResult(
                is_safe=is_safe,
                confidence=abs(overall_risk),  # Use absolute value for confidence
                reason=f"Overall risk: {overall_risk:.3f}",
                framework="LLM Guard",
                processing_time=processing_time,
                details={
                    "overall_risk": overall_risk,
                    "scanners_used": scanner_names,
                    "scan_results": scan_results,
                    "risk_scores": risk_scores,
                    "processed_prompt": processed_prompt
                }
            )
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            return GuardrailsResult(
                is_safe=False,
                confidence=1.0,
                reason=f"Error in LLM Guard: {str(e)}",
                framework="LLM Guard",
                processing_time=processing_time,
                details={"error": str(e)}
            )
    
    async def process_with_llama_prompt_guard(self, prompt: str) -> GuardrailsResult:
        """Process prompt with Llama Prompt Guard"""
        import time
        import torch
        start_time = time.time()
        
        try:
            framework_info = self.frameworks[GuardrailsFramework.LLAMA_PROMPT_GUARD]
            tokenizer = framework_info["tokenizer"]
            model = framework_info["model"]
            
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Class 0 is safe, Class 1 is unsafe
            is_safe = predicted_class == 0
            
            return GuardrailsResult(
                is_safe=is_safe,
                confidence=confidence,
                reason=f"Class: {'Safe' if is_safe else 'Unsafe'} (confidence: {confidence:.3f})",
                framework="Llama Prompt Guard",
                processing_time=processing_time,
                details={
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "model_name": framework_info["model_name"]
                }
            )
            
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            
            return GuardrailsResult(
                is_safe=False,
                confidence=1.0,
                reason=f"Error in Llama Prompt Guard: {str(e)}",
                framework="Llama Prompt Guard",
                processing_time=processing_time,
                details={"error": str(e)}
            )
    
    async def process_prompt(self, prompt: str, llm=None, config_content: str = None) -> List[GuardrailsResult]:
        """Process prompt with all available frameworks"""
        results = []
        
        # Process with NeMo-Guardrails if available
        if GuardrailsFramework.NEMO_GUARDRAILS in self.frameworks and llm is not None:
            result = await self.process_with_nemo_guardrails(prompt, llm, config_content)
            results.append(result)
        
        # Process with LLM Guard if available
        if GuardrailsFramework.LLM_GUARD in self.frameworks:
            result = await self.process_with_llm_guard(prompt)
            # If we have an LLM and LLM Guard allows the prompt, get the LLM response
            if llm and result.is_safe:
                try:
                    llm_response = await llm.ainvoke(prompt)
                    if hasattr(llm_response, 'content'):
                        llm_text = llm_response.content
                    else:
                        llm_text = str(llm_response)
                    
                    # Update the result with LLM response
                    result.details["llm_response"] = llm_text
                    result.details["response"] = llm_text  # For compatibility
                except Exception as e:
                    result.details["llm_response"] = f"Error getting LLM response: {str(e)}"
                    result.details["response"] = f"Error getting LLM response: {str(e)}"
            results.append(result)
        
        # Process with Llama Prompt Guard if available
        if GuardrailsFramework.LLAMA_PROMPT_GUARD in self.frameworks:
            result = await self.process_with_llama_prompt_guard(prompt)
            # If we have an LLM and Llama Prompt Guard allows the prompt, get the LLM response
            if llm and result.is_safe:
                try:
                    llm_response = await llm.ainvoke(prompt)
                    if hasattr(llm_response, 'content'):
                        llm_text = llm_response.content
                    else:
                        llm_text = str(llm_response)
                    
                    # Update the result with LLM response
                    result.details["llm_response"] = llm_text
                    result.details["response"] = llm_text  # For compatibility
                except Exception as e:
                    result.details["llm_response"] = f"Error getting LLM response: {str(e)}"
                    result.details["response"] = f"Error getting LLM response: {str(e)}"
            results.append(result)
        
        return results
    
    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks"""
        return [framework.value for framework in self.frameworks.keys()]
    
    def is_framework_available(self, framework: GuardrailsFramework) -> bool:
        """Check if a specific framework is available"""
        return framework in self.frameworks

# Global instance
guardrails_manager = GuardrailsManager()
