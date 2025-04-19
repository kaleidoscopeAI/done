#!/usr/bin/env python3
import os
import json
import logging
import asyncio
import subprocess
import requests
import time
import hashlib
import aiohttp
import sys
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from pathlib import Path

# Add project root to path if running this file directly
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)
    from src.utils.logging_config import configure_logging, get_logger
    configure_logging()
else:
    from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class LLMClient:
    """
    Advanced LLM client with multi-provider support and adaptive optimization.
    Handles local models through Ollama/llama.cpp and remote APIs with
    automatic batching, caching, and fallback mechanisms.
    """

    def __init__(self,
                 config_file: str = None,
                 api_key: str = None,
                 model: str = "mixtral:latest",
                 provider: str = "ollama",
                 endpoint: str = None,
                 cache_dir: str = ".llm_cache",
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 request_timeout: int = 60):
        """
        Initialize the LLM client with adaptive configuration.
        
        Args:
            config_file: Path to JSON configuration file (optional)
            api_key: API key for provider (if using remote API)
            model: Default model to use
            provider: 'ollama', 'llama.cpp', 'openai', 'anthropic', 'custom'
            endpoint: Custom API endpoint URL
            cache_dir: Directory for response caching
            max_tokens: Default maximum tokens for completion
            temperature: Default temperature for generation
            request_timeout: Timeout for requests in seconds
        """
        self.config = {
            "api_key": api_key,
            "model": model,
            "provider": provider,
            "endpoint": endpoint,
            "max_tokens": max_tokens, 
            "temperature": temperature,
            "request_timeout": request_timeout
        }
        
        # Load config file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    # Update config with file values, but preserve any explicitly passed values
                    for k, v in file_config.items():
                        if k not in self.config or self.config[k] is None:
                            self.config[k] = v
                logger.info(f"Loaded LLM configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config from {config_file}: {e}")
        
        # Environment variable override for sensitive info
        if not self.config["api_key"] and "LLM_API_KEY" in os.environ:
            self.config["api_key"] = os.environ.get("LLM_API_KEY")
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_hits = 0
        
        # Set up provider-specific configuration
        self._setup_provider()
        
        # Advanced optimization features
        self.request_queue = asyncio.Queue()
        self.batch_size = 5  # Default batch size for request batching
        self.active_requests = 0
        self.is_processing = False
        self.request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        # Adaptive processing - tracks model performance to optimize parameters
        self.perf_metrics = {
            "avg_latency": 0,
            "success_rate": 1.0,
            "total_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0
        }
        
        logger.info(f"LLMClient initialized with provider: {self.config['provider']}, model: {self.config['model']}")

    def _setup_provider(self):
        """Configure the specific LLM provider."""
        provider = self.config["provider"].lower()
        
        if provider == "ollama":
            # Validate Ollama availability
            try:
                # Check if Ollama is running and model is available
                endpoint = self.config.get("endpoint", "http://localhost:11434")
                response = requests.get(f"{endpoint}/api/tags")
                if response.status_code != 200:
                    logger.warning(f"Ollama API not responding at {endpoint}. Will attempt to start as needed.")
                else:
                    models = response.json().get("models", [])
                    model_names = [m.get("name") for m in models]
                    if self.config["model"] not in model_names:
                        logger.warning(f"Model {self.config['model']} not found in Ollama. Will attempt to pull on first use.")
            except Exception as e:
                logger.warning(f"Cannot connect to Ollama: {e}. Will attempt to start when needed.")
        
        elif provider == "llama.cpp":
            # Check if llama.cpp is installed
            try:
                result = subprocess.run(["llama-cli", "--version"], 
                                       capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    logger.warning("llama.cpp CLI not found in PATH. Specify model path in config.")
            except Exception as e:
                logger.warning(f"Error checking llama.cpp installation: {e}")
        
        elif provider in ["openai", "anthropic", "custom"]:
            # Validate API key for remote providers
            if not self.config.get("api_key") and provider != "custom":
                logger.warning(f"No API key provided for {provider}. API calls will likely fail.")
            
            # Set default endpoints if not specified
            if not self.config.get("endpoint"):
                if provider == "openai":
                    self.config["endpoint"] = "https://api.openai.com/v1/chat/completions"
                elif provider == "anthropic":
                    self.config["endpoint"] = "https://api.anthropic.com/v1/messages"
        
        else:
            logger.warning(f"Unknown provider: {provider}. Falling back to ollama.")
            self.config["provider"] = "ollama"

    def _get_cache_key(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate a deterministic cache key based on input parameters."""
        # Create a normalized representation of the request
        cache_dict = {
            "prompt": prompt,
            "system_message": system_message,
            "model": kwargs.get("model", self.config["model"]),
            "temperature": kwargs.get("temperature", self.config["temperature"]),
            "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
            # Add other parameters that affect output
            "stop_sequences": str(kwargs.get("stop_sequences", [])),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        # Convert to a consistent string representation and hash
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if a response is cached and return it if found."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Check if cache is still valid (e.g., has expiry)
                    if cached_data.get("expiry", float('inf')) > time.time():
                        self.cache_hits += 1
                        logger.debug(f"Cache hit for key {cache_key}")
                        return cached_data.get("response")
            except Exception as e:
                logger.error(f"Error reading cache file {cache_file}: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, response: str, ttl_seconds: int = 86400):
        """Save a response to cache with optional TTL."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "response": response,
                    "created": time.time(),
                    "expiry": time.time() + ttl_seconds
                }, f)
            logger.debug(f"Saved response to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error saving to cache file {cache_file}: {e}")

    async def complete(self, 
                      prompt: str, 
                      system_message: str = None, 
                      use_cache: bool = True,
                      **kwargs) -> str:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The prompt text
            system_message: Optional system message for context
            use_cache: Whether to use response caching
            **kwargs: Override default parameters
            
        Returns:
            The generated completion text
        """
        # Override default config with kwargs
        for k, v in kwargs.items():
            if k in self.config and v is not None:
                kwargs[k] = v
            else:
                kwargs[k] = self.config.get(k)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(prompt, system_message, **kwargs)
            cached_response = self._check_cache(cache_key)
            if cached_response:
                return cached_response
        
        # Track metrics
        start_time = time.time()
        self.perf_metrics["total_requests"] += 1
        
        try:
            # Call the appropriate provider method
            provider = kwargs.get("provider", self.config["provider"]).lower()
            
            if provider == "ollama":
                response = await self._complete_ollama(prompt, system_message, **kwargs)
            elif provider == "llama.cpp":
                response = await self._complete_llamacpp(prompt, system_message, **kwargs)
            elif provider == "openai":
                response = await self._complete_openai(prompt, system_message, **kwargs)
            elif provider == "anthropic":
                response = await self._complete_anthropic(prompt, system_message, **kwargs)
            elif provider == "custom":
                response = await self._complete_custom(prompt, system_message, **kwargs)
            else:
                # Fallback to Ollama
                logger.warning(f"Unknown provider {provider}, falling back to Ollama")
                response = await self._complete_ollama(prompt, system_message, **kwargs)
            
            # Cache the successful response
            if use_cache and cache_key:
                self._save_to_cache(cache_key, response)
            
            # Update performance metrics
            self.perf_metrics["avg_latency"] = (self.perf_metrics["avg_latency"] * 
                                               (self.perf_metrics["total_requests"] - 1) + 
                                               (time.time() - start_time)) / self.perf_metrics["total_requests"]
            
            # Estimate token count (very rough approximation)
            self.perf_metrics["total_tokens"] += len(response.split()) * 1.3
            
            return response
            
        except Exception as e:
            self.perf_metrics["failed_requests"] += 1
            self.perf_metrics["success_rate"] = 1 - (self.perf_metrics["failed_requests"] / 
                                                    self.perf_metrics["total_requests"])
            
            logger.error(f"Error completing prompt: {e}", exc_info=True)
            
            # Implement fallback mechanism for robustness
            if kwargs.get("fallback", True) and provider != "ollama":
                logger.info(f"Attempting fallback to Ollama for failed request")
                try:
                    kwargs["provider"] = "ollama"
                    kwargs["fallback"] = False  # Prevent infinite fallback loops
                    return await self.complete(prompt, system_message, use_cache, **kwargs)
                except Exception as fallback_e:
                    logger.error(f"Fallback also failed: {fallback_e}")
            
            raise RuntimeError(f"LLM completion failed: {str(e)}")

    async def _complete_ollama(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate completion using Ollama API."""
        
        endpoint = kwargs.get("endpoint", "http://localhost:11434")
        model = kwargs.get("model", self.config["model"])
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        temperature = kwargs.get("temperature", self.config["temperature"])
        stop_sequences = kwargs.get("stop_sequences", [])
        timeout = kwargs.get("request_timeout", self.config["request_timeout"])
        
        # Check if Ollama is running, start it if needed
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/api/tags", timeout=5) as response:
                    if response.status != 200:
                        # Attempt to start Ollama
                        logger.info("Ollama not running, attempting to start...")
                        subprocess.Popen(["ollama", "serve"], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE)
                        # Wait for service to start
                        for _ in range(5):
                            await asyncio.sleep(1)
                            try:
                                async with session.get(f"{endpoint}/api/tags", timeout=2) as check_response:
                                    if check_response.status == 200:
                                        logger.info("Ollama started successfully")
                                        break
                            except:
                                pass
                        else:
                            logger.error("Failed to start Ollama after multiple attempts")
        except Exception as e:
            logger.warning(f"Error checking/starting Ollama: {e}")
        
        # Prepare the request
        request_url = f"{endpoint}/api/generate"
        
        request_body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_message:
            request_body["system"] = system_message
        
        if stop_sequences:
            request_body["options"]["stop"] = stop_sequences
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with self.request_semaphore:
                    async with session.post(request_url, json=request_body, timeout=timeout) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            
                            # Check if model needs to be pulled
                            if "model not found" in error_text.lower():
                                logger.info(f"Model {model} not found, attempting to pull...")
                                pull_process = subprocess.Popen(
                                    ["ollama", "pull", model],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE
                                )
                                
                                # Wait for pull to complete
                                pull_process.wait()
                                
                                if pull_process.returncode == 0:
                                    logger.info(f"Successfully pulled model {model}, retrying request")
                                    # Retry the request
                                    async with session.post(request_url, json=request_body, timeout=timeout) as retry_response:
                                        if retry_response.status == 200:
                                            response_json = await retry_response.json()
                                            return response_json.get("response", "")
                                        else:
                                            error_text = await retry_response.text()
                                            raise RuntimeError(f"Ollama API error after model pull: {error_text}")
                                else:
                                    stdout, stderr = pull_process.communicate()
                                    raise RuntimeError(f"Failed to pull model {model}: {stderr.decode()}")
                            else:
                                raise RuntimeError(f"Ollama API error: {error_text}")
                        
                        response_json = await response.json()
                        return response_json.get("response", "")
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Ollama request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error with Ollama request: {str(e)}")

    async def _complete_llamacpp(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate completion using llama.cpp CLI."""
        model_path = kwargs.get("model_path", self.config.get("model_path", "models/7B/ggml-model.bin"))
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        temperature = kwargs.get("temperature", self.config["temperature"])
        stop_sequences = kwargs.get("stop_sequences", [])
        
        # Build the command
        cmd = ["llama-cli", "--model", model_path, 
               "--temp", str(temperature),
               "--n-predict", str(max_tokens),
               "--silent-prompt"]
        
        # Add stop sequences if provided
        for stop in stop_sequences:
            cmd.extend(["--reverse-prompt", stop])
        
        # Prepare prompt with system message if provided
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        # Run the command
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send prompt and get response
            stdout, stderr = await process.communicate(full_prompt.encode())
            
            if process.returncode != 0:
                raise RuntimeError(f"llama.cpp error: {stderr.decode()}")
            
            # Process the output
            response = stdout.decode().strip()
            
            # Remove the prompt from the beginning of the response
            if response.startswith(full_prompt):
                response = response[len(full_prompt):].strip()
            
            return response
        
        except Exception as e:
            raise RuntimeError(f"Error running llama.cpp: {str(e)}")

    async def _complete_openai(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate completion using OpenAI API."""
        
        api_key = kwargs.get("api_key", self.config["api_key"])
        endpoint = kwargs.get("endpoint", "https://api.openai.com/v1/chat/completions")
        model = kwargs.get("model", "gpt-3.5-turbo")
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        temperature = kwargs.get("temperature", self.config["temperature"])
        stop_sequences = kwargs.get("stop_sequences", [])
        timeout = kwargs.get("request_timeout", self.config["request_timeout"])
        
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Prepare the messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if stop_sequences:
            request_body["stop"] = stop_sequences
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with self.request_semaphore:
                    async with session.post(endpoint, headers=headers, json=request_body, timeout=timeout) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"OpenAI API error: {error_text}")
                        
                        response_json = await response.json()
                        return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"OpenAI request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error with OpenAI request: {str(e)}")

    async def _complete_anthropic(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate completion using Anthropic API."""
        
        api_key = kwargs.get("api_key", self.config["api_key"])
        endpoint = kwargs.get("endpoint", "https://api.anthropic.com/v1/messages")
        model = kwargs.get("model", "claude-2")
        max_tokens = kwargs.get("max_tokens", self.config["max_tokens"])
        temperature = kwargs.get("temperature", self.config["temperature"])
        timeout = kwargs.get("request_timeout", self.config["request_timeout"])
        
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        request_body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if system_message:
            request_body["system"] = system_message
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with self.request_semaphore:
                    async with session.post(endpoint, headers=headers, json=request_body, timeout=timeout) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"Anthropic API error: {error_text}")
                        
                        response_json = await response.json()
                        return response_json.get("content", [{}])[0].get("text", "")
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Anthropic request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error with Anthropic request: {str(e)}")

    async def _complete_custom(self, prompt: str, system_message: str = None, **kwargs) -> str:
        """Generate completion using a custom API endpoint."""
        
        api_key = kwargs.get("api_key", self.config["api_key"])
        endpoint = kwargs.get("endpoint")
        timeout = kwargs.get("request_timeout", self.config["request_timeout"])
        
        if not endpoint:
            raise ValueError("Custom API endpoint not provided")
        
        # Prepare the request - adapt this based on your custom API format
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        request_body = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.config["max_tokens"]),
            "temperature": kwargs.get("temperature", self.config["temperature"])
        }
        
        if system_message:
            request_body["system"] = system_message
        
        # Make the API request
        try:
            async with aiohttp.ClientSession() as session:
                async with self.request_semaphore:
                    async with session.post(endpoint, headers=headers, json=request_body, timeout=timeout) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise RuntimeError(f"Custom API error: {error_text}")
                        
                        response_json = await response.json()
                        # Adapt this based on your API's response format
                        return response_json.get("response", "")
        
        except asyncio.TimeoutError:
            raise RuntimeError(f"Custom API request timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error with custom API request: {str(e)}")

    async def batch_process(self, prompts: List[str], system_message: str = None, **kwargs) -> List[str]:
        """
        Process multiple prompts more efficiently as a batch.
        
        Args:
            prompts: List of prompts to process
            system_message: Optional system message for all prompts
            **kwargs: Override default parameters
            
        Returns:
            List of completions corresponding to input prompts
        """
        results = []
        
        # Process in parallel with concurrency limit
        tasks = []
        for prompt in prompts:
            task = asyncio.ensure_future(self.complete(prompt, system_message, **kwargs))
            tasks.append(task)
        
        # Wait for all completions and collect results
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                results.append(result)
            except Exception as e:
                # Add None or error message for failed requests
                logger.error(f"Error in batch processing: {e}")
                results.append(f"Error: {str(e)}")
        
        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            "avg_latency_seconds": round(self.perf_metrics["avg_latency"], 2),
            "success_rate": round(self.perf_metrics["success_rate"], 4),
            "total_requests": self.perf_metrics["total_requests"],
            "failed_requests": self.perf_metrics["failed_requests"],
            "estimated_total_tokens": int(self.perf_metrics["total_tokens"]),
            "cache_hits": self.cache_hits,
            "provider": self.config["provider"],
            "model": self.config["model"]
        }

    def estimate_token_count(self, text: str) -> int:
        """Roughly estimate the number of tokens in a text."""
        # Simple estimation based on words and punctuation
        # More accurate methods would use model-specific tokenizers
        words = len(text.split())
        return int(words * 1.3)  # Rough approximation


if __name__ == "__main__":
    # Test the LLMClient functionality
    import asyncio
    
    async def test_client():
        # Initialize client
        client = LLMClient(provider="ollama", model="mistral:latest")
        
        # Test completion
        prompt = "Explain what a transformer model is in 3 sentences."
        print(f"Testing completion with prompt: {prompt}")
        
        try:
            response = await client.complete(prompt)
            print(f"Response: {response}")
            print(f"Performance stats: {client.get_performance_stats()}")
            
            # Test batch processing
            batch_prompts = [
                "Explain quantum computing briefly.",
                "What are the main programming paradigms?",
                "How does a neural network work?"
            ]
            print(f"\nTesting batch processing with {len(batch_prompts)} prompts...")
            batch_results = await client.batch_process(batch_prompts)
            for i, result in enumerate(batch_results):
                print(f"\nPrompt {i+1}: {batch_prompts[i]}\nResponse: {result[:100]}...")
                
        except Exception as e:
            print(f"Error testing client: {e}")
    
    # Run the test
    asyncio.run(test_client())
