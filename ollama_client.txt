#!/usr/bin/env python3
"""
Ollama API Client
================
Client for interacting with Ollama API for local LLM inference.
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        """
        Initialize the Ollama client
        
        Args:
            base_url: Base URL for the Ollama API
        """
        self.base_url = base_url
        self.session = requests.Session()
        logger.debug(f"Initialized Ollama client with base URL: {base_url}")
        
    def list_models(self) -> Dict[str, Any]:
        """
        List available models
        
        Returns:
            List of available models
        """
        try:
            response = self.session.get(f"{self.base_url}/tags")
            return response.json()
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {"error": str(e)}
    
    def generate(self, 
                prompt: str, 
                model: str = "codellama", 
                temperature: float = 0.2,
                max_tokens: int = 2048,
                stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate text using the specified model
        
        Args:
            prompt: The prompt to generate from
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop: Optional list of strings that stop generation when encountered
            
        Returns:
            Generated text and metadata
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            if stop:
                payload["stop"] = stop
                
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return {"error": str(e)}
    
    def chat(self, 
            messages: List[Dict[str, str]],
            model: str = "codellama",
            temperature: float = 0.2,
            max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Chat with the specified model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Chat response and metadata
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "num_predict": max_tokens
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {"error": str(e)}
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Result of the pull operation
        """
        try:
            payload = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/pull",
                json=payload
            )
            
            return response.json()
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return {"error": str(e)}
