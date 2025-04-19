#!/usr/bin/env python3
"""
Unit tests for the Ollama client
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the project root to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ollama_client import OllamaClient

class TestOllamaClient(unittest.TestCase):
    """Test cases for the OllamaClient class"""
    
    def setUp(self):
        """Set up test environment"""
        self.client = OllamaClient("http://test-endpoint/api")
    
    @patch("requests.Session.get")
    def test_list_models(self, mock_get):
        """Test listing models"""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"models": [{"name": "model1"}, {"name": "model2"}]})
        mock_response.json.return_value = {"models": [{"name": "model1"}, {"name": "model2"}]}
        mock_get.return_value = mock_response
        
        # Call method
        result = self.client.list_models()
        
        # Verify
        mock_get.assert_called_once_with("http://test-endpoint/api/tags")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "model1")
        self.assertEqual(result[1]["name"], "model2")
    
    @patch("requests.Session.post")
    def test_generate(self, mock_post):
        """Test generating text"""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"response": "Generated text"})
        mock_response.json.return_value = {"response": "Generated text"}
        mock_post.return_value = mock_response
        
        # Call method
        result = self.client.generate(
            prompt="Test prompt",
            model="testmodel",
            temperature=0.5,
            max_tokens=100
        )
        
        # Verify
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://test-endpoint/api/generate")
        self.assertEqual(mock_post.call_args[1]["json"]["model"], "testmodel")
        self.assertEqual(mock_post.call_args[1]["json"]["prompt"], "Test prompt")
        self.assertEqual(mock_post.call_args[1]["json"]["temperature"], 0.5)
        self.assertEqual(mock_post.call_args[1]["json"]["num_predict"], 100)
        self.assertEqual(result["response"], "Generated text")
    
    @patch("requests.Session.post")
    def test_chat(self, mock_post):
        """Test chat functionality"""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"response": "Chat response"})
        mock_response.json.return_value = {"response": "Chat response"}
        mock_post.return_value = mock_response
        
        # Call method
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        result = self.client.chat(
            messages=messages,
            model="testmodel"
        )
        
        # Verify
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://test-endpoint/api/chat")
        self.assertEqual(mock_post.call_args[1]["json"]["model"], "testmodel")
        self.assertEqual(mock_post.call_args[1]["json"]["messages"], messages)
        self.assertEqual(result["response"], "Chat response")
    
    @patch("requests.Session.post")
    def test_pull_model(self, mock_post):
        """Test pulling a model"""
        # Mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"status": "success"})
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        
        # Call method
        result = self.client.pull_model("testmodel")
        
        # Verify
        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args[0][0], "http://test-endpoint/api/pull")
        self.assertEqual(mock_post.call_args[1]["json"]["name"], "testmodel")
        self.assertEqual(result["status"], "success")
    
    @patch("requests.Session.get")
    def test_request_error(self, mock_get):
        """Test error handling in requests"""
        # Mock error response
        mock_get.side_effect = Exception("Test error")
        
        # Call method
        result = self.client.list_models()
        
        # Verify
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Test error")

if __name__ == "__main__":
    unittest.main()
