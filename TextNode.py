from collections import deque
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextNode.py - Production-ready text processing node.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from kaleidoscope_ai.core.laws import GrowthLaws  # Assuming CoreLaws is in the 'core' package
from .CapabilityNode import CapabilityNode  # Adjust import if needed
from kaleidoscope_ai.llm import GPTProcessor  # Adjust import path

# --- Enhanced Imports (for production) ---
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import re
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TextNode(CapabilityNode):
    """
    A robust node for processing text, with advanced capabilities.
    """

    def __init__(self, core_laws: CoreLaws,
                 gpt_processor: Optional[GPTProcessor] = None,
                 node_id: Optional[str] = None,
                 max_workers: int = 4,  # For concurrent tasks
                 context_window_size: int = 5):  # For context-aware processing
        """
        Initializes the TextNode with production-focused settings.

        Args:
            core_laws (CoreLaws): Core laws governing node behavior.
            gpt_processor (Optional[GPTProcessor], optional): GPT processor.
            node_id (Optional[str], optional): Unique node ID.
            max_workers (int, optional): Max threads for concurrent tasks.
            context_window_size (int, optional): Size of the context window.
        """
        super().__init__(capability_name="text_processing", core_laws=core_laws, node_id=node_id)
        self.gpt_processor = gpt_processor
        self.text_processing_config: Dict[str, Any] = {
            "max_response_length": 200,
            "summary_length": 100,
            "entity_extraction_limit": 20,
            "batch_size": 4,  # Process text in batches
            "analysis_timeout": 10.0,  # Seconds
            "style_similarity_threshold": 0.8  # Threshold for style matching
        }
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_history: List[Dict[str, Any]] = []
        self.context_window: deque[str] = deque(maxlen=context_window_size)
        logger.info(f"TextNode '{self.id}' initialized (max_workers={max_workers}).")

    def _estimate_cost_factor(self, data: Any) -> float:
        """Robust cost estimation."""
        if isinstance(data, str):
            return len(data) * 0.005 + 1  # Base cost + cost per char
        elif isinstance(data, list):
            return sum(len(item) * 0.005 + 1 for item in data if isinstance(item, str))
        return 10.0  # Default

    def execute_capability(self, data: Any, action: str = "analyze", **kwargs) -> Any:
        """
        Executes a text processing action, handling batching and concurrency.
        """

        try:
            if isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    raise TypeError("Batch data must be a list of strings.")
                results = self._process_text_batch(data, action, **kwargs)
            elif isinstance(data, str):
                results = self._process_text_single(data, action, **kwargs)
            else:
                raise TypeError(f"Invalid data type: {type(data)}. Expected str or List[str].")

            return results

        except Exception as e:
            logger.error(f"{self.id}: Error executing '{action}': {e}", exc_info=True)
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def _process_text_batch(self, text_batch: List[str], action: str, **kwargs) -> List[Dict[str, Any]]:
        """Processes a batch of text data concurrently."""

        logger.info(f"{self.id}: Processing batch of {len(text_batch)} texts, action: {action}")
        futures = [self.executor.submit(self._process_text_single, text, action, **kwargs) for text in text_batch]
        results = []
        for future in as_completed(futures, timeout=self.text_processing_config["analysis_timeout"]):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"{self.id}: Batch processing error: {e}", exc_info=True)
                results.append({"status": "error", "message": str(e), "traceback": traceback.format_exc()})

        return results

    def _process_text_single(self, text: str, action: str, **kwargs) -> Dict[str, Any]:
        """Processes a single text item."""

        start_time = time.time()
        try:
            if action == "analyze":
                result = self.analyze_text(text, **kwargs)
            elif action == "summarize":
                result = self.summarize_text(text, **kwargs)
            elif action == "extract_entities":
                result = self.extract_entities(text, **kwargs)
            elif action == "generate_response":
                result = self.generate_response(text, **kwargs)
            else:
                raise ValueError(f"Unknown text processing action: {action}")

            processing_time_ms = int((time.time() - start_time) * 1000)
            self._record_processing_event(text, action, result, processing_time_ms)
            return {"status": "success", "action": action, "result": result, "processing_time_ms": processing_time_ms}

        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            self._record_processing_event(text, action, {"error": str(e)}, processing_time_ms)
            logger.error(f"{self.id}: Error processing '{action}': {e}", exc_info=True)
            return {"status": "error", "action": action, "message": str(e), "processing_time_ms": processing_time_ms, "traceback": traceback.format_exc()}

    def _record_processing_event(self, text: str, action: str, result: Dict[str, Any], processing_time_ms: int):
        """Records a processing event for analysis."""

        event = {
            "timestamp": time.time(),
            "text_length": len(text),
            "action": action,
            "result_type": type(result).__name__,
            "processing_time_ms": processing_time_ms
        }
        self.processing_history.append(event)
        logger.debug(f"{self.id}: Recorded processing event: {event}")

    # --- Text Processing Capabilities (Production-Ready) ---

    def analyze_text(self, text: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """
        Performs in-depth text analysis, including sentiment, topics, and structure.
        """
        logger.info(f"{self.id}: Analyzing text (detail: {detail_level})")
        analysis_result = {
            "status": "success",
            "analysis_type": "detailed",
            "word_count": len(text.split()),
        }

        try:
            analysis_result["sentiment"] = self._analyze_sentiment(text)
            analysis_result["topics"] = self._detect_topics(text)
            analysis_result["structure"] = self._analyze_text_structure(text)
            analysis_result["multimodal_hints"] = self._get_multimodal_hints()
            analysis_result["style"] = self._analyze_input_style(text)

        except Exception as e:
            logger.error(f"{self.id}: Analysis error: {e}", exc_info=True)
            analysis_result["status"] = "error"
            analysis_result["message"] = str(e)

        return analysis_result

    def _analyze_sentiment(self, text: str) -> str:
        """Placeholder: Analyze sentiment of text."""
        # Replace with a sentiment analysis library (e.g., NLTK, TextBlob)
        return "neutral"

    def _detect_topics(self, text: str) -> List[str]:
        """Placeholder: Detect main topics in text."""
        # Replace with topic modeling (e.g., Gensim, scikit-learn)
        return ["general"]

    def _analyze_text_structure(self, text: str) -> Dict[str, int]:
        """Placeholder: Analyze text structure (sentence lengths, etc.)."""
        sentences = text.split('.')
        return {"avg_sentence_length": int(np.mean([len(s.split()) for s in sentences])) if sentences else 0}

    def summarize_text(self, text: str, max_length: Optional[int] = None) -> Optional[str]:
        """
        Summarizes the input text using the GPT processor (if available).
        """
        if not self.gpt_processor or not self.gpt_processor.is_ready():
            logger.warning(f"{self.id}: GPT Processor unavailable. Cannot summarize.")
            return None

        logger.info(f"{self.id}: Summarizing text (max_length: {max_length})")
        try:
            summary = self.gpt_processor.query(
                prompt=f"Summarize the following text: {text}",
                max_length=max_length or self.text_processing_config["summary_length"]
            )
            return summary
        except Exception as e:
            logger.error(f"{self.id}: GPT error during summarization: {e}", exc_info=True)
            return None

    def extract_entities(self, text: str, limit: Optional[int] = None) -> Optional[List[Tuple[str, str]]]:
        """
        Extracts named entities from the input text (if GPT Processor available).
        """
        if not self.gpt_processor or not self.gpt_processor.is_ready():
            logger.warning(f"{self.id}: GPT Processor unavailable. Cannot extract entities.")
            return None

        logger.info(f"{self.id}: Extracting entities (limit: {limit})")
        try:
            # Placeholder: Replace with a real NER system using GPT (or other library)
            prompt = f"Extract named entities (PERSON, ORG, LOC) from: {text}"
            response = self.gpt_processor.query(prompt=prompt, max_length=200)
            if response:
                # Placeholder: Parse GPT output into a list of (entity, type) tuples
                # This parsing logic needs to be tailored to your GPT's output format
                entities = [("AI", "ORG"), ("Kaleidoscope", "ORG")]  # Dummy
                return entities[:limit or self.text_processing_config["entity_extraction_limit"]]
            else:
                return None
        except Exception as e:
            logger.error(f"{self.id}: GPT error during entity extraction: {e}", exc_info=True)
            return None

    def generate_response(self, text: str, max_length: Optional[int] = None) -> Optional[str]:
        """
        Generates a response to the input text using the GPT processor.
        """
        if not self.gpt_processor or not self.gpt_processor.is_ready():
            logger.warning(f"{self.id}: GPT Processor unavailable. Cannot generate response.")
            return None

        logger.info(f"{self.id}: Generating response (max_length: {max_length})")
        try:
            input_style = kwargs.get("input_style", {}) # Get style if provided
            temperature = 0.7 # Default temperature
            if input_style.get("avg_word_length", 0) > 6:
                temperature -= 0.1 # More complex text, less random
            elif input_style.get("avg_word_length", 0) < 4:
                temperature += 0.1 # Simpler text, more random

            temperature = max(0.2, min(1.0, temperature)) # Clamp temperature

            response = self.gpt_processor.query(
                prompt=text,
                max_length=max_length or self.text_processing_config["max_response_length"],
                temperature=temperature
            )
            return response
        except Exception as e:
            logger.error(f"{self.id}: GPT error generating response: {e}", exc_info=True)
            return None

    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns recent processing history."""
        return self.processing_history[-limit:]

    def clear_processing_history(self):
        """Clears the processing history."""
        self.processing_history.clear()
        logger.info(f"{self.id}: Processing history cleared.")
        return True

    # --- "Wow" Factor: Advanced Text Capabilities ---

    def _get_multimodal_hints(self) -> Dict[str, Any]:
        """
        Placeholder: Retrieves hints from other nodes (e.g., VisualNode)
        about associated images.
        """
        # Replace with actual inter-node communication mechanism
        return {"image_keywords": ["landscape", "sunset"]}  # Dummy hints

    def _analyze_input_style(self, text: str) -> Dict[str, float]:
        """
        Analyzes the input text's style (formality, complexity).
        (Simplified example)
        """
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        num_sentences = text.count('.')
        return {"avg_word_length": avg_word_length, "num_sentences": num_sentences}

    def speculate_on_arguments(self, text: str) -> List[Dict[str, str]]:
        """
        Placeholder: Identifies potential argumentative structures in text.
        """
        # Replace with actual argument parsing logic
        return [{"claim": "AI is beneficial", "evidence": "Increased efficiency"}]  # Dummy
