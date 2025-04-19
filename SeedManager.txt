#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeedManager.py - Enhanced with intelligent seed generation.
"""

import logging
import uuid
import time
import random
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SeedManager:
    """
    Enhanced SeedManager with intelligent seed generation and environmental influence.
    """

    def __init__(self, seed_modifiers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None):
        """
        Args:
            seed_modifiers (Dict[str, Callable]):
                Dictionary of functions to modify seed data based on context.
                Keys are modifier names, values are functions.
        """
        self.seeds: Dict[str, Dict[str, Any]] = {}
        self.seed_history: List[Dict[str, Any]] = []
        self.seed_modifiers = seed_modifiers if seed_modifiers else {}
        self.environmental_context: Dict[str, Any] = {} # Store last known context
        logger.info(f"SeedManager initialized (seed_modifiers: {self.seed_modifiers}).")

    def generate_seed(self, seed_type: str, parent_id: Optional[str] = None,
                      parent_dna: Optional[str] = None, initial_data: Dict[str, Any] = None) -> str:
        """
        Generates a unique seed, potentially influenced by modifiers and environment.
        """

        seed_id = str(uuid.uuid4())
        seed_data = {
            "seed_type": seed_type,
            "parent_id": parent_id,
            "parent_dna": parent_dna,
            "creation_time": time.time(),
            "initial_data": initial_data if initial_data else {},
            "environmental_context": self.environmental_context.copy() # Store current context
        }

        # --- "Wow" Factor: Intelligent Seed Modification ---
        # Apply seed modifiers
        for modifier_name, modifier_func in self.seed_modifiers.items():
            try:
                seed_data["initial_data"] = modifier_func(seed_data["initial_data"], self.environmental_context)
                logger.debug(f"Seed modified by: {modifier_name}")
            except Exception as e:
                logger.error(f"Seed modifier '{modifier_name}' failed: {e}", exc_info=True)

        self.seeds[seed_id] = seed_data
        self.seed_history.append({
            "seed_id": seed_id,
            "event_type": "seed_generated",
            "seed_data": seed_data
        })
        logger.debug(f"Generated seed: {seed_id} (type: {seed_type})")
        return seed_id

    def initialize_from_seed(self, seed_id: str, **kwargs) -> Optional[str]:
        if seed_id not in self.seeds:
            logger.warning(f"Seed '{seed_id}' not found. Cannot initialize node.")
            return None

        seed_data = self.seeds[seed_id]
        node_id = kwargs.get("node_id", str(uuid.uuid4()))

        node_init_data = {**seed_data["initial_data"], **kwargs}

        # Placeholder for node creation logic
        # new_node = YourNodeClass(**node_init_data)
        logger.info(f"Initializing node '{node_id}' from seed '{seed_id}'.")
        self.seed_history.append({
            "seed_id": seed_id,
            "event_type": "node_initialized",
            "node_id": node_id,
            "initialization_data": node_init_data
        })

        return node_id

    def get_seed_data(self, seed_id: str) -> Optional[Dict[str, Any]]:
        return self.seeds.get(seed_id)

    def remove_seed(self, seed_id: str) -> bool:
        if seed_id in self.seeds:
            del self.seeds[seed_id]
            self.seed_history.append({
                "seed_id": seed_id,
                "event_type": "seed_removed"
            })
            logger.info(f"Removed seed: {seed_id}")
            return True
        else:
            logger.warning(f"Seed '{seed_id}' not found.")
            return False

    def get_seed_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.seed_history[-limit:]

    # --- "Wow" Factor: Environmental Adaptation ---

    def update_environmental_context(self, new_context: Dict[str, Any]):
        """
        Updates the stored environmental context.
        This context is used to influence seed generation.
        """
        self.environmental_context.update(new_context)
        logger.info(f"Environmental context updated: {self.environmental_context}")

    def clear_environmental_context(self):
        """
        Clears the stored environmental context.
        """
        self.environmental_context.clear()
        logger.info("Environmental context cleared.")

    # --- Example Seed Modifiers (Adjust for your needs) ---

    @staticmethod
    def modify_for_temperature(initial_data: Dict[str, Any], env_context: Dict[str, float]) -> Dict[str, Any]:
        """
        Example: Modifies seed data based on temperature.
        """
        adapted_data = initial_data.copy()
        temperature = env_context.get("temperature", 25.0) # Default
