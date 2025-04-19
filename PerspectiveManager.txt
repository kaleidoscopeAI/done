#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerspectiveManager.py - Enhanced with Cube integration.
"""

import logging
import time
import numpy as np
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PerspectiveManager:
    """
    Enhanced PerspectiveManager with Cube-inspired transformations.
    """

    def __init__(self, tensor_dimensions: Optional[List[str]] = None):
        """
        Args:
            tensor_dimensions (Optional[List[str]]):
                If provided, hints at the dimensions of the tensors
                that might be used in subsequent processing.
                Example: ["time", "space", "molecule_type"].
        """
        self.perspectives: Dict[str, Callable[[Any], Any]] = {}
        self.transform_history: List[Dict[str, Any]] = []
        self.tensor_dimensions = tensor_dimensions
        logger.info(f"PerspectiveManager initialized (tensor dimensions: {tensor_dimensions}).")

    def add_perspective(self, name: str, transform_function: Callable[[Any], Any]):
        self.perspectives[name] = transform_function
        logger.info(f"Added perspective: {name}")

    def process_perspectives(self, data: Any) -> Dict[str, Any]:
        results = {}
        start_time = time.time()
        for name, transform in self.perspectives.items():
            try:
                results[name] = transform(data)
                logger.debug(f"Applied perspective '{name}'.")
            except Exception as e:
                logger.error(f"Perspective '{name}' failed: {e}", exc_info=True)
                results[name] = None

        processing_time = time.time() - start_time
        self._record_transform_event(data, results, processing_time)
        return results

    def get_perspective_names(self) -> List[str]:
        return list(self.perspectives.keys())

    def remove_perspective(self, name: str) -> bool:
        if name in self.perspectives:
            del self.perspectives[name]
            logger.info(f"Removed perspective: {name}")
            return True
        else:
            logger.warning(f"Perspective '{name}' not found.")
            return False

    def _record_transform_event(self, original_data: Any, transformed_data: Dict[str, Any], processing_time: float):
        event = {
            "timestamp": time.time(),
            "original_data": str(original_data)[:100],
            "transformed_data": {k: str(v)[:50] for k, v in transformed_data.items()},
            "processing_time": processing_time
        }
        self.transform_history.append(event)
        logger.debug(f"Transformation event: {event}")

    def get_transform_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.transform_history[-limit:]

    def clear_transform_history(self):
        self.transform_history.clear()
        logger.info("Transformation history cleared.")

    # --- "Wow" Factor: Cube-Inspired Perspectives ---

    def create_tensor_slices(self, data: np.ndarray, slice_dim: int) -> List[np.ndarray]:
        """
        Extracts slices from a tensor along a specified dimension.
        Mimics the Cube's multidimensional awareness.
        """
        if not isinstance(data, np.ndarray) or data.ndim <= slice_dim:
            logger.warning(f"Invalid data or slice dimension: {slice_dim}")
            return []

        slices = [np.take(data, i, axis=slice_dim) for i in range(data.shape[slice_dim])]
        return slices

    def calculate_string_tension(self, data: np.ndarray, node_positions: np.ndarray, k_constant=1.0) -> np.ndarray:
        """
        Simulates "string tension" between data points based on their values
        and positions, inspired by the Cube's dynamic strings.
        """
        if not isinstance(data, np.ndarray) or not isinstance(node_positions, np.ndarray) or data.shape[0] != node_positions.shape[0]:
            logger.warning("Invalid input for string tension calculation.")
            return None

        num_nodes = data.shape[0]
        tension_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Simplified "force" calculation (you can expand this)
                distance = np.linalg.norm(node_positions[i] - node_positions[j])
                value_difference = np.linalg.norm(data[i] - data[j])
                tension = k_constant * distance * value_difference
                tension_matrix[i, j] = tension
                tension_matrix[j, i] = tension
        return tension_matrix

    def apply_dynamic_filter(self, data: np.ndarray, time_weights: np.ndarray) -> np.ndarray:
        """
        Applies a dynamic filter that emphasizes certain data points
        based on external "time_weights", simulating stress adaptation.
        """
        if not isinstance(data, np.ndarray) or not isinstance(time_weights, np.ndarray) or data.shape[0] != time_weights.shape[0]:
            logger.warning("Invalid input for dynamic filter.")
            return data

        # Example: Emphasize data points with higher weights
        normalized_weights = time_weights / np.sum(time_weights)
        filtered_data = data * normalized_weights[:, np.newaxis]
        return filtered_data

    def extract_high_dimensional_features(self, data: np.ndarray, num_components=10) -> np.ndarray:
        """
        Extracts key features from high-dimensional data using PCA.
        (Placeholder for more sophisticated tensor factorization)
        """
        if not isinstance(data, np.ndarray) or data.ndim < 2:
            logger.warning("High-dimensional feature extraction requires a matrix.")
            return data

        # Simplified PCA (replace with Tensor Decomposition if available)
        try:
            from sklearn.decomposition import PCA  # Only import if needed
            pca = PCA(n_components=num_components)
            return pca.fit_transform(data)
        except ImportError:
            logger.warning("scikit-learn not available. Returning original data.")
            return data


# Example Usage (for testing)
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)

    perspective_manager = PerspectiveManager(tensor_dimensions=["time", "molecule", "space"])

    # Add Cube-inspired perspectives
    perspective_manager.add_perspective("tensor_slices", perspective_manager.create_tensor_slices)
    perspective_manager.add_perspective("string_tension", perspective_manager.calculate_string_tension)
    perspective_manager.add_perspective("dynamic_filter", perspective_manager.apply_dynamic_filter)
    perspective_manager.add_perspective("hd_features", perspective_manager.extract_high_dimensional_features)

    # Example data (replace with your actual biological data)
    numerical_data = np.random.rand(10, 20)  # 10 data points, 20 features
    position_data = np.random.rand(10, 3)  # 10 data points, 3D positions
    time_weights = np.random.rand(10)

    # Process data
    transformed_numerical = perspective_manager.process_perspectives(numerical_data)
    print("Transformed Numerical Data:", transformed_numerical.keys())

    # Example of using a specific perspective
    slices = perspective_manager.create_tensor_slices(numerical_data, slice_dim=0)
    if slices:
        print("Tensor Slices:", [s.shape for s in slices])

    tension_matrix = perspective_manager.calculate_string_tension(numerical_data, position_data)
    if tension_matrix is not None:
        print("Tension Matrix Shape:", tension_matrix.shape)

    filtered_data = perspective_manager.apply_dynamic_filter(numerical_data, time_weights)
    print("Filtered Data Shape:", filtered_data.shape)

    hd_features = perspective_manager.extract_high_dimensional_features(numerical_data)
    print("HD Features Shape:", hd_features.shape)

    print("Transformation History:", perspective_manager.get_transform_history())
