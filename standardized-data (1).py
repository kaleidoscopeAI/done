from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime

@dataclass
class StandardizedData:
    """
    A container class that standardizes different types of data for processing
    while preserving their original format and relationships.
    """
    raw_data: Any
    data_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Tuple[str, str]] = field(default_factory=list)
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        # Validate data type
        self.validate_data_type()
        
        # Add automatic metadata based on data type
        self.add_automatic_metadata()
    
    def validate_data_type(self):
        """Validates that the data matches its declared type."""
        valid_types = {
            "text": (str,),
            "image": (np.ndarray,),
            "numerical": (list, np.ndarray, int, float),
            "audio": (np.ndarray,),
            "structured": (dict, list),
        }
        
        if self.data_type not in valid_types:
            raise ValueError(f"Unsupported data type: {self.data_type}")
            
        if not isinstance(self.raw_data, valid_types[self.data_type]):
            raise TypeError(f"Data type mismatch. Expected {valid_types[self.data_type]} for {self.data_type}, got {type(self.raw_data)}")
    
    def add_automatic_metadata(self):
        """Adds automatic metadata based on the data type."""
        if self.data_type == "text":
            self.metadata.update({
                "length": len(self.raw_data),
                "word_count": len(self.raw_data.split())
            })
        elif self.data_type == "image":
            self.metadata.update({
                "shape": self.raw_data.shape,
                "dtype": str(self.raw_data.dtype)
            })
        elif self.data_type == "numerical":
            data_array = np.array(self.raw_data)
            self.metadata.update({
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "mean": float(np.mean(data_array)),
                "std": float(np.std(data_array))
            })
    
    def add_relationship(self, source: str, target: str):
        """Adds a relationship between this data and another piece of data."""
        self.relationships.append((source, target))
    
    def to_dict(self) -> Dict:
        """Converts the standardized data to a dictionary format."""
        return {
            "data_type": self.data_type,
            "metadata": self.metadata,
            "relationships": self.relationships,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }

class DataProcessor:
    """
    Processes different types of data into StandardizedData format.
    """
    def __init__(self):
        self.processors = {
            "text": self._process_text,
            "image": self._process_image,
            "numerical": self._process_numerical,
            "structured": self._process_structured
        }
    
    def process(self, data: Any, data_type: str) -> StandardizedData:
        """
        Processes raw data into StandardizedData format.
        """
        if data_type not in self.processors:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return self.processors[data_type](data)
    
    def _process_text(self, data: str) -> StandardizedData:
        return StandardizedData(
            raw_data=data,
            data_type="text",
            metadata={"format": "utf-8"}
        )
    
    def _process_image(self, data: np.ndarray) -> StandardizedData:
        return StandardizedData(
            raw_data=data,
            data_type="image"
        )
    
    def _process_numerical(self, data: Any) -> StandardizedData:
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        return StandardizedData(
            raw_data=data,
            data_type="numerical"
        )
    
    def _process_structured(self, data: Any) -> StandardizedData:
        return StandardizedData(
            raw_data=data,
            data_type="structured",
            metadata={"structure_type": type(data).__name__}
        )
