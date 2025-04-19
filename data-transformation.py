import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

@dataclass
class TransformationConfig:
    """Configuration for data transformation."""
    normalization_method: str = "standard"  # "standard" or "minmax"
    handle_outliers: bool = True
    outlier_threshold: float = 3.0  # For standard deviations
    handle_missing: bool = True
    categorical_encoding: str = "onehot"  # "onehot" or "label"
    text_vectorization: str = "tfidf"  # "tfidf" or "count"
    dimension_reduction: Optional[str] = None  # "pca" or "tsne"
    n_components: int = 2  # For dimension reduction

class DataTransformer:
    """Handles data transformation and normalization."""
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        self.config = config or TransformationConfig()
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.statistics = {}
        
    def transform_data(self, data: Any, data_type: str) -> Dict[str, Any]:
        """Transforms data based on its type."""
        if data_type == "numerical":
            return self._transform_numerical(data)
        elif data_type == "categorical":
            return self._transform_categorical(data)
        elif data_type == "text":
            return self._transform_text(data)
        elif data_type == "mixed":
            return self._transform_mixed(data)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
    def _transform_numerical(self, data: np.ndarray) -> Dict[str, Any]:
        """Transforms numerical data."""
        result = {
            'original_shape': data.shape,
            'statistics': self._calculate_statistics(data)
        }
        
        # Handle outliers if configured
        if self.config.handle_outliers:
            data = self._handle_outliers(data)
            
        # Apply normalization
        if self.config.normalization_method == "standard":
            if "numerical" not in self.scalers:
                self.scalers["numerical"] = StandardScaler()
            transformed_data = self.scalers["numerical"].fit_transform(data.reshape(-1, 1))
        else:  # minmax
            if "numerical" not in self.scalers:
                self.scalers["numerical"] = MinMaxScaler()
            transformed_data = self.scalers["numerical"].fit_transform(data.reshape(-1, 1))
            
        result['transformed_data'] = transformed_data
        result['transformation_info'] = {
            'method': self.config.normalization_method,
            'scale_params': {
                'mean': float(self.scalers["numerical"].mean_[0]),
                'scale': float(self.scalers["numerical"].scale_[0])
            }
        }
        
        return result
        
    def _transform_categorical(self, data: List[str]) -> Dict[str, Any]:
        """Transforms categorical data."""
        result = {
            'original_categories': sorted(set(data)),
            'statistics': {
                'unique_values': len(set(data)),
                'value_counts': pd.Series(data).value_counts().to_dict()
            }
        }
        
        # One-hot encoding
        if self.config.categorical_encoding == "onehot":
            # Create one-hot encoder if doesn't exist
            if "categorical" not in self.encoders:
                df = pd.get_dummies(data)
                self.encoders["categorical"] = {
                    'columns': df.columns.tolist(),
                    'mapping': {val: i for i, val in enumerate(sorted(set(data)))}
                }
            transformed_data = pd.get_dummies(data).values
            
        else:  # Label encoding
            if "categorical" not in self.encoders:
                unique_values = sorted(set(data))
                self.encoders["categorical"] = {
                    'mapping': {val: i for i, val in enumerate(unique_values)}
                }
            transformed_data = np.array([
                self.encoders["categorical"]['mapping'][val] 
                for val in data
            ])
            
        result['transformed_data'] = transformed_data
        result['transformation_info'] = {
            'method': self.config.categorical_encoding,
            'encoding_mapping': self.encoders["categorical"]
        }
        
        return result
        
    def _transform_text(self, data: List[str]) -> Dict[str, Any]:
        """Transforms text data."""
        result = {
            'original_length': len(data),
            'statistics': {
                'avg_length': np.mean([len(text) for text in data]),
                'vocab_size': len(set(' '.join(data).split()))
            }
        }
        
        # Create vectorizer if doesn't exist
        if "text" not in self.vectorizers:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            if self.config.text_vectorization == "tfidf":
                self.vectorizers["text"] = TfidfVectorizer()
            else:
                self.vectorizers["text"] = CountVectorizer()
                
        # Transform text data
        transformed_data = self.vectorizers["text"].fit_transform(data)
        
        result['transformed_data'] = transformed_data
        result['transformation_info'] = {
            'method': self.config.text_vectorization,
            'vocabulary_size': len(self.vectorizers["text"].vocabulary_),
            'feature_names': self.vectorizers["text"].get_feature_names_out()
        }
        
        return result
        
    def _transform_mixed(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Transforms mixed-type data."""
        result = {
            'original_shape': data.shape,
            'column_types': data.dtypes.to_dict()
        }
        
        transformed_data = pd.DataFrame()
        transformation_info = {}
        
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                # Numerical column
                col_result = self._transform_numerical(data[column].values)
                transformed_data[column] = col_result['transformed_data'].ravel()
                transformation_info[column] = {
                    'type': 'numerical',
                    'info': col_result['transformation_info']
                }
                
            elif data[column].dtype == 'object':
                if self._is_text_column(data[column]):
                    # Text column
                    col_result = self._transform_text(data[column].values)
                    if isinstance(col_result['transformed_data'], np.ndarray):
                        transformed_data[column] = col_result['transformed_data'].ravel()
                    else:  # sparse matrix
                        transformed_data = pd.concat([
                            transformed_data,
                            pd.DataFrame(
                                col_result['transformed_data'].toarray(),
                                columns=[f"{column}_{feat}" for feat in col_result['transformation_info']['feature_names']]
                            )
                        ], axis=1)
                    transformation_info[column] = {
                        'type': 'text',
                        'info': col_result['transformation_info']
                    }
                else:
                    # Categorical column
                    col_result = self._transform_categorical(data[column].values)
                    if self.config.categorical_encoding == "onehot":
                        transformed_data = pd.concat([
                            transformed_data,
                            pd.DataFrame(
                                col_result['transformed_data'],
                                columns=[f"{column}_{cat}" for cat in col_result['transformation_info']['encoding_mapping']['columns']]
                            )
                        ], axis=1)
                    else:
                        transformed_data[column] = col_result['transformed_data']
                    transformation_info[column] = {
                        'type': 'categorical',
                        'info': col_result['transformation_info']
                    }
                    
        result['transformed_data'] = transformed_data
        result['transformation_info'] = transformation_info
        
        return result
        
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """Handles outliers in numerical data."""
        mean = np.mean(data)
        std = np.std(data)
        threshold = self.config.outlier_threshold * std
        
        # Replace outliers with boundary values
        data = np.clip(data, mean - threshold, mean + threshold)
        
        return data
        
    def _calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculates basic statistics for numerical data."""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data))
        }
        
    def _is_text_column(self, series: pd.Series, min_words: int = 3) -> bool:
        """Determines if a series contains text data."""
        # Sample the series to check for text characteristics
        sample = series.dropna().head(100)
        word_counts = sample.str.split().str.len()
        
        # Consider it text if average word count exceeds threshold
        return word_counts.mean() >= min_words

    def get_transformation_summary(self) -> Dict[str, Any]:
        """Returns summary of transformations applied."""
        return {
            'scalers': {
                name: {
                    'type': type(scaler).__name__,
                    'params': {
                        'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                        'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                    }
                }
                for name, scaler in self.scalers.items()
            },
            'encoders': {
                name: {
                    'type': 'categorical',
                    'mapping': encoder['mapping']
                }
                for name, encoder in self.encoders.items()
            },
            'vectorizers': {
                name: {
                    'type': type(vectorizer).__name__,
                    'vocab_size': len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else None
                }
                for name, vectorizer in self.vectorizers.items()
            },
            'config': vars(self.config)
        }
