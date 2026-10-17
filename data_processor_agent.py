"""
Data Processor Agent - Cleans, transforms, and processes data.

Handles: structured data cleaning, NLP preprocessing, computer vision preprocessing
"""
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class DataProcessorAgent(BaseAgent):
    """Processes and transforms data for analysis."""

    def __init__(self, agent_id: str = "data_processor"):
        """Initialize data processor agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)

        agent_config = config.get_agent_config('data_processor')
        self.missing_value_strategy = agent_config.get('missing_value_strategy', 'mean')
        self.normalization_method = agent_config.get('normalization_method', 'standard')

        self.logger.info("Data Processor Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data processing task.

        Args:
            task: Task containing data to process

        Returns:
            Processed data result
        """
        task_id = task.get('task_id', 'unknown')
        self._log_task_start(task_id)
        start_time = time.time()

        try:
            input_data = task.get('input_data', {})

            # Extract data from previous agent's output
            if 'data' in input_data:
                data = input_data['data']
                metadata = input_data.get('metadata', {})
            elif 'output' in input_data and 'data' in input_data['output']:
                data = input_data['output']['data']
                metadata = input_data['output'].get('metadata', {})
            else:
                data = input_data
                metadata = {}

            # Determine data type
            data_type = self._determine_data_type(data, metadata)

            # Process based on type
            if data_type == 'structured':
                result = self.process_structured_data(data)
            elif data_type == 'text':
                result = self.process_text(data)
            elif data_type == 'image':
                result = self.process_images(data)
            else:
                result = self.process_structured_data(data)  # Default

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, True)

            return self._create_success_response(
                output={
                    'data': result['data'],
                    'processing_info': result['info'],
                    'metadata': metadata
                },
                message="Data processing completed",
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Data processing failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def _determine_data_type(self, data: Any, metadata: Dict) -> str:
        """Determine the type of data to process."""
        # Check metadata first
        source = metadata.get('source', '')
        file_format = metadata.get('file_format', '')

        if file_format in ['.txt', 'text'] or source == 'text':
            return 'text'
        elif file_format in ['.jpg', '.png', 'image'] or source == 'image':
            return 'image'
        else:
            return 'structured'

    def process_structured_data(self, data: Union[List[Dict], pd.DataFrame]) -> Dict[str, Any]:
        """
        Process structured data (tabular).

        Args:
            data: List of dicts or DataFrame

        Returns:
            Processed data with info
        """
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        original_shape = df.shape
        processing_steps = []

        # Step 1: Remove duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates()
            processing_steps.append(f"Removed {dup_count} duplicate rows")

        # Step 2: Handle missing values
        df = self._handle_missing_values(df)
        processing_steps.append(f"Handled missing values using {self.missing_value_strategy} strategy")

        # Step 3: Clean data types
        df = self._clean_data_types(df)
        processing_steps.append("Cleaned and standardized data types")

        # Step 4: Remove outliers (optional, configurable)
        # df = self._remove_outliers(df)

        # Step 5: Create derived features
        df = self._create_derived_features(df)
        processing_steps.append("Created derived features")

        # Convert back to list of dicts
        processed_data = df.to_dict('records')

        return {
            'data': processed_data,
            'info': {
                'original_shape': original_shape,
                'processed_shape': df.shape,
                'rows_removed': original_shape[0] - df.shape[0],
                'processing_steps': processing_steps,
                'timestamp': datetime.now().isoformat()
            }
        }

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Numeric columns
                    if self.missing_value_strategy == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif self.missing_value_strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif self.missing_value_strategy == 'forward_fill':
                        df_clean[col] = df_clean[col].ffill()
                    else:
                        df_clean[col].fillna(0, inplace=True)
                else:
                    # Categorical/string columns
                    mode_value = df_clean[col].mode()
                    if len(mode_value) > 0:
                        df_clean[col].fillna(mode_value[0], inplace=True)
                    else:
                        df_clean[col].fillna('unknown', inplace=True)

        return df_clean

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data types."""
        df_clean = df.copy()

        for col in df_clean.columns:
            # Try to convert to numeric if possible
            if df_clean[col].dtype == 'object':
                try:
                    # Try converting to numeric
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                except:
                    pass

                # Clean string columns
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.strip()

        return df_clean

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns."""
        df_enhanced = df.copy()

        # Check for datetime columns
        for col in df_enhanced.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df_enhanced[col] = pd.to_datetime(df_enhanced[col], errors='ignore')
                    if pd.api.types.is_datetime64_any_dtype(df_enhanced[col]):
                        df_enhanced[f'{col}_year'] = df_enhanced[col].dt.year
                        df_enhanced[f'{col}_month'] = df_enhanced[col].dt.month
                        df_enhanced[f'{col}_day'] = df_enhanced[col].dt.day
                        df_enhanced[f'{col}_dayofweek'] = df_enhanced[col].dt.dayofweek
                except:
                    pass

        return df_enhanced

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if self.normalization_method == 'standard':
                # Standardization (z-score normalization)
                mean = df_normalized[col].mean()
                std = df_normalized[col].std()
                if std > 0:
                    df_normalized[col] = (df_normalized[col] - mean) / std
            elif self.normalization_method == 'minmax':
                # Min-Max normalization
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)

        return df_normalized

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            # Label encoding for binary categories
            unique_values = df_encoded[col].nunique()
            if unique_values == 2:
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            elif unique_values <= 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(col, axis=1, inplace=True)

        return df_encoded

    def process_text(self, data: Union[List[Dict], List[str], str]) -> Dict[str, Any]:
        """
        Process text data (NLP preprocessing).

        Args:
            data: Text data to process

        Returns:
            Processed text with info
        """
        # Extract text from data
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # Extract text field from dicts
                texts = [d.get('text', str(d)) for d in data]
            else:
                texts = [str(t) for t in data]
        else:
            texts = [str(data)]

        processing_steps = []

        # Step 1: Clean text
        cleaned_texts = [self._clean_text(text) for text in texts]
        processing_steps.append("Cleaned text (lowercase, remove special chars)")

        # Step 2: Tokenize
        tokenized = [self._tokenize(text) for text in cleaned_texts]
        processing_steps.append("Tokenized text")

        # Step 3: Remove stopwords (optional)
        # filtered = [self._remove_stopwords(tokens) for tokens in tokenized]

        # Step 4: Extract features
        features = self._extract_text_features(cleaned_texts)
        processing_steps.append("Extracted text features")

        processed_data = [
            {
                'original_text': texts[i],
                'cleaned_text': cleaned_texts[i],
                'tokens': tokenized[i],
                'word_count': len(tokenized[i]),
                'char_count': len(cleaned_texts[i])
            }
            for i in range(len(texts))
        ]

        return {
            'data': processed_data,
            'info': {
                'total_texts': len(texts),
                'processing_steps': processing_steps,
                'average_word_count': np.mean([len(t) for t in tokenized]),
                'timestamp': datetime.now().isoformat()
            }
        }

    def _clean_text(self, text: str) -> str:
        """Clean text data."""
        import re

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.split()

    def _extract_text_features(self, texts: List[str]) -> Dict[str, Any]:
        """Extract features from text."""
        total_words = sum(len(text.split()) for text in texts)
        total_chars = sum(len(text) for text in texts)

        return {
            'total_words': total_words,
            'total_chars': total_chars,
            'avg_words_per_text': total_words / len(texts) if texts else 0,
            'avg_chars_per_text': total_chars / len(texts) if texts else 0
        }

    def process_images(self, data: Any) -> Dict[str, Any]:
        """
        Process image data (computer vision preprocessing).

        Args:
            data: Image data

        Returns:
            Processed images with info
        """
        # Placeholder for image processing
        processing_steps = [
            "Image preprocessing not fully implemented",
            "Would include: resize, normalize, augment"
        ]

        return {
            'data': data,
            'info': {
                'processing_steps': processing_steps,
                'note': 'Image processing placeholder',
                'timestamp': datetime.now().isoformat()
            }
        }

    def extract_features(self, df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract specific features from DataFrame.

        Args:
            df: Input DataFrame
            feature_list: Optional list of features to extract

        Returns:
            DataFrame with selected features
        """
        if feature_list:
            available_features = [f for f in feature_list if f in df.columns]
            return df[available_features]
        else:
            return df

    def clean_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Public method for cleaning structured data.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        result = self.process_structured_data(df)
        return pd.DataFrame(result['data'])
