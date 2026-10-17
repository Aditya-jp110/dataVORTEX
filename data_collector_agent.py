"""
Data Collector Agent - Collects data from multiple sources.

Supports: APIs, Databases, Files, Streaming sources
"""
import time
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger


class DataCollectorAgent(BaseAgent):
    """Collects data from multiple source types."""

    def __init__(self, agent_id: str = "data_collector"):
        """Initialize data collector agent."""
        super().__init__(agent_id)
        self.logger = get_logger(__name__)
        self.timeout = config.get_agent_config('data_collector').get('timeout', 30)
        self.retry_attempts = config.get_agent_config('data_collector').get('retry_attempts', 3)
        self.supported_sources = ['api', 'database', 'file', 'stream']
        self.logger.info("Data Collector Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data collection task.

        Args:
            task: Task containing source information and connection params

        Returns:
            Result with collected data
        """
        task_id = task.get('task_id', 'unknown')
        self._log_task_start(task_id)
        start_time = time.time()

        try:
            input_data = task.get('input_data', {})

            # Handle case where input_data is from previous agent
            if 'source' not in input_data and 'data_location' in input_data:
                # Extract source info from data_location
                data_location = input_data['data_location']
                if data_location.startswith('http'):
                    input_data['source'] = 'api'
                    input_data['url'] = data_location
                elif data_location.endswith(('.csv', '.json', '.xlsx')):
                    input_data['source'] = 'file'
                    input_data['file_path'] = data_location
                else:
                    input_data['source'] = 'database'

            result = self.collect_data(input_data)

            execution_time = time.time() - start_time
            self._log_task_complete(task_id, result['status'] == 'success')

            if result['status'] == 'success':
                return self._create_success_response(
                    output=result['output'],
                    message="Data collection completed",
                    execution_time=execution_time
                )
            else:
                return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task {task_id} failed: {e}")
            self._log_task_complete(task_id, False)
            return self._create_error_response(str(e), execution_time=execution_time)

    def collect_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main data collection method.

        Args:
            params: Collection parameters including source type and connection info

        Returns:
            Collection result
        """
        source_type = params.get('source', 'file')

        if source_type not in self.supported_sources:
            return {
                'status': 'error',
                'message': f"Unsupported source type: {source_type}. Supported: {self.supported_sources}"
            }

        self.logger.info(f"Collecting data from source: {source_type}")

        if source_type == 'api':
            return self._collect_from_api(params)
        elif source_type == 'database':
            return self._collect_from_database(params)
        elif source_type == 'file':
            return self._collect_from_file(params)
        elif source_type == 'stream':
            return self._collect_from_stream(params)
        else:
            return {
                'status': 'error',
                'message': f"Unknown source type: {source_type}"
            }

    def _collect_from_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from REST API with retry logic.

        Args:
            params: API parameters (url, method, headers, auth, etc.)

        Returns:
            Collection result
        """
        url = params.get('url')
        if not url:
            return {'status': 'error', 'message': 'URL is required for API collection'}

        method = params.get('method', 'GET').upper()
        headers = params.get('headers', {})
        auth_type = params.get('auth_type')
        api_key = params.get('api_key', config.get('api_key'))

        # Add authentication
        if auth_type == 'api_key' and api_key:
            headers['Authorization'] = f"Bearer {api_key}"
        elif auth_type == 'api_key_header' and api_key:
            headers['X-API-Key'] = api_key

        # Retry logic with exponential backoff
        for attempt in range(self.retry_attempts):
            try:
                self.logger.info(f"API request attempt {attempt + 1}/{self.retry_attempts}: {method} {url}")

                if method == 'GET':
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=self.timeout,
                        params=params.get('query_params', {})
                    )
                elif method == 'POST':
                    response = requests.post(
                        url,
                        headers=headers,
                        json=params.get('body', {}),
                        timeout=self.timeout
                    )
                else:
                    return {'status': 'error', 'message': f'Unsupported HTTP method: {method}'}

                response.raise_for_status()

                # Parse response
                content_type = response.headers.get('Content-Type', '')

                if 'application/json' in content_type:
                    data = response.json()
                elif 'text/csv' in content_type:
                    from io import StringIO
                    data = pd.read_csv(StringIO(response.text)).to_dict('records')
                else:
                    data = response.text

                # Convert to list if single dict
                if isinstance(data, dict):
                    data = [data]

                records_collected = len(data) if isinstance(data, list) else 1

                self.logger.info(f"Successfully collected {records_collected} records from API")

                return {
                    'status': 'success',
                    'output': {
                        'data': data,
                        'records_collected': records_collected,
                        'metadata': {
                            'source': 'api',
                            'url': url,
                            'method': method,
                            'status_code': response.status_code,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                }

            except requests.exceptions.Timeout:
                self.logger.warning(f"API request timeout (attempt {attempt + 1})")
                if attempt == self.retry_attempts - 1:
                    return {'status': 'error', 'message': f'API request timeout after {self.retry_attempts} attempts'}
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed: {e}")
                if attempt == self.retry_attempts - 1:
                    return {'status': 'error', 'message': f'API request failed: {str(e)}'}
                time.sleep(2 ** attempt)

            except Exception as e:
                return {'status': 'error', 'message': f'Unexpected error: {str(e)}'}

    def _collect_from_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from database.

        Args:
            params: Database parameters (connection string, query, etc.)

        Returns:
            Collection result
        """
        db_type = params.get('db_type', 'postgresql')
        connection_string = params.get('connection_string', config.get('database_url'))
        query = params.get('query')

        if not connection_string:
            return {'status': 'error', 'message': 'Database connection string is required'}

        if not query:
            return {'status': 'error', 'message': 'SQL query is required'}

        try:
            if db_type in ['postgresql', 'mysql', 'sqlite']:
                return self._collect_from_sql(connection_string, query, db_type)
            elif db_type == 'mongodb':
                return self._collect_from_mongodb(params)
            else:
                return {'status': 'error', 'message': f'Unsupported database type: {db_type}'}

        except Exception as e:
            self.logger.error(f"Database collection failed: {e}")
            return {'status': 'error', 'message': f'Database error: {str(e)}'}

    def _collect_from_sql(self, connection_string: str, query: str, db_type: str) -> Dict[str, Any]:
        """Collect from SQL database."""
        try:
            from sqlalchemy import create_engine

            engine = create_engine(connection_string)
            df = pd.read_sql_query(query, engine)

            data = df.to_dict('records')
            records_collected = len(data)

            self.logger.info(f"Collected {records_collected} records from {db_type} database")

            return {
                'status': 'success',
                'output': {
                    'data': data,
                    'records_collected': records_collected,
                    'metadata': {
                        'source': 'database',
                        'db_type': db_type,
                        'query': query,
                        'columns': list(df.columns),
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'SQL database error: {str(e)}'}

    def _collect_from_mongodb(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Collect from MongoDB."""
        try:
            from pymongo import MongoClient

            connection_string = params.get('connection_string', config.get('mongodb_uri'))
            database = params.get('database')
            collection = params.get('collection')
            query = params.get('query', {})
            limit = params.get('limit', 1000)

            client = MongoClient(connection_string, serverSelectionTimeoutMS=self.timeout * 1000)
            db = client[database]
            coll = db[collection]

            data = list(coll.find(query).limit(limit))

            # Convert ObjectId to string
            for doc in data:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])

            records_collected = len(data)

            self.logger.info(f"Collected {records_collected} records from MongoDB")

            return {
                'status': 'success',
                'output': {
                    'data': data,
                    'records_collected': records_collected,
                    'metadata': {
                        'source': 'mongodb',
                        'database': database,
                        'collection': collection,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }

        except Exception as e:
            return {'status': 'error', 'message': f'MongoDB error: {str(e)}'}

    def _collect_from_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read data from file.

        Args:
            params: File parameters (file_path, format, etc.)

        Returns:
            Collection result
        """
        file_path = params.get('file_path')
        if not file_path:
            return {'status': 'error', 'message': 'file_path is required'}

        file_path = Path(file_path)

        if not file_path.exists():
            return {'status': 'error', 'message': f'File not found: {file_path}'}

        try:
            file_extension = file_path.suffix.lower()

            self.logger.info(f"Reading file: {file_path}")

            if file_extension == '.csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                data = df.to_dict('records')
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
                data = df.to_dict('records')
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data = [{'text': text}]
            else:
                return {'status': 'error', 'message': f'Unsupported file format: {file_extension}'}

            records_collected = len(data)

            self.logger.info(f"Collected {records_collected} records from file")

            return {
                'status': 'success',
                'output': {
                    'data': data,
                    'records_collected': records_collected,
                    'metadata': {
                        'source': 'file',
                        'file_path': str(file_path),
                        'file_size_bytes': file_path.stat().st_size,
                        'file_format': file_extension,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }

        except Exception as e:
            self.logger.error(f"File reading failed: {e}")
            return {'status': 'error', 'message': f'File reading error: {str(e)}'}

    def _collect_from_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect from streaming source (Kafka, Kinesis, etc.).

        Args:
            params: Streaming parameters

        Returns:
            Collection result
        """
        stream_type = params.get('stream_type', 'kafka')
        batch_size = params.get('batch_size', 100)
        timeout_ms = params.get('timeout_ms', 5000)

        try:
            if stream_type == 'kafka':
                return self._collect_from_kafka(params, batch_size, timeout_ms)
            else:
                return {
                    'status': 'error',
                    'message': f'Streaming source {stream_type} not yet implemented'
                }

        except Exception as e:
            return {'status': 'error', 'message': f'Streaming error: {str(e)}'}

    def _collect_from_kafka(self, params: Dict[str, Any], batch_size: int, timeout_ms: int) -> Dict[str, Any]:
        """Collect from Kafka topic."""
        # Placeholder for Kafka implementation
        self.logger.warning("Kafka streaming not yet fully implemented")

        return {
            'status': 'success',
            'output': {
                'data': [],
                'records_collected': 0,
                'metadata': {
                    'source': 'kafka',
                    'note': 'Kafka integration placeholder',
                    'timestamp': datetime.now().isoformat()
                }
            }
        }

    def validate_source_config(self, source_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate source configuration before collection.

        Args:
            source_type: Type of data source
            params: Source parameters

        Returns:
            Validation result
        """
        required_fields = {
            'api': ['url'],
            'database': ['connection_string', 'query'],
            'file': ['file_path'],
            'stream': ['stream_type']
        }

        if source_type not in required_fields:
            return {
                'valid': False,
                'message': f'Unknown source type: {source_type}'
            }

        missing_fields = [field for field in required_fields[source_type] if field not in params]

        if missing_fields:
            return {
                'valid': False,
                'message': f'Missing required fields: {missing_fields}'
            }

        return {
            'valid': True,
            'message': 'Configuration is valid'
        }
