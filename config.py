"""
Configuration management module for loading and accessing system configuration.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class Config:
    """Configuration manager for the multi-agent platform."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self._load_config()

    def _load_config(self):
        """Load configuration from YAML and environment variables."""
        # Load environment variables
        load_dotenv()

        # Load YAML configuration
        config_path = Path(__file__).parent.parent / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = {}

        # Override with environment variables where applicable
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Redis
        if 'orchestrator' not in self._config:
            self._config['orchestrator'] = {}
        if 'redis' not in self._config['orchestrator']:
            self._config['orchestrator']['redis'] = {}

        self._config['orchestrator']['redis']['host'] = os.getenv(
            'REDIS_HOST',
            self._config['orchestrator']['redis'].get('host', 'localhost')
        )
        self._config['orchestrator']['redis']['port'] = int(os.getenv(
            'REDIS_PORT',
            self._config['orchestrator']['redis'].get('port', 6379)
        ))

        # Database
        self._config['database_url'] = os.getenv('DATABASE_URL', '')
        self._config['mongodb_uri'] = os.getenv('MONGODB_URI', '')

        # API
        self._config['api_base_url'] = os.getenv('API_BASE_URL', '')
        self._config['api_key'] = os.getenv('API_KEY', '')

        # AWS
        self._config['aws'] = {
            'access_key_id': os.getenv('AWS_ACCESS_KEY_ID', ''),
            'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY', ''),
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            's3_bucket': os.getenv('S3_BUCKET', '')
        }

        # Processing
        self._config['max_workers'] = int(os.getenv('MAX_WORKERS', 4))
        self._config['processing_mode'] = os.getenv('PROCESSING_MODE', 'single_node')

        # Model paths
        self._config['model_path'] = os.getenv('MODEL_PATH', './models/')
        self._config['use_gpu'] = os.getenv('USE_GPU', 'false').lower() == 'true'

        # Storage
        self._config['output_path'] = os.getenv('OUTPUT_PATH', './output/')
        self._config['data_path'] = os.getenv('DATA_PATH', './data/')

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key using dot notation."""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.get(f'agents.{agent_name}', {})

    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """Get configuration for a specific pipeline."""
        return self.get(f'pipelines.{pipeline_name}', {})

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return self.get('orchestrator.redis', {})

    @property
    def redis_host(self) -> str:
        return self.get('orchestrator.redis.host', 'localhost')

    @property
    def redis_port(self) -> int:
        return self.get('orchestrator.redis.port', 6379)

    @property
    def database_url(self) -> str:
        return self.get('database_url', '')

    @property
    def use_gpu(self) -> bool:
        return self.get('use_gpu', False)

    def reload(self):
        """Reload configuration from files."""
        self._config = {}
        self._load_config()


# Global configuration instance
config = Config()
