"""
Redis client module for state management and caching.
"""
import json
import redis
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Dict
from .config import config
from .logger import get_logger

logger = get_logger(__name__)


def _json_default(value: Any):
    """Convert common Python, NumPy, and path values for JSON storage."""
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class RedisClient:
    """Redis client wrapper for the multi-agent platform."""

    _instance: Optional['RedisClient'] = None
    _client: Optional[redis.Redis] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self._connect()

    def _connect(self):
        """Establish connection to Redis."""
        try:
            redis_config = config.get_redis_config()
            self._client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {redis_config.get('host')}:{redis_config.get('port')}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON serialized if dict/list)
            expire: Optional expiration time in seconds

        Returns:
            True if successful
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=_json_default)

            result = self._client.set(key, value)

            if expire:
                self._client.expire(key, expire)

            return result
        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Value if exists, None otherwise
        """
        try:
            value = self._client.get(key)
            if value is None:
                return None

            # Try to parse as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """Delete a key from Redis."""
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Failed to check key {key}: {e}")
            return False

    def hset(self, name: str, key: str, value: Any) -> bool:
        """Set hash field."""
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            return bool(self._client.hset(name, key, value))
        except Exception as e:
            logger.error(f"Failed to hset {name}:{key}: {e}")
            return False

    def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field."""
        try:
            value = self._client.hget(name, key)
            if value is None:
                return None

            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Failed to hget {name}:{key}: {e}")
            return None

    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields."""
        try:
            result = self._client.hgetall(name)
            parsed_result = {}
            for key, value in result.items():
                try:
                    parsed_result[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed_result[key] = value
            return parsed_result
        except Exception as e:
            logger.error(f"Failed to hgetall {name}: {e}")
            return {}

    def publish(self, channel: str, message: Any) -> int:
        """Publish message to a channel."""
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            return self._client.publish(channel, message)
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return 0

    def subscribe(self, channel: str):
        """Subscribe to a channel."""
        try:
            pubsub = self._client.pubsub()
            pubsub.subscribe(channel)
            return pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            return None

    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter."""
        try:
            return self._client.incr(key, amount)
        except Exception as e:
            logger.error(f"Failed to increment {key}: {e}")
            return None

    def keys(self, pattern: str = '*') -> list:
        """Get keys matching pattern."""
        try:
            return self._client.keys(pattern)
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []

    def flushdb(self):
        """Clear all keys in current database (use with caution!)."""
        try:
            self._client.flushdb()
            logger.warning("Redis database flushed")
        except Exception as e:
            logger.error(f"Failed to flush database: {e}")

    def ping(self) -> bool:
        """Check if Redis is alive."""
        try:
            return self._client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False


# Global Redis client instance - lazy initialization to avoid crash on import
_redis_client = None

def get_redis_client():
    """Get or create the Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client

# For backward compatibility - create a proxy that loads on first access
class _RedisClientProxy:
    def __getattr__(self, name):
        return getattr(get_redis_client(), name)

    def __setattr__(self, name, value):
        return setattr(get_redis_client(), name, value)

    def __delattr__(self, name):
        return delattr(get_redis_client(), name)

redis_client = _RedisClientProxy()
