"""
Base agent class for all agents in the multi-agent platform.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from utils import get_logger, redis_client


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, agent_id: str):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.status = "idle"
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.logger = get_logger(f"agent.{agent_id}")
        self.start_time = datetime.now()

    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task and return results.

        Args:
            task: Task dictionary containing:
                - task_id: Unique task identifier
                - input_data: Input data for processing
                - metadata: Additional metadata

        Returns:
            Result dictionary containing:
                - status: 'success' or 'error'
                - output: Processing results
                - message: Status message
                - agent_id: This agent's ID
                - execution_time: Time taken in seconds
        """
        pass

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Status dictionary with agent metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat()
        }

    def _update_status(self, status: str):
        """Update agent status and persist to Redis."""
        self.status = status
        redis_client.hset(f"agent_status:{self.agent_id}", "status", status)
        redis_client.hset(f"agent_status:{self.agent_id}", "updated_at", datetime.now().isoformat())

    def _log_task_start(self, task_id: str):
        """Log task start."""
        self.logger.info(f"Starting task {task_id}")
        self._update_status("busy")

    def _log_task_complete(self, task_id: str, success: bool):
        """Log task completion."""
        if success:
            self.tasks_completed += 1
            self.logger.info(f"Task {task_id} completed successfully")
        else:
            self.tasks_failed += 1
            self.logger.error(f"Task {task_id} failed")

        self._update_status("idle")

        # Update metrics in Redis
        redis_client.hset(f"agent_status:{self.agent_id}", "tasks_completed", self.tasks_completed)
        redis_client.hset(f"agent_status:{self.agent_id}", "tasks_failed", self.tasks_failed)

    def _create_success_response(self, output: Any, message: str = "Task completed successfully",
                                execution_time: float = 0.0) -> Dict[str, Any]:
        """Create a standardized success response."""
        return {
            "status": "success",
            "output": output,
            "message": message,
            "agent_id": self.agent_id,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }

    def _create_error_response(self, error: str, retryable: bool = True,
                              execution_time: float = 0.0) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "status": "error",
            "error_code": self._get_error_code(error),
            "message": str(error),
            "agent_id": self.agent_id,
            "retryable": retryable,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }

    def _get_error_code(self, error: str) -> str:
        """Determine error code from error message."""
        error_lower = str(error).lower()

        if "timeout" in error_lower:
            return "TIMEOUT_ERROR"
        elif "connection" in error_lower:
            return "CONNECTION_ERROR"
        elif "validation" in error_lower or "schema" in error_lower:
            return "VALIDATION_ERROR"
        elif "not found" in error_lower:
            return "NOT_FOUND_ERROR"
        elif "permission" in error_lower or "unauthorized" in error_lower:
            return "PERMISSION_ERROR"
        else:
            return "GENERAL_ERROR"

    def send_message(self, target_agent: str, message: Dict[str, Any]):
        """
        Send message to another agent via Redis pub/sub.

        Args:
            target_agent: Target agent ID
            message: Message payload
        """
        message_data = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source_agent": self.agent_id,
            "target_agent": target_agent,
            "payload": message
        }

        channel = f"agent:{target_agent}"
        redis_client.publish(channel, message_data)
        self.logger.debug(f"Sent message to {target_agent}")

    def health_check(self) -> bool:
        """
        Perform health check.

        Returns:
            True if agent is healthy
        """
        try:
            # Check Redis connection
            if not redis_client.ping():
                self.logger.error("Redis connection failed")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


class AgentMessage:
    """Standard message format for inter-agent communication."""

    def __init__(self, source_agent: str, target_agent: str,
                 message_type: str, payload: Dict[str, Any],
                 priority: str = "normal", correlation_id: Optional[str] = None):
        """
        Create an inter-agent message.

        Args:
            source_agent: Source agent ID
            target_agent: Target agent ID
            message_type: Type of message (task_request, task_response, status_update, etc.)
            payload: Message payload
            priority: Message priority (low, normal, high, critical)
            correlation_id: Optional correlation ID for tracking related messages
        """
        self.message_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.message_type = message_type
        self.priority = priority
        self.payload = payload
        self.metadata = {
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "retry_count": 0
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "message_type": self.message_type,
            "priority": self.priority,
            "payload": self.payload,
            "metadata": self.metadata
        }
