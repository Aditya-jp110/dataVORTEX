"""
Orchestrator Agent - Master controller for multi-agent system.

This agent coordinates all other agents, manages workflows, and handles
pipeline selection based on data characteristics.
"""
import asyncio
import argparse
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from utils.config import config
from utils.logger import get_logger
from utils.redis_client import redis_client
import time


class OrchestratorAgent(BaseAgent):
    """Master controller for multi-agent system."""

    def __init__(self, agent_id: str = "orchestrator"):
        """Initialize orchestrator agent."""
        super().__init__(agent_id)
        self.active_workflows = {}
        self.agent_registry = {}
        self.logger = get_logger(__name__)
        self.max_retries = config.get('orchestrator.max_retries', 3)
        self.retry_delay = config.get('orchestrator.retry_delay', 2)
        self.workflow_timeout = config.get('orchestrator.workflow_timeout', 3600)

        # Register available pipeline types
        self.pipeline_configs = {
            'small_structured': config.get_pipeline_config('small_structured'),
            'large_structured': config.get_pipeline_config('large_structured'),
            'nlp': config.get_pipeline_config('nlp'),
            'computer_vision': config.get_pipeline_config('computer_vision'),
            'streaming': config.get_pipeline_config('streaming')
        }

        self.logger.info("Orchestrator Agent initialized")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process orchestration task.

        Args:
            task: Task containing workflow creation or execution request

        Returns:
            Result with workflow status
        """
        task_type = task.get('task_type', 'create_workflow')

        if task_type == 'create_workflow':
            return self.create_workflow(task.get('data_info', {}))
        elif task_type == 'execute_workflow':
            workflow_id = task.get('workflow_id')
            return asyncio.run(self.execute_workflow(workflow_id))
        elif task_type == 'get_status':
            return self.get_workflow_status(task.get('workflow_id'))
        elif task_type == 'cancel_workflow':
            return self.cancel_workflow(task.get('workflow_id'))
        else:
            return self._create_error_response(f"Unknown task type: {task_type}")

    def create_workflow(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create workflow based on data characteristics.

        Args:
            data_info: Dictionary containing:
                - type: 'structured' | 'unstructured'
                - format: 'csv' | 'json' | 'text' | 'image' | 'video'
                - size_bytes: int
                - source: 'api' | 'database' | 'file' | 'stream'
                - is_streaming: bool

        Returns:
            Workflow configuration dictionary
        """
        start_time = time.time()

        try:
            self.logger.info(f"Creating workflow for data: {data_info}")

            # Determine optimal pipeline
            pipeline = self._determine_pipeline(data_info)

            # Create workflow ID
            workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"

            # Create workflow object
            workflow = {
                'workflow_id': workflow_id,
                'status': 'pending',
                'pipeline': pipeline,
                'data_info': data_info,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'results': {},
                'current_stage': None,
                'progress': 0.0
            }

            # Store in Redis
            redis_client.set(f"workflow:{workflow_id}", workflow, expire=86400)  # 24 hour TTL

            # Store in active workflows
            self.active_workflows[workflow_id] = workflow

            execution_time = time.time() - start_time

            self.logger.info(f"Workflow {workflow_id} created with pipeline: {pipeline['name']}")

            return self._create_success_response(
                output={'workflow': workflow},
                message=f"Workflow created successfully",
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            return self._create_error_response(str(e))

    def _determine_pipeline(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine optimal pipeline based on data characteristics.

        Args:
            data_info: Data characteristics

        Returns:
            Pipeline configuration
        """
        data_type = data_info.get('type', 'structured')
        data_format = data_info.get('format', 'csv')
        size_bytes = data_info.get('size_bytes', 0)
        is_streaming = data_info.get('is_streaming', False)

        size_mb = size_bytes / (1024 * 1024)

        # Pipeline selection logic
        if is_streaming:
            pipeline_name = 'streaming'
        elif data_type == 'unstructured':
            if data_format in ['text', 'txt', 'json']:
                pipeline_name = 'nlp'
            elif data_format in ['jpg', 'jpeg', 'png', 'image', 'video', 'mp4']:
                pipeline_name = 'computer_vision'
            else:
                pipeline_name = 'small_structured'
        else:  # structured
            if size_mb < 100:
                pipeline_name = 'small_structured'
            else:
                pipeline_name = 'large_structured'

        pipeline_config = self.pipeline_configs.get(pipeline_name, self.pipeline_configs['small_structured'])

        # Estimate execution time
        estimated_time = self._estimate_execution_time(size_mb, pipeline_name)

        return {
            'name': pipeline_name,
            'processing_mode': pipeline_config.get('processing_mode', 'single_node'),
            'agents': pipeline_config.get('agents', []),
            'estimated_time_seconds': estimated_time
        }

    def _estimate_execution_time(self, size_mb: float, pipeline_name: str) -> int:
        """Estimate execution time based on data size and pipeline."""
        base_times = {
            'small_structured': 30,
            'large_structured': 300,
            'nlp': 60,
            'computer_vision': 120,
            'streaming': 10
        }

        base_time = base_times.get(pipeline_name, 60)

        # Add time based on size (rough estimate)
        size_factor = max(1.0, size_mb / 10)

        return int(base_time * size_factor)

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute workflow by coordinating agents.

        Args:
            workflow_id: Workflow ID to execute

        Returns:
            Execution results
        """
        start_time = time.time()

        try:
            # Get workflow
            workflow = self._get_workflow(workflow_id)
            if not workflow:
                return self._create_error_response(f"Workflow {workflow_id} not found", retryable=False)

            self.logger.info(f"Executing workflow {workflow_id}")

            # Update status
            workflow['status'] = 'in_progress'
            workflow['started_at'] = datetime.now().isoformat()
            self._update_workflow(workflow_id, workflow)

            # Execute each agent in pipeline
            pipeline = workflow['pipeline']
            agents = pipeline.get('agents', [])
            total_stages = len(agents)

            results = {}

            for idx, agent_type in enumerate(agents):
                stage_num = idx + 1
                self.logger.info(f"Executing stage {stage_num}/{total_stages}: {agent_type}")

                # Update progress
                workflow['current_stage'] = agent_type
                workflow['progress'] = (idx / total_stages) * 100
                self._update_workflow(workflow_id, workflow)

                # Create task for agent
                task = self._create_agent_task(workflow, agent_type)

                # Execute with retry logic
                result = await self._execute_agent_task_with_retry(agent_type, task)

                if result['status'] == 'error':
                    self.logger.error(f"Agent {agent_type} failed: {result.get('message')}")
                    workflow['status'] = 'failed'
                    workflow['error'] = result.get('message')
                    workflow['failed_at'] = datetime.now().isoformat()
                    self._update_workflow(workflow_id, workflow)

                    execution_time = time.time() - start_time
                    return self._create_error_response(
                        f"Workflow failed at stage {agent_type}: {result.get('message')}",
                        execution_time=execution_time
                    )

                # Store result
                results[agent_type] = result
                workflow['results'][agent_type] = result
                self._update_workflow(workflow_id, workflow)

            # Workflow completed successfully
            workflow['status'] = 'completed'
            workflow['progress'] = 100.0
            workflow['completed_at'] = datetime.now().isoformat()
            self._update_workflow(workflow_id, workflow)

            execution_time = time.time() - start_time

            self.logger.info(f"Workflow {workflow_id} completed successfully in {execution_time:.2f}s")

            return self._create_success_response(
                output={'results': results, 'workflow_id': workflow_id},
                message="Workflow executed successfully",
                execution_time=execution_time
            )

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return self._create_error_response(str(e))

    def _get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow from Redis or active workflows."""
        # Check active workflows first
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]

        # Check Redis
        workflow = redis_client.get(f"workflow:{workflow_id}")
        if workflow:
            self.active_workflows[workflow_id] = workflow

        return workflow

    def _update_workflow(self, workflow_id: str, workflow: Dict[str, Any]):
        """Update workflow in Redis and active workflows."""
        workflow['updated_at'] = datetime.now().isoformat()
        redis_client.set(f"workflow:{workflow_id}", workflow, expire=86400)
        self.active_workflows[workflow_id] = workflow

    def _create_agent_task(self, workflow: Dict[str, Any], agent_type: str) -> Dict[str, Any]:
        """Create task for a specific agent."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"

        # Get previous agent's output as input for current agent
        previous_output = None
        if workflow['results']:
            # Get last successful result
            for result in reversed(list(workflow['results'].values())):
                if result['status'] == 'success':
                    previous_output = result.get('output')
                    break

        input_data = previous_output or workflow['data_info']
        if agent_type == 'visualization':
            processor_result = workflow['results'].get('data_processor', {})
            processed_data = processor_result.get('output', {}).get('data', [])
            input_data = {
                'data': processed_data,
                'insights': (previous_output or {}).get('insights', []),
                'analysis_results': (previous_output or {}).get('analysis_results', {})
            }

        task = {
            'task_id': task_id,
            'workflow_id': workflow['workflow_id'],
            'agent_type': agent_type,
            'input_data': input_data,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'retry_count': 0
            }
        }
        if agent_type == 'analyzer':
            task['analysis_type'] = workflow['data_info'].get('analysis_type', 'classification')
        return task

    async def _execute_agent_task_with_retry(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task with retry logic."""
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                result = await self._execute_agent_task(agent_type, task)

                if result['status'] == 'success':
                    return result

                last_error = result.get('message', 'Unknown error')

                # Check if retryable
                if not result.get('retryable', True):
                    return result

                retry_count += 1
                if retry_count < self.max_retries:
                    delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                    self.logger.warning(f"Retrying {agent_type} in {delay}s (attempt {retry_count + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                if retry_count < self.max_retries:
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    await asyncio.sleep(delay)

        return self._create_error_response(f"Max retries exceeded. Last error: {last_error}")

    async def _execute_agent_task(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the corresponding concrete agent implementation."""
        from agents.analyzer_agent import AnalyzerAgent
        from agents.data_collector_agent import DataCollectorAgent
        from agents.data_processor_agent import DataProcessorAgent
        from agents.insight_generator_agent import InsightGeneratorAgent
        from agents.quality_assurance_agent import QualityAssuranceAgent
        from agents.visualization_agent import VisualizationAgent

        agent_classes = {
            'data_collector': DataCollectorAgent,
            'quality_assurance': QualityAssuranceAgent,
            'data_processor': DataProcessorAgent,
            'analyzer': AnalyzerAgent,
            'insight_generator': InsightGeneratorAgent,
            'visualization': VisualizationAgent
        }
        agent_class = agent_classes.get(agent_type)
        if agent_class is None:
            return self._create_error_response(
                f"Unsupported agent type: {agent_type}", retryable=False
            )

        self.logger.info(f"Executing {agent_type} task {task['task_id']}")
        return await asyncio.to_thread(agent_class().process_task, task)

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status."""
        workflow = self._get_workflow(workflow_id)

        if not workflow:
            return self._create_error_response(f"Workflow {workflow_id} not found", retryable=False)

        return self._create_success_response(
            output={'workflow_status': workflow},
            message="Workflow status retrieved"
        )

    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a running workflow."""
        workflow = self._get_workflow(workflow_id)

        if not workflow:
            return self._create_error_response(f"Workflow {workflow_id} not found", retryable=False)

        if workflow['status'] not in ['pending', 'in_progress']:
            return self._create_error_response(
                f"Cannot cancel workflow in status: {workflow['status']}",
                retryable=False
            )

        workflow['status'] = 'cancelled'
        workflow['cancelled_at'] = datetime.now().isoformat()
        self._update_workflow(workflow_id, workflow)

        self.logger.info(f"Workflow {workflow_id} cancelled")

        return self._create_success_response(
            output={'workflow_id': workflow_id},
            message="Workflow cancelled successfully"
        )

    def get_agent_health(self) -> Dict[str, Any]:
        """Get health status of all registered agents."""
        health_status = {
            'orchestrator': self.health_check(),
            'redis': redis_client.ping(),
            'agents': {}
        }

        # Check registered agents
        for agent_id in self.agent_registry:
            agent_status = redis_client.hgetall(f"agent_status:{agent_id}")
            health_status['agents'][agent_id] = agent_status

        return self._create_success_response(
            output=health_status,
            message="Health check completed"
        )

    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register an agent with the orchestrator."""
        self.agent_registry[agent_id] = agent_info
        redis_client.hset(f"agent_registry", agent_id, agent_info)
        self.logger.info(f"Registered agent: {agent_id}")


def main() -> int:
    """Run the configured end-to-end workflow as a Docker Compose job."""
    parser = argparse.ArgumentParser(description="Run the multi-agent analytics workflow")
    parser.add_argument(
        '--input',
        default=os.getenv('PIPELINE_INPUT', '/app/data/sample_customers.csv'),
        help='CSV input path (defaults to the sample customer file)'
    )
    parser.add_argument('--analysis-type', default='classification')
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.is_file():
        print(f"Input file not found: {data_path}")
        return 2

    data_info = {
        'type': 'structured',
        'format': data_path.suffix.lstrip('.').lower() or 'csv',
        'source': 'file',
        'file_path': str(data_path),
        'size_bytes': data_path.stat().st_size,
        'is_streaming': False,
        'analysis_type': args.analysis_type
    }

    orchestrator = OrchestratorAgent()
    workflow_response = orchestrator.create_workflow(data_info)
    if workflow_response.get('status') != 'success':
        print(workflow_response.get('message', 'Could not create workflow'))
        return 1

    workflow_id = workflow_response['output']['workflow']['workflow_id']
    result = asyncio.run(orchestrator.execute_workflow(workflow_id))
    if result.get('status') != 'success':
        print(result.get('message', 'Workflow failed'))
        return 1

    agent_results = result['output']['results']
    insight_output = agent_results.get('insight_generator', {}).get('output', {})
    summary = insight_output.get('executive_summary', {})
    visualizations = agent_results.get('visualization', {}).get('output', {}).get('visualizations', [])

    print(f"Workflow {workflow_id} completed successfully")
    print(f"Input: {data_path}")
    print(summary.get('summary_text', 'Analysis completed.'))
    print(f"High-impact findings: {summary.get('high_impact_count', 0)}")
    print(f"Actionable recommendations: {summary.get('actionable_count', 0)}")
    print("Top insights:")
    for index, insight in enumerate(insight_output.get('insights', [])[:3], start=1):
        print(f"{index}. {insight.get('title', 'Insight')}: {insight.get('description', '')}")
    for visualization in visualizations:
        print(f"Output: {visualization.get('path')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
