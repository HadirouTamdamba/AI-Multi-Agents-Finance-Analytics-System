"""
Task routing and orchestration for AI Finance Agent Team.
Manages task distribution and agent coordination.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .schemas import AgentType, TaskStatus, AgentTask, AgentResponse
from .logging import get_logger, log_agent_activity
from .tracing import trace_span, trace_agent_activity


logger = get_logger(__name__)


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskQueue:
    """Task queue for managing pending tasks."""
    tasks: List[AgentTask] = None
    max_size: int = 100
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
    
    def add_task(self, task: AgentTask, priority: TaskPriority = TaskPriority.NORMAL):
        """Add task to queue with priority."""
        if len(self.tasks) >= self.max_size:
            logger.warning("Task queue is full, dropping oldest task")
            self.tasks.pop(0)
        
        # Simple priority implementation - insert at appropriate position
        if priority == TaskPriority.CRITICAL:
            self.tasks.insert(0, task)
        elif priority == TaskPriority.HIGH:
            insert_pos = min(1, len(self.tasks))
            self.tasks.insert(insert_pos, task)
        else:
            self.tasks.append(task)
        
        logger.info(f"Added task {task.id} to queue with priority {priority}")
    
    def get_next_task(self, agent_type: Optional[AgentType] = None) -> Optional[AgentTask]:
        """Get next task for specified agent type."""
        for i, task in enumerate(self.tasks):
            if agent_type is None or task.agent_type == agent_type:
                if task.status == TaskStatus.PENDING:
                    self.tasks.pop(i)
                    return task
        return None
    
    def get_task_count(self, agent_type: Optional[AgentType] = None) -> int:
        """Get count of pending tasks for agent type."""
        if agent_type is None:
            return len([t for t in self.tasks if t.status == TaskStatus.PENDING])
        return len([t for t in self.tasks if t.agent_type == agent_type and t.status == TaskStatus.PENDING])


class TaskRouter:
    """Routes tasks to appropriate agents based on type and load."""
    
    def __init__(self):
        self.task_queue = TaskQueue()
        self.agent_loads: Dict[AgentType, int] = {
            agent_type: 0 for agent_type in AgentType
        }
        self.agent_capabilities: Dict[AgentType, List[str]] = {
            AgentType.RESEARCHER: ["text_analysis", "news_processing", "sentiment_analysis"],
            AgentType.ANALYST: ["data_analysis", "chart_generation", "metric_calculation"],
            AgentType.RISK_MODELER: ["risk_assessment", "scenario_modeling", "stress_testing"],
            AgentType.WRITER: ["report_generation", "summary_creation", "document_assembly"],
            AgentType.ORCHESTRATOR: ["task_coordination", "workflow_management", "result_aggregation"]
        }
        self.max_concurrent_tasks = 5
    
    def can_handle_task(self, agent_type: AgentType, task_type: str) -> bool:
        """Check if agent can handle the task type."""
        capabilities = self.agent_capabilities.get(agent_type, [])
        return task_type in capabilities
    
    def get_best_agent(self, task_type: str) -> Optional[AgentType]:
        """Get the best agent for a task type based on capabilities and load."""
        suitable_agents = [
            agent_type for agent_type, capabilities in self.agent_capabilities.items()
            if task_type in capabilities
        ]
        
        if not suitable_agents:
            logger.warning(f"No suitable agent found for task type: {task_type}")
            return None
        
        # Return agent with lowest load
        return min(suitable_agents, key=lambda a: self.agent_loads[a])
    
    def route_task(self, task: AgentTask, priority: TaskPriority = TaskPriority.NORMAL) -> bool:
        """Route task to appropriate agent."""
        # Check if we can handle this task type
        best_agent = self.get_best_agent(task.task_type)
        if not best_agent:
            logger.error(f"Cannot route task {task.id}: no suitable agent for type {task.task_type}")
            task.status = TaskStatus.FAILED
            task.error_message = f"No suitable agent for task type: {task.task_type}"
            return False
        
        # Update task with assigned agent
        task.agent_type = best_agent
        
        # Check agent load
        if self.agent_loads[best_agent] >= self.max_concurrent_tasks:
            logger.warning(f"Agent {best_agent} is at capacity, queuing task {task.id}")
            self.task_queue.add_task(task, priority)
            return True
        
        # Route immediately
        self.agent_loads[best_agent] += 1
        task.status = TaskStatus.IN_PROGRESS
        
        logger.info(f"Routed task {task.id} to {best_agent}")
        return True
    
    def complete_task(self, task_id: str, agent_type: AgentType):
        """Mark task as completed and update agent load."""
        self.agent_loads[agent_type] = max(0, self.agent_loads[agent_type] - 1)
        
        # Process next queued task for this agent
        next_task = self.task_queue.get_next_task(agent_type)
        if next_task:
            next_task.status = TaskStatus.IN_PROGRESS
            self.agent_loads[agent_type] += 1
            logger.info(f"Started queued task {next_task.id} for {agent_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "agent_loads": dict(self.agent_loads),
            "queued_tasks": self.task_queue.get_task_count(),
            "queued_by_agent": {
                agent_type.value: self.task_queue.get_task_count(agent_type)
                for agent_type in AgentType
            },
            "total_capacity": len(AgentType) * self.max_concurrent_tasks,
            "utilization": sum(self.agent_loads.values()) / (len(AgentType) * self.max_concurrent_tasks)
        }


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows and task dependencies."""
    
    def __init__(self):
        self.router = TaskRouter()
        self.workflows: Dict[str, List[AgentTask]] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        self.completed_tasks: Dict[str, AgentResponse] = {}
    
    def create_workflow(self, workflow_id: str, tasks: List[AgentTask]) -> bool:
        """Create a new workflow with tasks."""
        try:
            # Validate task dependencies
            for task in tasks:
                if task.id in self.task_dependencies:
                    deps = self.task_dependencies[task.id]
                    for dep_id in deps:
                        if not any(t.id == dep_id for t in tasks):
                            logger.error(f"Workflow {workflow_id}: dependency {dep_id} not found for task {task.id}")
                            return False
            
            self.workflows[workflow_id] = tasks
            logger.info(f"Created workflow {workflow_id} with {len(tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_id}: {e}")
            return False
    
    def add_task_dependency(self, task_id: str, depends_on: List[str]):
        """Add dependencies for a task."""
        self.task_dependencies[task_id] = depends_on
        logger.debug(f"Added dependencies for task {task_id}: {depends_on}")
    
    def can_execute_task(self, task_id: str) -> bool:
        """Check if task can be executed (dependencies satisfied)."""
        if task_id not in self.task_dependencies:
            return True
        
        deps = self.task_dependencies[task_id]
        return all(dep_id in self.completed_tasks for dep_id in deps)
    
    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow and return results."""
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return {"success": False, "error": "Workflow not found"}
        
        tasks = self.workflows[workflow_id]
        results = {}
        
        logger.info(f"Starting workflow {workflow_id} with {len(tasks)} tasks")
        
        try:
            # Execute tasks in dependency order
            remaining_tasks = tasks.copy()
            executed_tasks = set()
            
            while remaining_tasks:
                # Find tasks that can be executed
                executable_tasks = [
                    task for task in remaining_tasks
                    if self.can_execute_task(task.id) and task.id not in executed_tasks
                ]
                
                if not executable_tasks:
                    # Check for circular dependencies or missing dependencies
                    logger.error(f"Workflow {workflow_id}: no executable tasks remaining")
                    break
                
                # Execute tasks in parallel (simplified - in real implementation would use asyncio)
                for task in executable_tasks:
                    if self.router.route_task(task):
                        executed_tasks.add(task.id)
                        remaining_tasks.remove(task)
                
                # Simulate task completion (in real implementation, this would be async)
                for task in executable_tasks:
                    if task.id in executed_tasks:
                        # Create mock response
                        response = AgentResponse(
                            agent_type=task.agent_type,
                            task_id=task.id,
                            content=f"Completed {task.task_type}",
                            confidence="medium",
                            execution_time=1.0
                        )
                        self.completed_tasks[task.id] = response
                        self.router.complete_task(task.id, task.agent_type)
            
            results = {
                "success": True,
                "workflow_id": workflow_id,
                "completed_tasks": len(self.completed_tasks),
                "total_tasks": len(tasks),
                "results": self.completed_tasks
            }
            
            logger.info(f"Workflow {workflow_id} completed: {len(self.completed_tasks)}/{len(tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            results = {"success": False, "error": str(e)}
        
        return results
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow."""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        tasks = self.workflows[workflow_id]
        completed = len([t for t in tasks if t.id in self.completed_tasks])
        
        return {
            "workflow_id": workflow_id,
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "progress": completed / len(tasks) if tasks else 0,
            "status": "completed" if completed == len(tasks) else "in_progress"
        }


# Global instances
task_router = TaskRouter()
workflow_orchestrator = WorkflowOrchestrator()


def get_task_router() -> TaskRouter:
    """Get global task router."""
    return task_router


def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get global workflow orchestrator."""
    return workflow_orchestrator


def route_task(task: AgentTask, priority: TaskPriority = TaskPriority.NORMAL) -> bool:
    """Route a task using the global router."""
    return task_router.route_task(task, priority)


def create_workflow(workflow_id: str, tasks: List[AgentTask]) -> bool:
    """Create a workflow using the global orchestrator."""
    return workflow_orchestrator.create_workflow(workflow_id, tasks)


def execute_workflow(workflow_id: str) -> Dict[str, Any]:
    """Execute a workflow using the global orchestrator."""
    return workflow_orchestrator.execute_workflow(workflow_id)