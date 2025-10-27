"""
Tools - Orchestration Components

Thin orchestration layer for coordinating Active Inference workflows and processes.
Provides workflow management, task scheduling, and coordination between different
platform components.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents an orchestratable task"""
    id: str
    name: str
    function: Callable
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = datetime.now()


class WorkflowManager:
    """Manages complex workflows composed of multiple tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflows: Dict[str, List[Task]] = {}
        self.task_results: Dict[str, Any] = {}

        logger.info("WorkflowManager initialized")

    def create_workflow(self, workflow_name: str, tasks: List[Task]) -> str:
        """Create a new workflow"""
        workflow_id = f"workflow_{workflow_name}_{len(self.workflows)}"

        # Validate dependencies
        task_ids = {task.id for task in tasks}
        for task in tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    logger.warning(f"Task {task.id} has unknown dependency: {dep}")

        self.workflows[workflow_id] = tasks
        logger.info(f"Created workflow {workflow_id} with {len(tasks)} tasks")

        return workflow_id

    def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            return {"error": f"Workflow {workflow_id} not found"}

        logger.info(f"Executing workflow: {workflow_id}")

        tasks = self.workflows[workflow_id]
        results = {
            "workflow_id": workflow_id,
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "task_results": {}
        }

        # Execute tasks in dependency order
        executed = set()
        while len(executed) < len(tasks):
            # Find tasks that can be executed (no pending dependencies)
            ready_tasks = [
                task for task in tasks
                if task.id not in executed
                and all(dep in executed for dep in task.dependencies)
            ]

            if not ready_tasks:
                # Check for circular dependencies or other issues
                break

            # Execute ready tasks
            for task in ready_tasks:
                try:
                    logger.debug(f"Executing task: {task.name}")
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()

                    # Execute task function
                    task.result = task.function(**task.parameters)

                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    results["completed_tasks"] += 1

                except Exception as e:
                    logger.error(f"Task {task.name} failed: {e}")
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                    results["failed_tasks"] += 1

                executed.add(task.id)
                results["task_results"][task.id] = {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error
                }

        results["success"] = results["failed_tasks"] == 0
        logger.info(f"Workflow {workflow_id} completed: {results['completed_tasks']} completed, {results['failed_tasks']} failed")

        return results

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        if workflow_id not in self.workflows:
            return None

        tasks = self.workflows[workflow_id]

        return {
            "workflow_id": workflow_id,
            "total_tasks": len(tasks),
            "tasks_by_status": {
                status.value: len([t for t in tasks if t.status == status])
                for status in TaskStatus
            },
            "progress": len([t for t in tasks if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]]) / len(tasks) if tasks else 0
        }


class TaskScheduler:
    """Schedules and manages recurring tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        logger.info("TaskScheduler initialized")

    def schedule_task(self, task_id: str, function: Callable, interval: float,
                     parameters: Dict[str, Any] = None) -> bool:
        """Schedule a recurring task"""
        if task_id in self.scheduled_tasks:
            logger.warning(f"Task {task_id} already scheduled")
            return False

        self.scheduled_tasks[task_id] = {
            "function": function,
            "interval": interval,
            "parameters": parameters or {},
            "last_run": None,
            "next_run": datetime.now(),
            "running": False
        }

        logger.info(f"Scheduled task {task_id} with interval {interval}s")
        return True

    def start_scheduler(self) -> None:
        """Start the task scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("Task scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the task scheduler"""
        self.running = False

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)

        logger.info("Task scheduler stopped")

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self.running:
            current_time = datetime.now()

            # Check and execute due tasks
            for task_id, task_info in list(self.scheduled_tasks.items()):
                if current_time >= task_info["next_run"] and not task_info["running"]:
                    self._execute_scheduled_task(task_id, task_info)

            # Sleep for a short interval
            time.sleep(1.0)

    def _execute_scheduled_task(self, task_id: str, task_info: Dict[str, Any]) -> None:
        """Execute a scheduled task"""
        task_info["running"] = True
        task_info["last_run"] = datetime.now()

        try:
            logger.debug(f"Executing scheduled task: {task_id}")
            result = task_info["function"](**task_info["parameters"])

            # Store result (could be used for monitoring)
            task_info["last_result"] = result

        except Exception as e:
            logger.error(f"Scheduled task {task_id} failed: {e}")
            task_info["last_error"] = str(e)

        finally:
            # Schedule next run
            task_info["next_run"] = datetime.now().timestamp() + task_info["interval"]
            task_info["running"] = False

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "scheduled_tasks": len(self.scheduled_tasks),
            "tasks": {
                task_id: {
                    "interval": task_info["interval"],
                    "last_run": task_info["last_run"].isoformat() if task_info["last_run"] else None,
                    "next_run": datetime.fromtimestamp(task_info["next_run"]).isoformat() if task_info["next_run"] else None,
                    "running": task_info["running"]
                }
                for task_id, task_info in self.scheduled_tasks.items()
            }
        }


class Orchestrator:
    """Main orchestrator coordinating workflows and tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_manager = WorkflowManager(config.get("workflows", {}))
        self.task_scheduler = TaskScheduler(config.get("scheduler", {}))

        logger.info("Orchestrator initialized")

    def create_task(self, name: str, function: Callable, parameters: Dict[str, Any] = None,
                   dependencies: List[str] = None) -> Task:
        """Create a new task"""
        task_id = f"task_{name}_{int(datetime.now().timestamp())}"

        return Task(
            id=task_id,
            name=name,
            function=function,
            parameters=parameters or {},
            dependencies=dependencies or []
        )

    def execute_task(self, task: Task) -> Any:
        """Execute a single task"""
        logger.info(f"Executing task: {task.name}")

        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()

            result = task.function(**task.parameters)

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            logger.info(f"Task {task.name} completed successfully")
            return result

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            raise

    def create_and_execute_workflow(self, workflow_name: str, tasks: List[Task]) -> Dict[str, Any]:
        """Create and execute a workflow"""
        workflow_id = self.workflow_manager.create_workflow(workflow_name, tasks)
        return self.workflow_manager.execute_workflow(workflow_id)

    def schedule_recurring_task(self, task_id: str, function: Callable, interval: float,
                               parameters: Dict[str, Any] = None) -> bool:
        """Schedule a recurring task"""
        return self.task_scheduler.schedule_task(task_id, function, interval, parameters)

    def start_scheduled_execution(self) -> None:
        """Start scheduled task execution"""
        self.task_scheduler.start_scheduler()

    def stop_scheduled_execution(self) -> None:
        """Stop scheduled task execution"""
        self.task_scheduler.stop_scheduler()

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "workflow_manager": {
                "active_workflows": len(self.workflow_manager.workflows)
            },
            "task_scheduler": self.task_scheduler.get_scheduler_status(),
            "timestamp": datetime.now().isoformat()
        }
