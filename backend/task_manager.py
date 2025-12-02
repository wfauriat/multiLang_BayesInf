import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import traceback

class TaskStatus:
    """Represents the status of a background task"""

    PENDING = 'PENDING'
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.state = self.PENDING
        self.progress = 0
        self.status_message = 'Task queued'
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'state': self.state,
            'progress': self.progress,
            'status': self.status_message,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class TaskManager:
    """Manages background tasks using threading"""

    def __init__(self):
        self.tasks: Dict[str, TaskStatus] = {}
        self.lock = threading.Lock()

    def create_task(self) -> str:
        """Create a new task and return its ID"""
        task_id = str(uuid.uuid4())
        with self.lock:
            self.tasks[task_id] = TaskStatus(task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status by ID"""
        with self.lock:
            return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        """Update task attributes"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                for key, value in kwargs.items():
                    setattr(task, key, value)

    def run_task(self, task_id: str, func, *args, **kwargs):
        """Run a function in a background thread"""
        def wrapper():
            task = self.get_task(task_id)
            if not task:
                return

            try:
                # Mark as running
                self.update_task(
                    task_id,
                    state=TaskStatus.RUNNING,
                    status_message='Computation started',
                    started_at=datetime.now()
                )

                # Execute the function
                result = func(task_id, *args, **kwargs)

                # Mark as successful
                self.update_task(
                    task_id,
                    state=TaskStatus.SUCCESS,
                    status_message='Computation completed',
                    result=result,
                    progress=100,
                    completed_at=datetime.now()
                )

            except Exception as e:
                # Mark as failed
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.update_task(
                    task_id,
                    state=TaskStatus.FAILURE,
                    status_message='Computation failed',
                    error=error_msg,
                    completed_at=datetime.now()
                )
                print(f"Task {task_id} failed:")
                traceback.print_exc()

        # Start thread
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()

    def cleanup_old_tasks(self, max_age_seconds: int = 3600):
        """Remove tasks older than max_age_seconds"""
        now = datetime.now()
        with self.lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.completed_at:
                    age = (now - task.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self.tasks[task_id]


# Global task manager instance
task_manager = TaskManager()
