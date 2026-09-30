"""Cloud Tasks queue adapter for async report jobs."""

from __future__ import annotations

from typing import Any

from google.cloud import tasks_v2


class CloudTasksQueue:
    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        queue_name: str,
        worker_base_url: str,
        service_account_email: str,
        client: Any | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("GCP project id is required.")
        if not location:
            raise ValueError("Cloud Tasks location is required.")
        if not queue_name:
            raise ValueError("Cloud Tasks queue name is required.")
        if not worker_base_url:
            raise ValueError("Worker base URL is required.")
        if not service_account_email:
            raise ValueError("Cloud Tasks service account email is required.")
        self.project_id = project_id
        self.location = location
        self.queue_name = queue_name
        self.worker_base_url = worker_base_url.rstrip("/")
        self.service_account_email = service_account_email
        self.client = client or tasks_v2.CloudTasksClient()

    def enqueue_report_job(self, job_id: str) -> None:
        parent = self.client.queue_path(self.project_id, self.location, self.queue_name)
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{self.worker_base_url}/tasks/reports/{job_id}",
                "headers": {"Content-Type": "application/json"},
                "oidc_token": {
                    "service_account_email": self.service_account_email,
                    "audience": self.worker_base_url,
                },
            }
        }
        self.client.create_task(request={"parent": parent, "task": task})
