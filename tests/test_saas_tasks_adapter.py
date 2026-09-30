from __future__ import annotations

import pytest
from google.cloud import tasks_v2

from core.saas.adapters.tasks import CloudTasksQueue


class _FakeTasksClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def queue_path(self, project_id: str, location: str, queue_name: str) -> str:
        return f"projects/{project_id}/locations/{location}/queues/{queue_name}"

    def create_task(self, request: dict) -> None:
        self.requests.append(request)


def test_cloud_tasks_queue_enqueues_report_job_with_oidc() -> None:
    client = _FakeTasksClient()
    queue = CloudTasksQueue(
        project_id="survey-insight",
        location="europe-central2",
        queue_name="report-jobs",
        worker_base_url="https://worker.run.app/",
        service_account_email="tasks@survey-insight.iam.gserviceaccount.com",
        client=client,
    )

    queue.enqueue_report_job("job_1")

    request = client.requests[0]
    assert (
        request["parent"] == "projects/survey-insight/locations/europe-central2/queues/report-jobs"
    )
    http_request = request["task"]["http_request"]
    assert http_request["http_method"] == tasks_v2.HttpMethod.POST
    assert http_request["url"] == "https://worker.run.app/tasks/reports/job_1"
    assert http_request["oidc_token"] == {
        "service_account_email": "tasks@survey-insight.iam.gserviceaccount.com",
        "audience": "https://worker.run.app",
    }


@pytest.mark.parametrize(
    ("field", "kwargs", "message"),
    [
        ("project_id", {"project_id": ""}, "project id"),
        ("location", {"location": ""}, "location"),
        ("queue_name", {"queue_name": ""}, "queue name"),
        ("worker_base_url", {"worker_base_url": ""}, "Worker base URL"),
        ("service_account_email", {"service_account_email": ""}, "service account"),
    ],
)
def test_cloud_tasks_queue_requires_production_fields(
    field: str,
    kwargs: dict[str, str],
    message: str,
) -> None:
    defaults = {
        "project_id": "survey-insight",
        "location": "europe-central2",
        "queue_name": "report-jobs",
        "worker_base_url": "https://worker.run.app",
        "service_account_email": "tasks@survey-insight.iam.gserviceaccount.com",
        "client": _FakeTasksClient(),
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CloudTasksQueue(**defaults)

    assert field in defaults
