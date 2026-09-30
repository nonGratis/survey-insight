"""Cloud Tasks worker service for async report jobs."""

from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from core.saas.container import SaaSContainer
from core.saas.errors import JobConflict
from core.saas.models import JobStatus, ReportStatus


class WorkerJobResponse(BaseModel):
    job_id: str
    status: str
    attempts: int


def create_worker_app(container: SaaSContainer | None = None) -> FastAPI:
    app = FastAPI(title="Survey Insight Worker", version="0.1.0")
    app.state.container = container or SaaSContainer.from_settings()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "survey-insight-worker"}

    @app.post("/tasks/reports/{job_id}", response_model=WorkerJobResponse)
    def run_report_job(job_id: str, request: Request) -> WorkerJobResponse:
        container = _container(request)
        _require_cloud_tasks_invocation(request, container)

        before = container.jobs.get(job_id)
        if before is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
        if before.status == JobStatus.SUCCEEDED:
            return _response(before)

        try:
            running = container.job_state_machine.start(job_id)
        except JobConflict as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

        if before.status == JobStatus.RUNNING:
            return _response(running)

        report = container.reports.get(running.report_id)
        if report is None:
            failed = container.job_state_machine.fail(
                job_id,
                error_code="report_not_found",
                error_summary="Report metadata is missing.",
                retryable=False,
            )
            return _response(failed)

        # Placeholder PDF keeps the worker path testable without persisting raw
        # Google Forms responses. The real report builder will run here once the
        # Google adapters and Cloud Storage adapter are wired.
        artifact = container.artifacts.save_pdf(
            report_id=report.id,
            user_id=report.user_id,
            content=b"%PDF-1.4\n% Survey Insight report placeholder\n",
        )
        container.reports.update(
            replace(
                report,
                status=ReportStatus.SUCCEEDED,
                artifact_uri=artifact.gcs_uri,
                updated_at=artifact.created_at,
            )
        )
        succeeded = container.job_state_machine.succeed(job_id)
        return _response(succeeded)

    return app


def _container(request: Request) -> SaaSContainer:
    return request.app.state.container


def _require_cloud_tasks_invocation(request: Request, container: SaaSContainer) -> None:
    if not container.settings.is_production:
        return
    authorization = request.headers.get("authorization", "")
    task_name = request.headers.get("x-cloudtasks-taskname", "")
    if not authorization.startswith("Bearer ") or not task_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="worker_requires_cloud_tasks_oidc",
        )


def _response(job) -> WorkerJobResponse:
    return WorkerJobResponse(
        job_id=job.id,
        status=job.status.value,
        attempts=job.attempts,
    )


app = create_worker_app()
