# Cloud Run MVP Deploy

This setup runs one Docker image in three Cloud Run services:

- `survey-insight-web` with `SERVICE=web`
- `survey-insight-api` with `SERVICE=api`
- `survey-insight-worker` with `SERVICE=worker`

Current GCP resources:

```txt
Project ID: survey-insight
Cloud Run region: europe-central2
Firestore database: (default)
Firestore location: eur3
GCS bucket: survey-insight-reports-1046685202661
GCS location: eu
Cloud Tasks queue: report-jobs
Cloud Tasks location: europe-central2
KMS key: projects/survey-insight/locations/global/keyRings/survey-insight/cryptoKeys/oauth-tokens
```

Secrets must stay in Secret Manager:

```txt
SESSION_PEPPER
GOOGLE_OAUTH_CLIENT_CONFIG_JSON
```

## Required Service Accounts

Recommended accounts:

```txt
survey-insight-api@survey-insight.iam.gserviceaccount.com
survey-insight-worker@survey-insight.iam.gserviceaccount.com
survey-insight-tasks@survey-insight.iam.gserviceaccount.com
```

Minimum IAM:

```txt
api:
- Cloud Datastore User
- Cloud Tasks Enqueuer
- Cloud KMS CryptoKey Encrypter/Decrypter
- Secret Manager Secret Accessor

worker:
- Cloud Datastore User
- Storage Object User on the reports bucket
- Cloud KMS CryptoKey Encrypter/Decrypter
- Secret Manager Secret Accessor

tasks:
- Cloud Run Invoker on survey-insight-worker
```

## Image Build

```powershell
gcloud builds submit --tag europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest
```

The Artifact Registry repository must exist before this command.

## Deploy API

Bootstrap note: the first deploy may use temporary HTTPS placeholders for
`API_BASE_URL` and `WORKER_BASE_URL`. After Cloud Run returns real service URLs,
run `gcloud run services update` or redeploy with the final values.

```powershell
gcloud run deploy survey-insight-api `
  --image europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest `
  --region europe-central2 `
  --service-account survey-insight-api@survey-insight.iam.gserviceaccount.com `
  --set-env-vars SERVICE=api,APP_ENV=production,APP_BASE_URL=https://<web-run-url>,API_BASE_URL=https://<api-run-url>,WORKER_BASE_URL=https://<worker-run-url>,GCP_PROJECT_ID=survey-insight,FIRESTORE_DATABASE="(default)",KMS_KEY_NAME=projects/survey-insight/locations/global/keyRings/survey-insight/cryptoKeys/oauth-tokens,GCS_BUCKET=survey-insight-reports-1046685202661,CLOUD_TASKS_LOCATION=europe-central2,TASKS_QUEUE_NAME=report-jobs,CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL=survey-insight-tasks@survey-insight.iam.gserviceaccount.com `
  --set-secrets SESSION_PEPPER=SESSION_PEPPER:latest,GOOGLE_OAUTH_CLIENT_CONFIG_JSON=GOOGLE_OAUTH_CLIENT_CONFIG_JSON:latest
```

After deploy, copy the API URL and set:

```txt
API_BASE_URL=https://<api-run-url>
```

Also add OAuth redirect URI:

```txt
https://<api-run-url>/v1/auth/google/callback
```

## Deploy Worker

```powershell
gcloud run deploy survey-insight-worker `
  --image europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest `
  --region europe-central2 `
  --no-allow-unauthenticated `
  --service-account survey-insight-worker@survey-insight.iam.gserviceaccount.com `
  --set-env-vars SERVICE=worker,APP_ENV=production,APP_BASE_URL=https://<web-run-url>,API_BASE_URL=https://<api-run-url>,WORKER_BASE_URL=https://<worker-run-url>,GCP_PROJECT_ID=survey-insight,FIRESTORE_DATABASE="(default)",KMS_KEY_NAME=projects/survey-insight/locations/global/keyRings/survey-insight/cryptoKeys/oauth-tokens,GCS_BUCKET=survey-insight-reports-1046685202661,CLOUD_TASKS_LOCATION=europe-central2,TASKS_QUEUE_NAME=report-jobs,CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL=survey-insight-tasks@survey-insight.iam.gserviceaccount.com `
  --set-secrets SESSION_PEPPER=SESSION_PEPPER:latest,GOOGLE_OAUTH_CLIENT_CONFIG_JSON=GOOGLE_OAUTH_CLIENT_CONFIG_JSON:latest
```

After deploy, copy the worker URL and set:

```txt
WORKER_BASE_URL=https://<worker-run-url>
CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL=survey-insight-tasks@survey-insight.iam.gserviceaccount.com
```

The API service needs those values so it can enqueue Cloud Tasks with OIDC.

## Deploy Web

```powershell
gcloud run deploy survey-insight-web `
  --image europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest `
  --region europe-central2 `
  --set-env-vars SERVICE=web,APP_ENV=production,APP_BASE_URL=https://<web-run-url>,API_BASE_URL=https://<api-run-url>,WORKER_BASE_URL=https://<worker-run-url>
```

The current Streamlit UI still uses the older local OAuth widget. Full SaaS web auth requires the next step: Streamlit session bridge against the FastAPI session endpoint.
