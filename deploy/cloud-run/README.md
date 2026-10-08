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
$tag = git rev-parse --short HEAD
gcloud builds submit --config deploy/cloud-run/cloudbuild.yaml `
  --substitutions "_IMAGE=europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest,_APP_VERSION=$tag,_APP_BUILD_DATE=$(Get-Date -Format yyyy-MM-dd)"
```

The Artifact Registry repository must exist before this command. Run it from the repository root of a clean checkout: the commit SHA and the build date are baked into the image, the web sidebar shows them, and `/health` of the API and the worker returns the SHA as `version`. An image built with plain `docker build` or `gcloud builds submit --tag` reports `dev`.

## Deploy API

Bootstrap note: the first deploy may use temporary HTTPS placeholders for
`API_BASE_URL` and `WORKER_BASE_URL`. After Cloud Run returns real service URLs,
run `gcloud run services update` or redeploy with the final values.

```powershell
gcloud run deploy survey-insight-api `
  --image europe-central2-docker.pkg.dev/survey-insight/survey-insight/app:latest `
  --region europe-central2 `
  --memory 1Gi `
  --min-instances 1 `
  --service-account survey-insight-api@survey-insight.iam.gserviceaccount.com `
  --set-env-vars SERVICE=api,APP_ENV=production,APP_BASE_URL=https://<web-run-url>,API_BASE_URL=https://<api-run-url>,WORKER_BASE_URL=https://<worker-run-url>,GCP_PROJECT_ID=survey-insight,FIRESTORE_DATABASE="(default)",KMS_KEY_NAME=projects/survey-insight/locations/global/keyRings/survey-insight/cryptoKeys/oauth-tokens,GCS_BUCKET=survey-insight-reports-1046685202661,CLOUD_TASKS_LOCATION=europe-central2,TASKS_QUEUE_NAME=report-jobs,CLOUD_TASKS_SERVICE_ACCOUNT_EMAIL=survey-insight-tasks@survey-insight.iam.gserviceaccount.com `
  --set-secrets SESSION_PEPPER=SESSION_PEPPER:latest,GOOGLE_OAUTH_CLIENT_CONFIG_JSON=GOOGLE_OAUTH_CLIENT_CONFIG_JSON:latest
```

`--memory 1Gi`: a catalog load runs 30 Google calls in parallel (`SI_CATALOG_STREAM_WORKERS`), and with 512 MiB the API kept ~85 % of its memory after a load and was restarted by Cloud Run for running out of it (2026-10-07). `--min-instances 1` keeps one instance warm: a cold start cost the first page ~6 s, and the API's Google data cache, quota guard and single-flight of identical loads live in that instance's memory.

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
  --session-affinity `
  --timeout 3600 `
  --min 1 `
  --max 1 `
  --set-env-vars SERVICE=web,APP_ENV=production,APP_BASE_URL=https://<web-run-url>,API_BASE_URL=https://<api-run-url>,WORKER_BASE_URL=https://<worker-run-url>
```

With `APP_ENV=production` the web service signs users in through the API (Google OAuth on the API, then a login ticket exchanged for a session) and reads all Google data through it. It needs `APP_BASE_URL` and `API_BASE_URL`, both HTTPS: without them the container refuses to start (`ui/startup.py` runs before Streamlit), the revision never becomes ready and traffic stays on the previous one. The local demo sign-in, where Streamlit talks to Google directly and holds the tokens, exists only outside production.

`--session-affinity` keeps a browser on the instance that holds its Streamlit session. Streamlit keeps download files (`st.download_button`) and component assets in that instance's memory, so without affinity a second instance answers those requests with 404: the PDF download fails as an empty file named by a hash. Affinity is best effort, so an instance shutting down can still cut a session; serving reports through the API would remove the dependency. Redeploys with `--image` keep the setting.

`--timeout 3600`: a Streamlit page keeps one websocket open for the whole visit, and Cloud Run ends every request at its timeout (300 s by default), which cut the page's connection every five minutes. `--min 1 --max 1` (service level): one always-on instance holds every Streamlit session, so a page never waits for a cold start and no session lands on an instance that does not know it.

## Changing settings later

`--set-env-vars` replaces every variable of the service: running a deploy command above again drops any variable added since. To change one variable, use `--update-env-vars` (and `--remove-env-vars` to drop one):

```powershell
gcloud run services update survey-insight-api --region europe-central2 --update-env-vars SI_CATALOG_STREAM_WORKERS=20
```

Redeploying only a new image (`gcloud run deploy <service> --image ...` without the flags above) keeps memory, instances, timeout, affinity and variables.

## Production settings

As of 2026-10-08 (`gcloud run services describe <service> --region europe-central2`):

| Service | CPU | Memory | Instances | Request timeout | Notes |
|---|---|---|---|---|---|
| `survey-insight-api` | 1 | 1 GiB | min 1, max 20 | 300 s | public; a catalog load ends by itself after 240 s |
| `survey-insight-web` | 1 | 512 MiB | exactly 1 (service level) | 3600 s | session affinity; CPU only during requests |
| `survey-insight-worker` | 1 | 512 MiB | max 20 | 300 s | invoked by Cloud Tasks only (`--no-allow-unauthenticated`) |

The API's cache, quota guard and single-flight are per instance. With more than one API instance they still work, but each instance keeps its own: two instances can ask Google for the same form, and each spends up to the per-user quota.

## Tuning variables

Optional; the defaults are what production runs. Set them with `--update-env-vars` on the service that reads them.

| Variable | Service | Default | What it sets |
|---|---|---|---|
| `SI_CATALOG_STREAM_WORKERS` | api | `30` | Google calls in flight in one catalog load. Above 30 is untested against Google's bursts; watch `google_429_count`. |
| `SI_CATALOG_STREAM_DEADLINE_SECONDS` | api | `240` | A catalog load stops here; what is left ends as `timeout`. Keep it below the request timeout. |
| `SI_CATALOG_STREAM_MAX_FORMS` | api | `1000` | Most forms one catalog load accepts. |
| `SI_GOOGLE_CALL_TIMEOUT_SECONDS` | api | `10` | A catalog call Google does not answer in this time is asked once more, then ends as `timeout`. |
| `SI_SLOW_GOOGLE_CALL_MS` | api | `5000` | Catalog Google calls at least this slow get a `google_call_slow` log line. |
| `SI_FORMS_READS_PER_MINUTE` | api | `300` | Per-user `forms.get` calls a minute (Google allows 390). |
| `SI_FORMS_RESPONSE_LISTS_PER_MINUTE` | api | `140` | Per-user `forms.responses.list` calls a minute (Google allows 180). |
| `SI_API_CATALOG_SUMMARY_TTL_SECONDS` | api | `600` | How long a form's details stay cached. |
| `SI_API_RESPONSE_STATS_TTL_SECONDS` | api | `120` | How long an open form's response count stays cached. |
| `SI_API_CLOSED_FORM_STATS_TTL_SECONDS` | api | `21600` | How long the count of a form that accepts no responses stays cached. |
| `SI_DRIVE_FORMS_PAGE_SIZE` | api | `1000` | Forms per Drive list page (Drive allows 1000). |
| `SI_DRIVE_FORMS_MAX` | api | `1000` | Most forms the catalog lists. |
| `SI_RAW_RESPONSES_CACHE_MAX_ROWS` | web | `10000` | Largest response set the web keeps in its cache. |
| `SI_RAW_RESPONSES_CACHE_MAX_BYTES` | web | `8000000` | The same limit in bytes. |

`LOG_LEVEL` (default `INFO`) sets the log level of every service.

## Operations

Every service writes one JSON line per event to stderr; Cloud Logging puts the event name in `jsonPayload.msg` and the level in `severity`. Web lines carry `session_ref` (a digest of the Streamlit session) and `user_id` (a digest of the user); no line carries tokens or emails, and the catalog lines carry no form ids.

| Event | Service | What it tells |
|---|---|---|
| `forms_catalog_stream_completed` | api | One per catalog load: `summaries_ms` (all statuses in), `counts_ms` (all counts in), `duration_ms`, Google calls with their median and slowest time, `cache_hit_count`, `quota_wait_events`, `google_429_count`, `timeout_count`. |
| `google_call_slow` | api | A catalog Google call slower than `SI_SLOW_GOOGLE_CALL_MS`, with its `target` and `duration_ms`. |
| `api_request_timing` | api | Every API request: path, status, duration. |
| `ui_page_run` | web | Every page run: `page`, `run_kind` (full or fragment), `outcome`, `duration_ms`. |
| `ui_saas_api_request` | web | Every call from the web to the API, with its duration and error code. |
| `ui_session_restored_from_cookie`, `ui_session_cookie_*` | web | Sign-in restored from the cookie, cookie written or deleted. |

Run the queries where `gcloud` is signed in. In PowerShell 5.1 a filter with inner double quotes breaks, so use Git Bash or Cloud Shell:

```bash
# Catalog loads of the last day
gcloud logging read 'jsonPayload.msg="forms_catalog_stream_completed"' --project survey-insight --freshness=1d --format="value(timestamp,jsonPayload.summaries_ms,jsonPayload.counts_ms,jsonPayload.google_get_ms_p50,jsonPayload.google_429_count)"

# Slow Google calls, warnings and errors of the last day
gcloud logging read 'resource.type="cloud_run_revision" AND severity>=WARNING' --project survey-insight --freshness=1d

# Instances restarted for running out of memory
gcloud logging read 'textPayload:"Memory limit"' --project survey-insight --freshness=7d
```

Google's own latency and status codes for the Forms API are in Cloud Monitoring: metrics `serviceruntime.googleapis.com/api/request_latencies` and `api/request_count` on resource `consumed_api`, service `forms.googleapis.com`. Compare them with `google_get_ms_p50` to tell Google's time from the API's. Instance memory is `run.googleapis.com/container/memory/utilizations`.
