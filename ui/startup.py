"""Start-up check of the web service settings, run by the Docker image before Streamlit.

Streamlit has no start-up hook: a settings error raised inside the app would only appear
in a browser session, while Cloud Run keeps serving the broken revision. This check runs
first (`python -m ui.startup && exec streamlit run ...`), so a production web service
without APP_BASE_URL or API_BASE_URL fails to start and the deploy keeps the previous
revision (invariant 5).
"""

from __future__ import annotations

import sys

from core.saas.settings import load_web_settings


def main() -> int:
    try:
        load_web_settings()
    except ValueError as exc:
        print(f"survey-insight-web: refusing to start: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
