"""«Каталог»: вибір форми під час довантаження деталей. Справжні web і API, фейковий Google."""

from __future__ import annotations

from tests.test_catalog_page import (  # noqa: F401 - the fixture switches the web to production
    _app_with,
    _ManyOpenForms,
    _production_web,
    _seed_google_grant,
    _seed_user_session,
    _signed_in_web,
    _test_container,
    _web_talking_to,
)


def test_choosing_a_form_does_not_wait_for_the_next_loading_step() -> None:
    container = _test_container()
    session_id = _seed_user_session(container)
    _seed_google_grant(container)
    _, api_app = _app_with(container, _ManyOpenForms())

    def loaded(at) -> int:
        return int((at.dataframe[0].value["DataStatus"] != "Завантажується").sum())

    with _web_talking_to(api_app):
        at = _signed_in_web(session_id)
        at.run()
        before = loaded(at)
        at.selectbox(key="global_form_select_catalog").set_value("form_10").run()
        after_choice = loaded(at)
        at.run()  # the loading timer's next tick
        after_tick = loaded(at)

    assert not at.exception, [e.value for e in at.exception]
    # The run that changes the form only redraws the page; a loading step there kept the
    # click waiting 2-5 s for Google.
    assert after_choice == before
    assert after_tick > before
