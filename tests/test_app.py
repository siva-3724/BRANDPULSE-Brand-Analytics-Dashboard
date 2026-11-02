import pytest

from pathlib import Path

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_renders_without_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("BRANDPULSE_DB", str(tmp_path / "app.db"))
    import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")
    at = st_testing.AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    # load demo data via the sidebar button, then re-render
    next(b for b in at.sidebar.button if "demo history" in b.label).click()
    at.run()
    assert not at.exception
    assert any("Total mentions" in m.label for m in at.metric)
