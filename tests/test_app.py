"""app.py: pure helpers (ported from milestone-1 test_streamlit_helpers) and a headless smoke run."""
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

import app
import config
from src.pack import load_pack
from src.twin import service
from src.twin.facts import build_view
from src.twin.store import TwinStore


def test_save_upload_uses_content_names_inside_the_project(tmp_path):
    folder = config.PROJECT_ROOT / ".test-temp" / "uploads"
    path = app.save_upload("../../evil name.PDF", b"%PDF-1.4 synthetic", "u", folder)
    assert path.parent == folder.resolve() and path.suffix == ".pdf" and ".." not in path.name
    assert path == app.save_upload("other.pdf", b"%PDF-1.4 synthetic", "u", folder)


@pytest.mark.parametrize("name,content,user", [("a.exe", b"x", "u"), ("a.txt", b"", "u")])
def test_save_upload_rejects_bad_files(name, content, user):
    with pytest.raises(ValueError):
        app.save_upload(name, content, user, config.PROJECT_ROOT / ".test-temp" / "uploads")


def test_save_upload_refuses_folders_outside_the_project(tmp_path):
    outside = config.PROJECT_ROOT.parent / "outside_uploads"
    with pytest.raises(ValueError):
        app.save_upload("a.txt", b"x", "u", outside)


@pytest.mark.parametrize("message,expected", [
    ("RapidOCR failed on page 1", "scanned pages"), ("Local Ollama timed out during extraction", "timed out"),
    ("Ollama unavailable: connection refused", "not running"), ("PDF contains no extractable text", "No readable text"),
    ("something else", "could not finish")])
def test_friendly_errors(message, expected):
    assert expected in app.friendly_error(message)


def test_fact_rows_show_provenance_and_derived_values():
    pack = load_pack("ikamet_first")
    with TwinStore(":memory:") as store:
        service.record_app_value(store, pack, "u", "age", 22)
        view = build_view(pack, store.current_facts("u", pack.id), [], date(2026, 11, 18))
    rows = {row["field"]: row for row in app.fact_rows(view)}
    assert rows["age"]["from"] == "user_stated (app)" and rows["current_date"]["from"] == "computed"


def test_rule_rows_mark_status_and_missing_fields():
    rows = app.rule_rows(load_pack("ikamet_first"), {}, "passport_photo")
    assert [r["rule"] for r in rows] == ["R006", "R008", "R009"] and rows[0]["status"].endswith("unknown")
    assert "passport_expiry_date" in rows[0]["missing"]


def test_app_renders_without_errors_on_an_empty_and_a_seeded_twin(tmp_path, monkeypatch):
    database = tmp_path / "twin.sqlite"
    monkeypatch.setattr(config, "DATABASE_PATH", database)
    pack = load_pack("ikamet_first")
    with TwinStore(database) as store:
        service.record_app_value(store, pack, "demo-user", "passport_expiry_date", "2027-07-15")
    empty = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not empty.exception
    with TwinStore(database) as store:
        store.set_active_procedure("demo-user", "ikamet_first")
    seeded = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not seeded.exception
    headers = [h.value for h in seeded.header]
    assert "4. Where you are" in headers and "6. Proposed actions (your approval is required)" in headers
