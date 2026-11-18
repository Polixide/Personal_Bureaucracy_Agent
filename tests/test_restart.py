"""Acceptance criterion: user state survives an application restart.

The strongest version runs each session in a separate Python process, so no
object, cache or connection can carry state over: only the SQLite file can.
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from src.core.graph import build_graph
from src.core.state import UserEvent
from src.memory.twin_store import TwinStore
from src.memory.session import run_event, start_state

ROOT = Path(__file__).resolve().parents[1]

SESSION_SCRIPT = """
import json, sys
from datetime import date
from src.core.graph import build_graph
from src.core.state import UserEvent
from src.memory.twin_store import TwinStore
from src.memory.session import run_event

db, action = sys.argv[1], sys.argv[2]
with TwinStore(db) as store:
    if action == "seed":
        store.save_twin({"user_id": "student_001", "active_procedure": "ikamet",
                         "extracted_facts": {"appointment_date": date(2026, 11, 20)}})
    elif action == "upload":
        final = run_event(build_graph(), store, "student_001",
                          UserEvent(kind="upload", file_path=sys.argv[3]))
        print(json.dumps([e.code for e in final["validation_errors"]]))
    elif action == "dump":
        twin = store.load_twin("student_001")
        print(json.dumps({
            "documents": {d.doc_id: [d.status, str(d.expiry_date)] for d in twin["documents"]},
            "appointment": [type(twin["extracted_facts"]["appointment_date"]).__name__,
                            str(twin["extracted_facts"]["appointment_date"])],
            "blockers": twin["blockers"],
            "runs": [r.outcome for r in store.list_runs("student_001")],
        }))
"""


def session(db, *args) -> str:
    """Run one 'application session' in a brand-new interpreter."""
    done = subprocess.run([sys.executable, "-c", SESSION_SCRIPT, str(db), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_state_survives_process_restarts(db_path):
    session(db_path, "seed")
    assert json.loads(session(db_path, "upload", "insurance_expiring.pdf")) == ["DATE_CONFLICT"]
    assert json.loads(session(db_path, "upload", "insurance_valid.pdf")) == []
    state = json.loads(session(db_path, "dump"))
    assert state == {
        "documents": {"insurance_expiring.pdf": ["analyzed", "2026-11-10"],
                      "insurance_valid.pdf": ["analyzed", "2027-11-01"]},
        "appointment": ["date", "2026-11-20"],          # still a date, not a string
        "blockers": ["REQ_STUDENT_CERTIFICATE"],
        "runs": ["fail", "pass"],
    }


def test_remembered_document_is_not_analysed_again(db_path):
    """After a restart, the supervisor sees the stored analyzed document and skips the analyzer."""
    with TwinStore(db_path) as store:
        run_event(build_graph(), store, "s1", UserEvent(kind="upload",
                                                        file_path="student_certificate.pdf"))
    path: list[str] = []
    with TwinStore(db_path) as store:                    # new connection, new graph
        run_event(build_graph(), store, "s1", UserEvent(kind="message", text="next?"), path=path)
    assert "document_analyzer" not in path
    assert "rules_engine" in path


def test_start_state_resets_run_memory_but_keeps_the_twin(db_path):
    with TwinStore(db_path) as store:
        store.save_twin({"user_id": "s1", "active_procedure": "ikamet", "blockers": ["R1"],
                         "extracted_facts": {"appointment_date": date(2026, 11, 20)}})
        state = start_state(store, "s1", UserEvent(kind="message", text="hi"))
    assert state["blockers"] == ["R1"]
    assert state["extracted_facts"] == {"appointment_date": date(2026, 11, 20)}
    assert state["candidate_plan"] is None and state["validation_errors"] == []
    assert state["retry_count"] == 0
