"""Synthetic AppTest coverage for running and resuming a detailed plan."""
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from config import MAX_REFLECTION_ATTEMPTS
from scripts.evaluate_agent import load_suite
from src.core.graph import build_graph
from src.core.state import ActionPlan, PlanStep, ProcedureState, UserEvent
from src.evaluation.router import _accepted_input, _mock_transport
from src.memory import checkpointer
from src.ui.components import profile_lock, start_detailed_plan
from test_ui_workspaces import choose, entered, ui_case


def show_plan_action(ui_case):
    ui_case.nav("next")
    ui_case.healthy()
    assert not any(toggle.label == "Detailed action plan" for toggle in ui_case.app.toggle)
    assert any(button.label == "Generate plan" for button in ui_case.app.button)


def wait_for_plan(ui_case):
    lock = profile_lock(ui_case.config["configurable"]["thread_id"])
    deadline = monotonic() + 10
    while lock.locked() and monotonic() < deadline:
        sleep(0.02)
    assert not lock.locked(), "Planning did not finish"
    ui_case.app.run()
    ui_case.healthy()


def planning_fixture(ui_case, name):
    suite = load_suite()
    spec = next(row for row in suite["scenarios"] if row["id"] == name)
    state = _accepted_input(suite, spec, UserEvent(kind="reevaluate"))
    state["user_id"] = ui_case.config["configurable"]["thread_id"]
    ui_case.seed(state)
    show_plan_action(ui_case)
    return suite, spec, state


def reopen_case(ui_case):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=25)
    app.run()
    choose(app, "text_input", "Profile name", ui_case.config["configurable"]["thread_id"])
    next(button for button in app.button if button.label == "Open my case").click()
    app.run()
    assert not app.exception
    return app


def test_one_click_allows_reruns_and_keeps_the_approval_on_reopen(ui_case):
    suite, spec, initial = planning_fixture(ui_case, "F")
    with checkpointer.sqlite_checkpointer() as saver:
        before = build_graph(saver).get_state(ui_case.config)
    started = Event()
    release = Event()
    with _mock_transport(suite, spec) as model:
        reply = model.side_effect

        def blocked_reply(**kwargs):
            started.set()
            assert release.wait(10)
            return reply(**kwargs)

        model.side_effect = blocked_reply
        try:
            ui_case.click("Generate plan")
            assert started.wait(5)
            ui_case.app.run()
            assert "Preparing your detailed plan" in ui_case.text()
            assert not any(button.label == "Generate plan" for button in ui_case.app.button)
            ui_case.click("Documents")
            assert ui_case.view() == "documents"
            assert "Preparing your detailed plan" in ui_case.text()
            assert not any(button.label == "Upload and analyze" for button in ui_case.app.button)
            with patch("src.ui.components.st.info"):
                assert not start_detailed_plan(before, ui_case.config)
            assert model.call_count == 1
        finally:
            release.set()
            wait_for_plan(ui_case)

        pending = ui_case.read()
        assert pending["run_status"] == "VALIDATED" and not pending.get("approved_plan")
        assert pending["documents"] == initial["documents"]
        reopened = reopen_case(ui_case)
        assert any(button.label == "Approve" and not button.disabled for button in reopened.button)
        assert not any(button.label == "Generate plan" for button in reopened.button)
        next(button for button in reopened.button if button.label == "Approve").click()
        reopened.run()
        assert not reopened.exception
        approved = ui_case.read()
        assert approved["approved_plan"].actions
        assert approved["documents"] == initial["documents"]
        again = reopen_case(ui_case)
        assert not any(button.label in {"Generate plan", "Update plan"} for button in again.button)
        assert any("Upload Passport Identity Page" in str(item.value) for item in again.markdown)
        assert model.call_count == 1


def test_plan_for_an_earlier_stage_is_not_presented_as_the_current_plan(ui_case):
    earlier_step = PlanStep(
        step_id="old-task", title="Upload the earlier document", action_type="REQUEST_DOCUMENT",
        priority=1, reason="Earlier stage evidence",
    )
    plan = ActionPlan(actions=[earlier_step], summary="Earlier stage")
    ui_case.seed(entered({
        "procedure": ProcedureState(current_stage="EIKAMET_PREPARATION"),
        "workflow_stage": "DOCUMENT_COLLECTION",
        "run_status": "APPROVED",
        "approved_plan": plan,
        "user_id": "demo-user",
        "trace_id": "previous-stage",
    }))
    with patch("ollama.Client.chat") as model:
        show_plan_action(ui_case)
        model.assert_not_called()
    assert "Upload the earlier document" not in ui_case.text()
    assert ui_case.read()["approved_plan"] == plan


def test_schema_rejection_and_reflection_progress_then_valid_approval(ui_case):
    suite, spec, initial = planning_fixture(ui_case, "F-repair")
    repairing = Event()
    release = Event()
    with _mock_transport(suite, spec) as model:
        reply = model.side_effect

        def pause_repair(**kwargs):
            if model.call_count == 2:
                repairing.set()
                assert release.wait(10)
            return reply(**kwargs)

        model.side_effect = pause_repair
        try:
            ui_case.click("Generate plan")
            assert repairing.wait(5)
            ui_case.app.run()
            assert "Checking the plan and correcting inconsistencies" in ui_case.text()
        finally:
            release.set()
            wait_for_plan(ui_case)
        assert model.call_count == 2
    state = ui_case.read()
    assert state["retry_count"] == 1 and state["run_status"] == "VALIDATED"
    assert not state["validation_errors"]
    assert state["documents"] == initial["documents"]
    assert any(button.label == "Approve" and not button.disabled for button in ui_case.app.button)


def test_reflection_exhaustion_leaves_a_retry_and_preserves_evidence(ui_case):
    _, _, initial = planning_fixture(ui_case, "F")
    with patch("ollama.Client.chat", return_value=SimpleNamespace(message=SimpleNamespace(content="{"))) as model:
        ui_case.click("Generate plan")
        wait_for_plan(ui_case)
        assert model.call_count == 1 + MAX_REFLECTION_ATTEMPTS
    state = ui_case.read()
    assert state["run_status"] == "HUMAN_REVIEW_REQUIRED"
    assert state["retry_count"] == MAX_REFLECTION_ATTEMPTS
    assert state["validation_errors"][0].code == "PLANNER_SCHEMA_ERROR"
    assert "could not prepare a plan that passed the checks" in ui_case.text()
    assert not any(button.label == "Approve" for button in ui_case.app.button)
    assert any(button.label == "Generate plan" and not button.disabled for button in ui_case.app.button)
    assert state["documents"] == initial["documents"]


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("Synthetic timeout"), ConnectionError("Synthetic connection failure")])
def test_transport_failure_keeps_the_validated_fallback_subject_to_approval(ui_case, failure):
    _, _, initial = planning_fixture(ui_case, "H")
    with patch("ollama.Client.chat", side_effect=failure) as model:
        ui_case.click("Generate plan")
        wait_for_plan(ui_case)
        assert model.call_count == 1
    state = ui_case.read()
    assert state["plan_origin"] == "deterministic_fallback" and state["run_status"] == "VALIDATED"
    assert not state["approved_plan"]
    assert state["documents"] == initial["documents"]
    assert "The local model was unavailable" in ui_case.text()
    assert any(button.label == "Approve" and not button.disabled for button in ui_case.app.button)


def test_completed_case_explains_empty_result_without_inference_or_approval(ui_case):
    ui_case.seed(entered({"procedure": ProcedureState(current_stage="COMPLETE"), "user_id": "demo-user",
                         "trace_id": "synthetic-completed"}))
    ui_case.nav("next")
    assert not any(button.label == "Generate plan" for button in ui_case.app.button)
    # A deliberate request from the existing Copilot shortcut also works in a completed case.
    ui_case.app.session_state[f"show_plan_{ui_case.key}"] = True
    show_plan_action(ui_case)
    with patch("ollama.Client.chat", side_effect=AssertionError("A completed case needs no inference")) as model:
        ui_case.click("Generate plan")
        wait_for_plan(ui_case)
        model.assert_not_called()
    state = ui_case.read()
    assert state["candidate_plan"] == [] and state["plan_origin"] == "stage_complete"
    assert state["plan_summary"] in ui_case.text()
    assert not any(button.label in {"Generate plan", "Approve"} for button in ui_case.app.button)


def test_interrupted_worker_releases_the_case_and_resumes_through_the_same_request(ui_case):
    suite, spec, initial = planning_fixture(ui_case, "F")
    with patch("ollama.Client.chat", side_effect=RuntimeError("Synthetic worker interrupted")):
        ui_case.click("Generate plan")
        wait_for_plan(ui_case)
    assert "Plan preparation stopped" in ui_case.text()
    assert ui_case.read()["documents"] == initial["documents"]
    trace = ui_case.read()["trace_id"]
    other_tab = reopen_case(ui_case)
    with _mock_transport(suite, spec) as model:
        next(button for button in other_tab.button if button.label == "Retry unfinished processing").click()
        other_tab.run()
        assert not other_tab.exception
        wait_for_plan(ui_case)
        assert model.call_count == 1
    state = ui_case.read()
    assert state["trace_id"] == trace
    assert state["run_status"] == "VALIDATED"
    assert "Plan preparation stopped" not in ui_case.text()
    assert state["documents"] == initial["documents"]
