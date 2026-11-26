"""Run the existing Streamlit entry point against isolated synthetic cases."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import evaluate_agent
from test_procedure import ready_bundle
from test_ui_workspaces import choose, entered, ui_case


def _settings(case):
    choose(case.app, "toggle", "Case settings", True)
    case.app.run()
    choose(case.app, "selectbox", "AI model", "openai")
    case.app.run()
    case.healthy()


def test_cloud_copilot_uses_one_request_and_preserves_case_on_rerun(ui_case, ready_bundle, monkeypatch):
    monkeypatch.setattr("src.ui.model_settings.OPENAI_MODEL", "synthetic-cloud-model")
    monkeypatch.setattr("src.agents.assistant.OPENAI_MODEL", "synthetic-cloud-model")
    monkeypatch.setattr("src.ui.model_settings.cloud_configuration_issue", lambda *a: None)
    reply = SimpleNamespace(output_text='{"source_indices":[0]}', status="completed",
                            usage=SimpleNamespace(input_tokens=24, output_tokens=4, total_tokens=28))
    create = Mock(return_value=reply)
    client = SimpleNamespace(responses=SimpleNamespace(create=create), close=Mock())
    monkeypatch.setattr("src.core.model_provider.create_openai_client", lambda timeout: client)

    ui_case.seed(entered(ready_bundle[2]))
    ui_case.nav("copilot")
    before = ui_case.read()
    _settings(ui_case)
    assert "Leave out personal details" in ui_case.text()
    assert create.call_count == 0
    ui_case.app.chat_input[0].set_value("Why is passport validity required?")
    ui_case.app.run()
    ui_case.healthy()
    assert "60 days" in ui_case.text()
    assert any(row.label == "Sources used" for row in ui_case.app.get("expander"))
    assert create.call_count == 1
    assert ui_case.read()["documents"] == before["documents"]
    assert ui_case.read()["residence_permit_context"] == before["residence_permit_context"]

    ui_case.click("Advanced diagnostics")
    choose(ui_case.app, "selectbox", "Advanced", "AI performance")
    ui_case.app.run()
    ui_case.healthy()
    metrics = {metric.label: metric.value for metric in ui_case.app.metric}
    assert metrics["Model requests"] == "1" and metrics["Input tokens"] == "24"
    assert metrics["Output tokens"] == "4" and metrics["Estimated OpenAI cost"] == "Unavailable"
    assert "Latest Copilot source retrieval" in ui_case.text()
    ui_case.nav("details")
    ui_case.nav("copilot")
    ui_case.app.run()
    assert "60 days" in ui_case.text() and create.call_count == 1

    choose(ui_case.app, "toggle", "Use source retrieval in Copilot", False)
    ui_case.app.run()
    ui_case.app.chat_input[0].set_value("Why is passport validity required?")
    ui_case.app.run()
    assert "Source retrieval is off" in ui_case.text() and create.call_count == 1


def test_missing_openai_configuration_leaves_saved_case_and_local_document_upload_available(ui_case, ready_bundle, monkeypatch):
    monkeypatch.setattr("src.ui.model_settings.cloud_configuration_issue", lambda *a: "missing_key")
    ui_case.seed(entered(ready_bundle[2]))
    ui_case.nav("documents")
    before = ui_case.read()
    _settings(ui_case)
    assert "Set OPENAI_API_KEY on the server" in ui_case.text()
    labels = [button.label for button in ui_case.app.button]
    assert "Upload and analyze" in labels and all("with OpenAI" not in label for label in labels)
    choose(ui_case.app, "toggle", "Case settings", False)
    ui_case.app.run()
    ui_case.nav("copilot")
    assert "OpenAI API" in ui_case.text()
    assert ui_case.read() == before


def test_advanced_renders_saved_evaluation_without_starting_a_new_run(ui_case, monkeypatch):
    results = ui_case.tmp_path / "evaluation" / "results"
    results.mkdir(parents=True)
    payload = {"metadata": {"architecture": "deterministic_router", "provider": "scripted", "model": "scripted"},
               "summary": {"task_success_rate": 1.0, "retrieval_mrr_at_k": 0.75, "citation_accuracy": 1.0}}
    (results / "synthetic-report.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(evaluate_agent, "RESULTS_DIR", results)
    ui_case.nav("technical")
    choose(ui_case.app, "selectbox", "Advanced", "AI performance")
    ui_case.app.run()
    choose(ui_case.app, "selectbox", "Saved benchmark", "synthetic-report")
    ui_case.app.run()
    ui_case.healthy()
    assert "scripted" in ui_case.text()
    assert "| Reciprocal rank | 0.7500 |" in ui_case.text()

def test_insufficient_evidence_reply_does_not_show_a_previous_tool_source(ui_case, ready_bundle):
    from src.core.state import UserEvent

    state = entered(ready_bundle[2])
    state["event"] = UserEvent(kind="assistant_message", text="What does official guidance say about xenon flux?")
    state["assistant_messages"] = [{"role": "assistant",
                                    "content": "I do not have enough retrieved official evidence.", "source_refs": []}]
    state["last_tool_outcome"] = {"status": "SUCCEEDED", "name": "search_official_sources",
                                 "data": {"results": [{"title": "Earlier unrelated source"}]}}
    ui_case.seed(state)
    ui_case.nav("copilot")
    ui_case.healthy()
    assert "enough retrieved official evidence" in ui_case.text()
    assert "Earlier unrelated source" not in ui_case.text()
    assert not any(item.label == "Sources used" for item in ui_case.app.get("expander"))
