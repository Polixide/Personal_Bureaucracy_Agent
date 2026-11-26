"""Synthetic provider, retrieval and evaluation integration regressions."""
from dataclasses import asdict
from copy import deepcopy
import csv
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from scripts import evaluate_agent
from src.agents.assistant import assistant_turn
from src.core.graph import build_graph
from src.core.llm_diagnostics import collect_model_calls
from src.core.model_provider import AIOptions, OpenAIRequestError, RequestBudget, cloud_configuration_issue, openai_response, use_ai_options
from src.core.state import ProcedureState, ResidencePermitContext, UserEvent
from src.evaluation.metrics import summarize_calls
from src.evaluation.router import _accepted_input, _mock_replies
from src.rag.ingestion import ingest_guidance
from src.rag.retriever import search_official_sources

from conftest import wire_planner_response


def _client(reply):
    return SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=reply)), close=Mock())


def _response(payload, *, usage=None, status="completed"):
    return SimpleNamespace(output_text=json.dumps(payload), status=status, usage=usage)


def _query(client=None):
    return openai_response(model="synthetic-structured-model", task="synthetic_task",
        messages=[{"role": "user", "content": "synthetic question"}], schema={
            "type": "object", "properties": {"answer": {"type": "string", "default": ""}}
        }, max_output_tokens=120, client=client)


def test_openai_usage_configuration_and_request_budget(monkeypatch):
    monkeypatch.setenv("OPENAI_PRICING_JSON", json.dumps({"synthetic-structured-model": {
        "as_of": "2026-11-26", "input_per_million": 1.5, "output_per_million": 6.0,
        "cached_input_per_million": 0.5
    }}))
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120,
                            input_tokens_details=SimpleNamespace(cached_tokens=40))
    client = _client(lambda **kwargs: _response({"answer": "synthetic response"}, usage=usage))
    budget = RequestBudget(1)
    with use_ai_options(AIOptions(provider="openai", budget=budget)), collect_model_calls() as calls:
        assert json.loads(_query(client)) == {"answer": "synthetic response"}
        with pytest.raises(OpenAIRequestError, match="request limit"):
            _query(client)
    assert budget.used == client.responses.create.call_count == len(calls) == 1
    arguments = client.responses.create.call_args.kwargs
    assert arguments["store"] is False and arguments["max_output_tokens"] == 120
    wire = arguments["text"]["format"]
    assert wire["strict"] and wire["schema"]["required"] == ["answer"]
    assert wire["schema"]["additionalProperties"] is False
    assert "default" not in wire["schema"]["properties"]["answer"]
    result = summarize_calls([asdict(row) for row in calls])
    assert result["provider_calls"] == {"openai": 1}
    assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (100, 20, 120)
    assert result["estimated_cost_usd"] == pytest.approx((60 * 1.5 + 40 * 0.5 + 20 * 6) / 1_000_000)
    assert result["transport_retries"] == 0 and calls[0].pricing["as_of"] == "2026-11-26"


class APIStatusError(RuntimeError):
    def __init__(self, status):
        self.status_code = status
        super().__init__("SYNTHETIC PRIVATE SDK BODY")


@pytest.mark.parametrize(("failure", "code"), [
    (APIStatusError(401), "authentication"), (APIStatusError(429), "rate_limit"),
    (httpx.ReadTimeout("SYNTHETIC PRIVATE SDK BODY"), "timeout"),
    (httpx.ConnectError("SYNTHETIC PRIVATE SDK BODY"), "connection"),
    (APIStatusError(500), "response"),
])
def test_provider_errors_count_actual_attempts_and_keep_sdk_body_private(failure, code, capsys):
    client = _client(failure)
    with collect_model_calls() as calls, pytest.raises(OpenAIRequestError) as raised:
        _query(client)
    assert raised.value.code == code
    assert len(calls) == 1 and calls[0].error_code == code
    assert not calls[0].request_success and calls[0].timed_out == (code == "timeout")
    assert calls[0].total_tokens is None and calls[0].estimated_cost_usd is None
    assert "SYNTHETIC PRIVATE SDK BODY" not in str(raised.value) + capsys.readouterr().err


def test_incomplete_response_retains_usage_without_marking_schema_valid():
    client = _client(lambda **kwargs: _response({}, status="incomplete",
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8)))
    with collect_model_calls() as calls, pytest.raises(OpenAIRequestError) as error:
        _query(client)
    assert error.value.code == "invalid_output"
    assert calls[0].structured_output_valid is False and calls[0].total_tokens == 8


def test_cloud_readiness_requires_key_and_sdk_without_sending_requests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cloud_configuration_issue("synthetic-model") == "missing_key"
    assert cloud_configuration_issue("") == "missing_model"
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-only")
    monkeypatch.setattr("src.core.model_provider.find_spec", lambda name: None)
    assert cloud_configuration_issue("synthetic-model") == "missing_sdk"


@pytest.mark.parametrize("scenario", ["F-repair", "F-stop"])
def test_openai_planner_uses_the_same_gate_and_keeps_private_repair_data_local(monkeypatch, scenario):
    suite = evaluate_agent.load_suite()
    spec = next(row for row in suite["scenarios"] if row["id"] == scenario)
    drafts = iter(_mock_replies(suite, spec))

    def reply(**kwargs):
        packet = json.loads(kwargs["input"][1]["content"])
        assert "relevant_context" not in packet and "unresolved_conflicts" not in packet
        assert all("tool_call" not in row for row in packet["canonical_candidates"])
        assert "SYNTHETIC IDENTITY" not in json.dumps(kwargs)
        raw = next(drafts)
        wired = wire_planner_response(raw, packet).message.content
        return _response(json.loads(wired))

    client = _client(reply)
    monkeypatch.setattr("src.agents.planner.OPENAI_MODEL", "synthetic-structured-model")
    monkeypatch.setattr("src.core.model_provider.create_openai_client", lambda timeout: client)
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "isolated-openai-planner"}}
    state = _accepted_input(suite, spec, UserEvent(kind="generate_plan"))
    state["residence_permit_context"].holder_name = "SYNTHETIC IDENTITY"
    with use_ai_options(AIOptions(provider="openai", budget=RequestBudget(2))), collect_model_calls() as calls:
        result = graph.invoke(state, cfg)
    if scenario == "F-repair":
        assert result["run_status"] == "VALIDATED"
        assert result["plan_origin"] == "openai" and "human_approval" in graph.get_state(cfg).next
        ids = [identifier for action in result["candidate_plan"] for identifier in action.grounding_ids]
        assert len(ids) == len(set(ids)) == 3
    else:
        assert result["run_status"] == "HUMAN_REVIEW_REQUIRED"
        assert result["validation_errors"][0].code == "UNCHANGED_INVALID_PLAN"
    assert [call.task for call in calls] == ["planner", "reflection_replan"]
    assert all(call.provider == "openai" for call in calls)
    repair = json.loads(client.responses.create.call_args.kwargs["input"][1]["content"])["repair"]
    assert repair["affected_candidate_ids"] and repair["previous_model_plan"]["actions"]
    assert set(repair["previous_model_plan"]["actions"][0]) == {"candidate_ids"}


def test_cloud_guidance_receives_only_current_question_and_public_passages(monkeypatch):
    monkeypatch.setattr("src.agents.assistant.OPENAI_MODEL", "synthetic-structured-model")
    client = _client(lambda **kwargs: _response({"source_indices": [0]}))
    monkeypatch.setattr("src.core.model_provider.create_openai_client", lambda timeout: client)
    state = {"procedure": ProcedureState(), "residence_permit_context": ResidencePermitContext(holder_name="SYNTHETIC-CASE-A"),
             "assistant_messages": [{"role": "user", "content": "SYNTHETIC PREVIOUS QUESTION"}]}
    before = deepcopy(state)
    question = "Why is passport validity required?"
    with use_ai_options(AIOptions(provider="openai", embedding_model="")), collect_model_calls() as calls:
        result = assistant_turn(state, question, copilot=True)
    wire = json.loads(client.responses.create.call_args.kwargs["input"][1]["content"])
    assert set(wire) == {"question", "passages"} and wire["question"] == question
    assert "SYNTHETIC" not in json.dumps(wire)
    answer = result["assistant_messages"][-1]
    assert "60 days" in answer["content"]
    assert len(answer["source_refs"]) == 1 and answer["source_refs"][0]["source_id"] == "GOV_FAQ"
    assert answer["source_refs"][0]["url"] and answer["source_refs"][0]["source_version"]
    assert len(calls) == 1 and calls[0].structured_output_valid
    assert result["assistant_diagnostics"]["rag"]["queries"][0]["mode"] == "lexical"
    assert before == state


@pytest.mark.parametrize(("enabled", "broken", "expected"), [
    (False, False, "Source retrieval is off"),
    (True, False, "enough retrieved official evidence"),
    (True, True, "Source search is unavailable"),
])
def test_no_cloud_request_for_disabled_empty_or_failed_retrieval(monkeypatch, enabled, broken, expected):
    if broken:
        def fail(*args, **kwargs):
            raise OSError("Synthetic unreadable public corpus")
        monkeypatch.setattr("src.rag.retriever.search_official_sources", fail)
    with use_ai_options(AIOptions(provider="openai", rag_enabled=enabled, embedding_model="")), collect_model_calls() as calls:
        updated = assistant_turn({"procedure": ProcedureState()}, "What does official guidance say about quantum xenon flux?", copilot=True)
    answer = updated["assistant_messages"][-1]
    assert expected in answer["content"] and not answer["source_refs"]
    assert not calls
    assert updated["assistant_diagnostics"]["rag"]["used"] == enabled


def test_invalid_cloud_citation_uses_real_retrieved_passages(monkeypatch):
    monkeypatch.setattr("src.agents.assistant.OPENAI_MODEL", "synthetic-structured-model")
    client = _client(lambda **kwargs: _response({"source_indices": [999]}))
    monkeypatch.setattr("src.core.model_provider.create_openai_client", lambda timeout: client)
    with use_ai_options(AIOptions(provider="openai", embedding_model="")), collect_model_calls() as calls:
        result = assistant_turn({"procedure": ProcedureState()}, "Why is passport validity required?", copilot=True)
    reply = result["assistant_messages"][-1]
    assert "60 days" in reply["content"] and all(row["url"].startswith("https://") for row in reply["source_refs"])
    assert "999" not in reply["content"] and calls[0].fallback_used
    assert not calls[0].structured_output_valid


def test_semantic_index_reuses_vectors_and_updates_changed_source_bytes(tmp_path, monkeypatch):
    from src.rag import retriever, semantic

    corpus = tmp_path / "public-snapshot.json"
    monkeypatch.setattr(retriever, "GUIDANCE_CACHE", corpus)
    monkeypatch.setattr(semantic, "INDEX_DIR", tmp_path / "index")
    embed = Mock(side_effect=lambda *, model, input, truncate: SimpleNamespace(
        embeddings=[[1.0, float(index)] for index, _ in enumerate(input)], prompt_eval_count=9))
    monkeypatch.setattr("ollama.Client.embed", embed)
    record = {"source_id": "GOV_FAQ", "content": "Synthetic xenon evidence alpha.",
              "verified_at": "2026-11-26", "page": 2, "section": "Synthetic section", "source_version": "v1"}
    ingest_guidance([record], corpus)
    initial = corpus.stat()
    first = search_official_sources("xenon", embedding_model="synthetic-embed")
    assert len(first) == 1 and first[0].page == 2 and first[0].section == record["section"]
    assert first[0].procedure == "ikamet_first"
    index = next((tmp_path / "index").glob("*.json"))
    assert first[0].chunk_id in json.loads(index.read_text(encoding="utf-8"))["vectors"]
    assert search_official_sources("xenon", embedding_model="synthetic-embed") == first
    assert embed.call_count == 3  # Two queries and one chunk batch.
    ingest_guidance([{**record, "content": "Synthetic xenon evidence bravo."}], corpus)
    os.utime(corpus, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    new = search_official_sources("xenon", embedding_model="synthetic-embed")
    assert new[0].chunk_id != first[0].chunk_id and "bravo" in new[0].content
    persisted = json.loads(index.read_text(encoding="utf-8"))["vectors"]
    assert first[0].chunk_id not in persisted and new[0].chunk_id in persisted
    assert embed.call_count == 5
    index.write_text("corrupt synthetic index", encoding="utf-8")
    assert search_official_sources("xenon", embedding_model="synthetic-embed") == new
    assert embed.call_count == 7


def test_optional_embedding_failure_keeps_local_lexical_evidence(monkeypatch):
    monkeypatch.setattr("ollama.Client.embed", Mock(side_effect=ConnectionError("local embedder offline")))
    diagnostics = {}
    results = search_official_sources("passport validity", embedding_model="synthetic-embed", details=diagnostics)
    assert any(row.source_id == "GOV_FAQ" for row in results)
    assert diagnostics["mode"] == "lexical" and diagnostics["embedding_unavailable"]


def test_document_cloud_environment_needs_an_explicit_document_choice(monkeypatch):
    from src.document_processing import processor

    monkeypatch.setattr(processor, "DOCUMENT_AI_PROVIDER", "cloud")
    monkeypatch.setattr(processor, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(processor, "DOCUMENT_FALLBACK_PROVIDER", "openai")
    monkeypatch.setattr(processor, "cloud_document_configured", lambda: True)
    assert processor._provider_settings(None, None, None) == ("ollama", "none", "local-only")
    assert processor._provider_settings("cloud", None, None) == ("openai", "none", "cloud")


def test_private_document_cache_is_scoped_to_profile_and_cloud_is_uncached(tmp_path, monkeypatch):
    import config
    from src.document_processing import processor
    from src.document_processing.schemas import UnknownDocument

    file_path = tmp_path / "synthetic.txt"
    file_path.write_text("private synthetic sample", encoding="utf-8")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DOCUMENT_CACHE_ENABLED", True)
    fake = Mock(side_effect=lambda *args, **kwargs: UnknownDocument())
    monkeypatch.setattr(processor, "process_document", fake)
    assert not processor.cached_process_document(file_path, cache_scope="profile-one").extraction_diagnostics["cache_hit"]
    assert processor.cached_process_document(file_path, cache_scope="profile-one").extraction_diagnostics["cache_hit"]
    assert not processor.cached_process_document(file_path, cache_scope="profile-two").extraction_diagnostics["cache_hit"]
    for _ in range(2):
        processor.cached_process_document(file_path, cache_scope="profile-one", mode="cloud")
    assert fake.call_count == 4


def test_rag_benchmark_exports_labeled_quality_without_prompts_or_credentials(tmp_path, monkeypatch):
    results = tmp_path / "evaluation" / "results"
    monkeypatch.setattr(evaluate_agent, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_agent, "RESULTS_DIR", results)
    monkeypatch.setenv("OPENAI_API_KEY", "SYNTHETIC-KEY-MUST-NOT-BE-EXPORTED")
    assert evaluate_agent.main(["--mode", "rag-offline", "--label", "check"]) == 0
    content = (results / "check.json").read_text(encoding="utf-8")
    assert "SYNTHETIC-KEY-MUST-NOT-BE-EXPORTED" not in content and "quantum xenon flux" not in content
    data = json.loads(content)
    runs = {row["scenario_id"]: row for row in data["runs"]}
    supported, unsupported = runs["RAG-passport"], runs["RAG-unsupported"]
    assert runs["RAG-dormitory"]["answer_correct"] and runs["RAG-dormitory"]["indexed_chunks"] > 0
    assert supported["citation_accuracy"] == 1.0 and supported["retrieval_mrr_at_k"] == 1.0
    assert supported["rag_used"] and supported["provider"] == "scripted"
    assert unsupported["answer_correct"] and unsupported["retrieval_precision_at_k"] is None
    assert unsupported["estimated_cost_usd"] is None and unsupported["llm_calls"] == 0
    with (results / "check.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["retrieval_recall_at_k"] and rows[1]["estimated_cost_usd"] == ""
    assert {"provider", "reflection_calls", "answer_correct"} <= rows[0].keys()


def test_benchmark_without_explicit_cloud_budget_cannot_dispatch(capsys):
    with pytest.raises(SystemExit) as raised:
        evaluate_agent.main(["--mode", "rag-live", "--provider", "openai", "--allow-openai"])
    assert raised.value.code == 2
    assert "--max-openai-requests" in capsys.readouterr().err

def test_document_consent_binds_to_the_selected_file_and_event(monkeypatch):
    from src.core.state import Document
    from src.document_processing.schemas import UnknownDocument
    from src.workers import document_analyzer as worker

    analyze = Mock(side_effect=lambda *args, **kwargs: UnknownDocument())
    monkeypatch.setattr(worker, "process_document", analyze)
    monkeypatch.setattr(worker, "_cached_analyze", analyze)
    documents = [Document(doc_id="one", file_path="synthetic-one.txt"),
                 Document(doc_id="two", file_path="synthetic-two.txt")]
    event = UserEvent(kind="analyze_document", file_path=documents[0].file_path)
    state = {"event": event, "user_id": "synthetic-profile", "documents": documents}
    options = AIOptions(provider="openai", cloud_document=(event.kind, documents[0].file_path))
    with use_ai_options(options):
        worker.document_analyzer(state)
    assert analyze.call_args_list[0].kwargs == {"cache_scope": "synthetic-profile", "mode": "cloud", "prefer_native": True, "page_sink": []}
    assert analyze.call_args_list[1].kwargs == {"cache_scope": "synthetic-profile"}
    analyze.reset_mock()
    with use_ai_options(AIOptions(provider="openai")):
        worker.document_analyzer(state)
    assert all("mode" not in call.kwargs for call in analyze.call_args_list)


def test_case_document_reads_are_isolated_from_other_profiles_and_public_search():
    from src.core.state import Document

    document = Document(doc_id="private-xenon-marker", file_path="synthetic-A.txt", status="analyzed",
                        doc_type="passport_copy")
    profile_a = {"procedure": ProcedureState(), "documents": [document], "user_id": "synthetic-A"}
    profile_b = {"procedure": ProcedureState(), "documents": [], "user_id": "synthetic-B"}
    question = "Which documents have I uploaded?"
    own = assistant_turn(profile_a, question, model=False, copilot=True)
    other = assistant_turn(profile_b, question, model=False, copilot=True)
    assert own["assistant_events"][-1]["data"] != other["assistant_events"][-1]["data"]
    assert "private-xenon-marker" not in json.dumps(other, default=str)
    assert not search_official_sources("private-xenon-marker", embedding_model="")

def test_labeled_answer_requires_a_real_relevant_citation_for_the_expected_fact():
    from src.evaluation.rag_eval import labeled_guidance_metrics

    retrieved = [{"chunk_id": "actual", "source_id": "OTHER", "url": "https://example.invalid/other",
                  "content": "A different synthetic procedure."}]
    correct_ref = {"chunk_id": "actual", "source_id": "OTHER", "url": "https://example.invalid/other",
                   "excerpt": retrieved[0]["content"]}
    fabricated = {"chunk_id": "fake", "source_id": "EXPECTED", "url": "https://example.invalid/fake",
                  "excerpt": "60 days of synthetic coverage."}
    content = correct_ref["excerpt"] + " " + fabricated["excerpt"]
    result = labeled_guidance_metrics({"content": content, "source_refs": [correct_ref, fabricated]}, retrieved,
                                      {"relevant_source_ids": ["EXPECTED"], "required_phrases": ["60 days"]})
    assert result["citation_accuracy"] == 0.5 and result["unsupported_claim_rate"] == 0.5
    assert result["answer_correct"] is False
