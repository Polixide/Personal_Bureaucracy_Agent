"""Tool registry and read tools: valid input, invalid input (error observation), realistic use."""
import json
import shutil
from datetime import date

import pytest
from pydantic import BaseModel

from config import PROCEDURES_DIR
from src.document_processing.schemas import HealthInsurance, SourceEvidence
from src.pack import load_all_packs, load_pack
from src.tools import extract_document
from src.tools.common import ToolContext, ToolError, ToolSpec
from src.tools.registry import build_registry, describe_tools, run_tool, tools_for
from src.twin import service

USER = "synthetic_student_01"
TODAY = date(2026, 11, 18)


def make_ctx(store, procedure_id="ikamet_first", packs=None):
    return ToolContext(store=store, packs=packs or load_all_packs(), user_id=USER, today=TODAY, procedure_id=procedure_id)


@pytest.fixture
def ctx(store):
    return make_ctx(store)


def call(ctx, name, args=None, role="orchestrator", registry=None):
    return run_tool(registry or build_registry(), name, args, ctx, role)


# --------------------------------------------------------------- registry --
class EchoInput(BaseModel):
    n: int


class EchoOutput(BaseModel):
    n: int


def test_invalid_arguments_are_returned_as_an_error_and_the_tool_does_not_run(ctx):
    calls = []
    spec = ToolSpec(name="echo", description="d", input_model=EchoInput, output_model=EchoOutput,
                    fn=lambda args, c: calls.append(args) or EchoOutput(n=args.n), kind="read")
    observation = call(ctx, "echo", {"n": "not a number"}, registry={"echo": spec})
    assert not observation.ok and "Invalid arguments" in observation.error and calls == []
    assert call(ctx, "echo", {"n": 3}, registry={"echo": spec}).output == {"n": 3}


def test_unknown_tool_and_wrong_role_are_refused(ctx):
    assert "Unknown tool" in call(ctx, "send_email").error
    assert "not available to the worker" in call(ctx, "record_fact", {"field": "age", "value": 3}, role="worker").error


def test_crashing_tool_becomes_an_error_observation(ctx):
    def broken(args, c):
        raise KeyError("boom")
    spec = ToolSpec(name="broken", description="d", input_model=EchoInput, output_model=EchoOutput, fn=broken, kind="read")
    observation = call(ctx, "broken", {"n": 1}, registry={"broken": spec})
    assert not observation.ok and "KeyError" in observation.error


def test_worker_tools_are_a_restricted_set():
    worker = tools_for(build_registry(), "worker")
    assert {"read_twin", "evaluate_rules", "extract_document"} <= set(worker)
    assert not {"record_fact", "ask_user", "search_procedures"} & set(worker)


def test_tool_descriptions_are_compact_and_complete():
    described = {item["name"]: item for item in describe_tools(build_registry())}
    assert described["record_fact"]["args"]["field"].startswith("string")
    assert "optional" in described["evaluate_rules"]["args"]["section_id"]
    assert len(json.dumps(list(described.values()))) < 4000


# ------------------------------------------------------- evaluate_rules --
def test_evaluate_rules_for_one_section(ctx):
    service.record_app_value(ctx.store, ctx.pack(), USER, "passport_expiry_date", "2027-07-15")
    service.record_app_value(ctx.store, ctx.pack(), USER, "requested_permit_end_date", "2027-06-01")
    output = call(ctx, "evaluate_rules", {"section_id": "passport_photo"}).output
    assert set(output["failing"]) | set(output["unknown"]) | set(output["passing"]) == {"R006", "R008", "R009"}
    assert list(output["failing"]) == ["R006"] and output["failing"]["R006"]
    assert output["counts"] == {"pass": 0, "fail": 1, "unknown": 2}


def test_evaluate_rules_rejects_unknown_section_and_missing_procedure(store, ctx):
    assert "Unknown section" in call(ctx, "evaluate_rules", {"section_id": "visa"}).error
    assert "No active procedure" in call(make_ctx(store, None), "evaluate_rules", {}).error


def test_unconfirmed_statement_does_not_change_the_verdict(ctx):
    call(ctx, "record_fact", {"field": "is_in_turkiye", "value": "yes"})
    call(ctx, "record_fact", {"field": "entered_with_passport", "value": True})
    assert "R003" in call(ctx, "evaluate_rules", {"section_id": "eligibility"}).output["unknown"]


# -------------------------------------------------------- record_fact --
def test_record_fact_stores_an_unconfirmed_statement(ctx):
    output = call(ctx, "record_fact", {"field": "latest_entry_date", "value": "2026-09-15", "note": "said in chat"}).output
    assert output["confirmed"] is False and output["value"] == "2026-09-15"
    assert ctx.store.get_fact(output["fact_id"]).provenance == "user_stated"


@pytest.mark.parametrize("args,message", [
    ({"field": "favourite_colour", "value": "blue"}, "Unknown field"),
    ({"field": "latest_entry_date", "value": "15/09/2026"}, "YYYY-MM-DD"),
    ({"field": "age"}, "Invalid arguments")])
def test_record_fact_errors(ctx, args, message):
    observation = call(ctx, "record_fact", args)
    assert not observation.ok and message in observation.error


# ---------------------------------------------------------- read_twin --
def test_read_twin_shows_documents_facts_conflicts_and_unconfirmed(ctx):
    pack = ctx.pack()
    service.register_upload(ctx.store, USER, "doc1", "data/uploads/a.pdf", pack.id)
    service.record_app_value(ctx.store, pack, USER, "age", 22)
    service.correct_document_field(ctx.store, pack, USER, "doc1", "holder_name", "Alex Example")
    service.record_statement(ctx.store, pack, USER, "holder_name", "Alexa Sample", confirmed=True, source_ref="app")
    service.record_statement(ctx.store, pack, USER, "is_in_turkiye", True, confirmed=False)
    output = call(ctx, "read_twin").output
    assert output["procedure_id"] == "ikamet_first" and output["documents"][0]["status"] == "pending"
    facts = {row["field"]: row for row in output["facts"]}
    assert facts["age"]["value"] == "22" and facts["current_date"]["provenance"] == "derived"
    assert set(output["conflicts"]["holder_name"]) == {"Alex Example", "Alexa Sample"}
    assert output["unconfirmed"][0]["field"] == "is_in_turkiye"
    assert [row["field"] for row in call(ctx, "read_twin", {"fields": ["age"]}).output["facts"]] == ["age"]


def test_read_twin_without_procedure_still_lists_documents(store):
    service.register_upload(store, USER, "doc1", "x.pdf")
    output = call(make_ctx(store, None), "read_twin").output
    assert output["procedure_id"] is None and len(output["documents"]) == 1 and output["facts"] == []


# --------------------------------------------------- search_procedures --
def test_search_selects_the_only_matching_procedure(store):
    ctx = make_ctx(store, None)
    output = call(ctx, "search_procedures", {"query": "I want to apply for İkamet"}).output
    assert output["selected"] == "ikamet_first" and ctx.procedure_id == "ikamet_first"
    assert store.active_procedure(USER) == "ikamet_first"


def test_search_says_when_nothing_matches(store):
    output = call(make_ctx(store, None), "search_procedures", {"query": "I want to open a bank account"}).output
    assert output["selected"] is None and "No procedure" in output["message"]


def test_search_reports_ambiguity_instead_of_choosing(store, tmp_path):
    shutil.copytree(PROCEDURES_DIR / "ikamet_first", tmp_path / "ikamet_first")
    shutil.copytree(PROCEDURES_DIR / "ikamet_first", tmp_path / "ikamet_copy")
    path = tmp_path / "ikamet_copy" / "procedure.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["id"] = "ikamet_copy"
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx = make_ctx(store, None, packs=load_all_packs(tmp_path))
    output = call(ctx, "search_procedures", {"query": "apply for ikamet"}).output
    assert output["selected"] is None and "Ask the user" in output["message"] and ctx.procedure_id is None


def test_search_rejects_an_empty_query(store):
    assert not call(make_ctx(store, None), "search_procedures", {"query": ""}).ok


# ------------------------------------------------------------ ask_user --
def test_ask_user_records_the_question(ctx):
    output = call(ctx, "ask_user", {"question": "When did you enter Türkiye?", "fields": ["latest_entry_date"]}).output
    assert output["question"].startswith("When") and ctx.question == output["question"]
    assert ctx.store.list_events(USER, kind="question_asked")
    assert not call(ctx, "ask_user", {"question": "Which colour?", "fields": ["colour"]}).ok
    assert not call(ctx, "ask_user", {"question": "?"}).ok


# ------------------------------------------------------ extract_document --
def test_extract_document_tool_stores_facts_in_the_twin(ctx, tmp_path, monkeypatch):
    path = tmp_path / "insurance.txt"
    path.write_text("SYNTHETIC", encoding="utf-8")
    info = HealthInsurance(coverage_end_date="2027-06-30", source_pages={"coverage_end_date": 1},
                           source_snippets={"coverage_end_date": "Coverage end: 30.06.2027"},
                           source_evidence=[SourceEvidence(field="coverage_end_date", original_value="30.06.2027",
                                                           normalized_value="2027-06-30", snippet="Coverage end: 30.06.2027",
                                                           line_number=1, page_number=1)])
    monkeypatch.setattr("src.document_processing.processor.process_document", lambda *a, **k: info)
    service.register_upload(ctx.store, USER, "ins", str(path), "ikamet_first")
    output = call(ctx, "extract_document", {"doc_id": "ins"}).output
    row = output["documents"][0]
    assert row["status"] == "analyzed" and row["facts"][0]["field"] == "insurance_end_date"
    assert ctx.view().values["insurance_end_date"] == date(2027, 6, 30)


def test_extract_document_unknown_id_is_an_error(ctx):
    assert "No uploaded document" in call(ctx, "extract_document", {"doc_id": "nope"}).error


def test_extract_document_failure_marks_the_document_rejected(ctx, tmp_path):
    service.register_upload(ctx.store, USER, "gone", str(tmp_path / "missing.txt"), "ikamet_first")
    output = call(ctx, "extract_document", {"doc_id": "gone"}).output
    assert output["error"] and output["documents"][0]["status"] == "rejected"


def test_extract_document_module_exposes_its_spec():
    assert extract_document.SPEC.kind == "read" and extract_document.SPEC.name == "extract_document"


def test_tool_error_type_is_shared():
    with pytest.raises(ToolError):
        raise ToolError("x")


PACK_RULES = load_pack("ikamet_first").rules


# ------------------------------------------- shapes seen from qwen3:4b live --
def test_empty_section_id_means_all_sections(ctx):
    output = call(ctx, "evaluate_rules", {"section_id": ""}).output
    assert output["section_id"] is None and sum(output["counts"].values()) == len(PACK_RULES)


@pytest.mark.parametrize("word", ["null", "None", " all "])
def test_null_none_and_all_mean_all_sections(ctx, word):
    output = call(ctx, "evaluate_rules", {"section_id": word}).output
    assert output["section_id"] is None and sum(output["counts"].values()) == len(PACK_RULES)


def test_a_single_field_name_is_accepted_as_a_list(ctx):
    service.record_app_value(ctx.store, ctx.pack(), USER, "age", 22)
    assert [row["field"] for row in call(ctx, "read_twin", {"fields": "age"}).output["facts"]] == ["age"]
    assert call(ctx, "read_twin", {"fields": ""}).ok
    assert not call(ctx, "read_twin", {"fields": 5}).ok
