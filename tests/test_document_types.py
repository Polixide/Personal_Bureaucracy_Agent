from pathlib import Path
from unittest.mock import patch
from src.core.state import Document, UserEvent, new_run, upsert_documents, merge_facts
from src.document_processing.schemas import HealthInsurance
from src.workers.document_analyzer import document_analyzer
from src.guardrails.gate_0 import gate_0
from config import PROJECT_ROOT


def test_document_adapter_retains_quality_without_legal_claims():
    doc=Document(doc_id="test",file_path="synthetic.txt")
    info=HealthInsurance(coverage_start_date="2026-02-30",ambiguous_fields=["policy_number"])
    with patch("src.workers.document_analyzer.process_document",return_value=info): result=document_analyzer({"documents":[doc]})
    assert result["documents"][0].doc_type == "health_insurance"
    assert result["documents"][0].info.validation_issues
    assert result["extracted_facts"]["test"]["coverage_start_date"] is None
    assert "residence_permit_context" not in result

def test_processing_error_rejects_without_loop():
    doc=Document(doc_id="test",file_path="bad.pdf")
    with patch("src.workers.document_analyzer.process_document",side_effect=ValueError("corrupt")):
        result=document_analyzer({"documents":[doc]})
    assert result["documents"][0].status == "rejected"

def test_reducers_preserve_other_documents_and_facts():
    a=Document(doc_id="a",file_path="a.txt"); b=Document(doc_id="b",file_path="b.txt")
    assert len(upsert_documents([a,b],[a.model_copy(update={"status":"analyzed"})])) == 2
    assert merge_facts({"a":1},{"b":2}) == {"a":1,"b":2}

def test_gate0_real_file_and_extension():
    good=new_run(UserEvent(kind="upload",file_path=str(PROJECT_ROOT / "data/synthetic_samples/student_enrollment.txt")),"test")
    assert gate_0(good)["documents"][0].status == "pending"
    for path in ["missing.pdf","photo.jpg","../outside.txt"]:
        bad=new_run(UserEvent(kind="upload",file_path=path),"test")
        assert gate_0(bad)["validation_errors"][0].code == "INVALID_INPUT"

def test_scope_rejection():
    state=new_run(UserEvent(kind="message",text="help"),"test")
    state["residence_permit_context"]={"application_kind":"extension"}
    assert gate_0(state)["validation_errors"]


import pytest
from pydantic import ValidationError

@pytest.mark.parametrize("label,expected",[("passport","passport_copy"),("Passport","passport_copy"),("Health insurance","health_insurance"),("valid_health_insurance","health_insurance"),("health_insurance_certificate","health_insurance"),("student_enrollment_certificate","student_certificate")])
def test_legacy_labels_normalize(label,expected):
    assert Document(doc_id="a",file_path="a.txt",doc_type=label).doc_type == expected

def test_invented_type_rejected():
    with pytest.raises(ValidationError): Document(doc_id="a",file_path="a.txt",doc_type="invented")


def test_deadlines_are_validated_at_input_boundary():
    state=new_run(UserEvent(kind="message",text="help"),"test")
    state["deadlines"]={"appointment":"tomorrow"}
    assert gate_0(state)["validation_errors"]
    state["deadlines"]={"appointment":"2026-11-15"}
    assert gate_0(state)["deadlines"]["appointment"].isoformat() == "2026-11-15"
