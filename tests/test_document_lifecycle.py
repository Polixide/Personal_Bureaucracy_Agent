"""Offline lifecycle, scoped coverage, MRZ, bundles, reset and orientation."""
from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch,MagicMock
import pytest
import pymupdf
from src.core.graph import build_graph
from src.core.state import UserEvent,new_run,ValidationIssue
from src.memory.checkpointer import sqlite_checkpointer,reset_thread
from src.workers.verification import MILESTONE_SCOPE,milestone_checklist
from src.workers.rules_engine import evaluate_requirements
from src.document_processing.processor import process_document
from test_milestone_documents import SAMPLES,response


def run(graph,config,kind,**kwargs):
    with patch("src.agents.planner.create_action_plan") as planner:
        state=graph.invoke(new_run(UserEvent(kind=kind,**kwargs),"lifecycle",scope=MILESTONE_SCOPE),config)
    planner.assert_not_called();return state


def upload(graph,config,tmp_path,kind,filename=None,text=None):
    path=tmp_path/(filename or kind+".txt");path.write_text(text or SAMPLES[kind],encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":kind}),response({})]
        return run(graph,config,"analyze_document",file_path=str(path))


@pytest.mark.parametrize("requested,program,end,expected",[
    ("2027-06-01",None,"2027-07-01","passed"),
    ("2027-09-01","2027-06-25","2027-07-01","passed"),
    ("2027-09-01","2027-06-25","2027-05-01","failed"),
    (None,None,"2027-07-01","missing_information"),
    ("2027-06-01",None,None,"missing_information"),
])
def test_scoped_R015_end_only_three_valued_rule(requested,program,end,expected):
    state={"planning_scope":MILESTONE_SCOPE,"residence_permit_context":{"insurance_valid_for_ikamet":True,"requested_permit_end_date":requested,"program_end_date":program,"insurance_end_date":end}}
    result=next(r for r in evaluate_requirements(state) if r.rule_id=="R015")
    assert result.status==expected
    assert "insurance_start_date" not in result.missing_fields


def test_same_document_correction_is_authoritative_and_can_be_removed(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"correction"}}
    state=upload(graph,config,tmp_path,"health_insurance_certificate")
    doc=state["documents"][0];raw=deepcopy(doc.info);facts=deepcopy(state["extracted_facts"])
    state=run(graph,config,"confirm_information",document_id=doc.doc_id,confirmed_fields={"coverage_end_date":"2028-07-01"})
    assert state["residence_permit_context"].insurance_end_date==date(2028,7,1)
    assert "insurance_end_date" not in state["verification_conflicts"]
    assert state["documents"][0].info==raw and state["extracted_facts"]==facts
    state=run(graph,config,"remove_correction",document_id=doc.doc_id,confirmed_fields={"coverage_end_date":None})
    assert state["residence_permit_context"].insurance_end_date==date(2027,7,1)
    assert state["verification_records"][doc.doc_id]["coverage_end_date"].origin=="source_grounded"


def test_independent_documents_can_still_conflict(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"independent"}}
    state=upload(graph,config,tmp_path,"student_enrollment_certificate")
    state=upload(graph,config,tmp_path,"health_insurance_certificate",text=SAMPLES["health_insurance_certificate"].replace("Alex Example","Different Person"))
    assert "holder_name" in state["verification_conflicts"] and state["residence_permit_context"].holder_name is None


def test_delete_preserves_independent_provenance_and_clears_stale_plans(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"delete"}}
    state=upload(graph,config,tmp_path,"passport_identity_page")
    passport=state["documents"][0]
    state=upload(graph,config,tmp_path,"ikamet_application_form")
    form=next(d for d in state["documents"] if d.doc_type=="ikamet_application_form")
    state=run(graph,config,"confirm_information",document_id="__declarations__",confirmed_fields={"is_in_turkiye":True})
    graph.update_state(config,{"run_status":"HUMAN_REVIEW_REQUIRED","retry_count":3,"repair_history":[{"attempt":3}],"validation_errors":[ValidationIssue(code="OLD",message="stale",source="gate_1")]})
    state=run(graph,config,"delete_document",document_id=passport.doc_id,confirmed=True)
    assert [d.doc_id for d in state["documents"]]==[form.doc_id]
    assert passport.doc_id not in state["extracted_facts"] and passport.doc_id not in state["verification_records"]
    c=state["residence_permit_context"]
    assert c.passport_expiry_date is None and c.passport_or_foreign_id=="P1234567" and c.is_in_turkiye is True
    assert state["candidate_plan"] is state["approved_plan"] is None
    assert not state["validation_errors"] and state["retry_count"]==0 and state["repair_history"]==[]
    assert state["run_status"]!="HUMAN_REVIEW_REQUIRED" and len(state["relevant_requirements"])==24


def test_delete_requires_explicit_confirmation(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"confirmed-delete"}}
    state=upload(graph,config,tmp_path,"card_fee_receipt");doc=state["documents"][0]
    state=run(graph,config,"delete_document",document_id=doc.doc_id)
    assert state["run_status"]=="INVALID_INPUT" and state["documents"][0].doc_id==doc.doc_id


def test_duplicate_is_pending_until_explicit_replace_or_cancel(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"replace"}}
    state=upload(graph,config,tmp_path,"health_insurance_certificate",filename="old.txt")
    old=state["documents"][0]
    state=upload(graph,config,tmp_path,"health_insurance_certificate",filename="new.txt",text=SAMPLES["health_insurance_certificate"].replace("2027-07-01","2028-07-01"))
    pending=next(d for d in state["documents"] if d.status=="replacement_pending")
    assert state["residence_permit_context"].insurance_end_date==date(2027,7,1)
    state=run(graph,config,"replace_document",document_id=pending.doc_id,confirmed=True)
    assert len(state["documents"])==1 and state["documents"][0].doc_id==pending.doc_id
    assert state["residence_permit_context"].insurance_end_date==date(2028,7,1)
    assert old.doc_id not in state["verification_records"] and old.doc_id not in state["extracted_facts"]
    state=upload(graph,config,tmp_path,"health_insurance_certificate",filename="cancel.txt")
    cancelled=next(d for d in state["documents"] if d.status=="replacement_pending")
    state=run(graph,config,"cancel_document",document_id=cancelled.doc_id,confirmed=True)
    assert len(state["documents"])==1 and state["residence_permit_context"].insurance_end_date==date(2028,7,1)


def test_declaration_heading_wins_over_application_reference(tmp_path):
    path=tmp_path/"declaration.txt";path.write_text("BEYAN/KONTROL BELGESI\nSabanci University\nResidence Permit Application Form checklist\nSigned: yes",encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"ikamet_application_form"}),response({})]
        info=process_document(path)
    assert info.document_type=="declaration_control_document" and info.document_present is info.signed is True


def test_tax_office_receipt_identifies_relevant_card_payment(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"tax-receipt"}}
    text="T.C. Hazine ve Maliye Bakanligi\nTAHSILAT ALINDISI\nIkamet izin belge bedeli\nTahsilat tarihi: 14.10.2026\nTutar: 964.00 TL\nAlindi no: RECEIPT-123\nVergi no: 99887766"
    state=upload(graph,config,tmp_path,"card_fee_receipt",text=text)
    info=state["documents"][0].info
    assert info.receipt_present is True and info.reference_number=="RECEIPT-123"
    assert state["residence_permit_context"].fee_receipt_present is True


MRZ="P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_readable_mrz_has_check_digit_validated_fields_and_provenance(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"mrz"}}
    state=upload(graph,config,tmp_path,"passport_identity_page",text=MRZ)
    info=state["documents"][0].info
    assert info.passport_number=="L898902C3" and info.nationality=="UTO" and info.expiry_date==date(2012,4,15)
    assert info.holder_name=="ANNA MARIA ERIKSSON" and info.source_pages["passport_number"]==1
    assert "L898902C3" in info.source_snippets["passport_number"]
    assert state["residence_permit_context"].entered_with_passport is None


def test_passport_pdf_bundle_creates_two_logical_documents_with_original_pages(tmp_path):
    path=tmp_path/"bundle.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50,50),MRZ,fontsize=8)
        pdf.new_page().insert_text((50,50),"TURKIYE ENTRY STAMP\nEntry date: 2026-09-15",fontsize=10)
        pdf.save(path)
    graph=build_graph();config={"configurable":{"thread_id":"bundle"}}
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.side_effect=[response({"document_type":"passport_identity_page"})]+[response({})]*5
        state=run(graph,config,"analyze_document",file_path=str(path))
    assert {d.doc_type for d in state["documents"]}=={"passport_copy","turkiye_entry_stamp"}
    identity=next(d for d in state["documents"] if d.doc_type=="passport_copy")
    entry=next(d for d in state["documents"] if d.doc_type=="turkiye_entry_stamp")
    assert identity.info.source_pages["passport_number"]==1 and entry.info.source_pages["latest_entry_date"]==2
    assert identity.file_path==entry.file_path and identity.doc_id!=entry.doc_id
    assert state["residence_permit_context"].latest_entry_date is None


def test_low_confidence_scanned_page_tries_rotations_and_retains_page_number(tmp_path):
    from src.document_processing.extractor import extract_pages
    path=tmp_path/"rotated.pdf";path.write_bytes(b"synthetic")
    page=MagicMock();page.rect=pymupdf.Rect(0,0,595,842);page.get_text.return_value="";page.get_images.return_value=[True]
    page.get_pixmap.return_value.tobytes.return_value=b"image"
    document=MagicMock();document.needs_pass=False;document.__iter__.return_value=iter([page])
    results=[SimpleNamespace(txts=["x"],scores=[0.1]),SimpleNamespace(txts=["CARD FEE PAYMENT RECEIPT"],scores=[0.99]),SimpleNamespace(txts=[],scores=[]),SimpleNamespace(txts=[],scores=[])]
    with patch("src.document_processing.extractor.pymupdf.open") as opened,patch("src.document_processing.extractor.get_ocr_engine") as engine:
        opened.return_value.__enter__.return_value=document
        engine.return_value.side_effect=results
        pages=extract_pages(path)
    assert pages[0].number==1 and pages[0].text=="CARD FEE PAYMENT RECEIPT"
    assert page.get_pixmap.call_count==4 and "matrix" in page.get_pixmap.call_args_list[1].kwargs


def test_manual_reclassification_uses_same_grounded_pipeline(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"reclass"}}
    path=tmp_path/"reclass.txt";path.write_text(SAMPLES["card_fee_receipt"],encoding="utf-8")
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value=response({"document_type":"unknown"})
        # Force an uncertain first response rather than relying on model output.
        with patch("src.document_processing.processor.checked_classification",return_value="unknown"):
            state=run(graph,config,"analyze_document",file_path=str(path))
    doc=state["documents"][0]
    with patch("src.document_processing.processor.Client") as client:
        client.return_value.chat.return_value=response({})
        state=run(graph,config,"reprocess_document",document_id=doc.doc_id,document_type_hint="card_fee_receipt")
        assert client.return_value.chat.call_count==1
    assert state["documents"][0].doc_type=="card_fee_receipt" and state["residence_permit_context"].fee_receipt_present is True


def test_reset_deletes_only_current_thread_checkpoints(tmp_path):
    db=tmp_path/"reset.db"
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        for tid in ("current","other"):
            run(graph,{"configurable":{"thread_id":tid}},"confirm_information",document_id="__declarations__",confirmed_fields={"is_in_turkiye":True})
    with pytest.raises(ValueError):reset_thread("current","wrong",db)
    reset_thread("current","current",db)
    with sqlite_checkpointer(db) as saver:
        graph=build_graph(saver)
        assert not graph.get_state({"configurable":{"thread_id":"current"}}).values
        assert graph.get_state({"configurable":{"thread_id":"other"}}).values["residence_permit_context"].is_in_turkiye is True



def test_missing_insurance_start_does_not_keep_a_passing_policy_in_the_plan(tmp_path):
    from src.agents.planner import actionable_needs
    graph=build_graph();config={"configurable":{"thread_id":"insurance-plan"}}
    state=upload(graph,config,tmp_path,"student_enrollment_certificate")
    text=SAMPLES["health_insurance_certificate"].replace("Coverage start: 2026-09-01\n","")
    state=upload(graph,config,tmp_path,"health_insurance_certificate",text=text)
    doc=next(d for d in state["documents"] if d.doc_type=="health_insurance")
    state=run(graph,config,"confirm_information",document_id=doc.doc_id,confirmed_fields={"insurance_valid_for_ikamet":True,"insurance_evidence_type":"private_health_insurance"})
    assert state["residence_permit_context"].insurance_start_date is None
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R015").passed is True
    assert next(row for row in milestone_checklist(state) if row["document_type"]=="health_insurance")["status"]=="complete"
    assert not any(n["rule_id"] in {"R015","R016"} for n in actionable_needs(state))


def test_all_missing_supported_documents_are_visible_independently_of_action_limit(tmp_path):
    from src.agents.planner import actionable_needs
    from src.agents.planning_policy import selected_needs
    graph=build_graph();config={"configurable":{"thread_id":"all-missing"}}
    state=run(graph,config,"confirm_information",document_id="__declarations__",confirmed_fields={"is_in_turkiye":True})
    needs=actionable_needs(state)
    assert len({n.get("document_type") for n in needs if n.get("document_type")})==7
    assert len(milestone_checklist(state))==7 and all(row["status"]=="missing" for row in milestone_checklist(state))
    assert len(selected_needs(needs,limit=3))==3


def test_saved_duplicate_versions_can_be_explicitly_canonicalized(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"legacy-duplicates"}}
    state=upload(graph,config,tmp_path,"student_enrollment_certificate")
    original=state["documents"][0]
    other=original.model_copy(update={"doc_id":"old-duplicate-version"})
    graph.update_state(config,{"documents":[other],"extracted_facts":{other.doc_id:other.info.model_dump(mode="json")}})
    state=run(graph,config,"reevaluate")
    assert len(state["documents"])==2
    state=run(graph,config,"replace_document",document_id=original.doc_id,confirmed=True)
    assert len(state["documents"])==1 and state["documents"][0].doc_id==original.doc_id
    assert other.doc_id not in state["verification_records"]
    assert "holder_name" not in state["verification_conflicts"]


def test_replacement_and_deletion_reassess_coverage(tmp_path):
    graph=build_graph();config={"configurable":{"thread_id":"reassess"}}
    upload(graph,config,tmp_path,"student_enrollment_certificate")
    state=upload(graph,config,tmp_path,"health_insurance_certificate",filename="current.txt")
    old=next(d for d in state["documents"] if d.doc_type=="health_insurance")
    state=run(graph,config,"confirm_information",document_id=old.doc_id,confirmed_fields={"insurance_valid_for_ikamet":True})
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R015").passed is True
    state=upload(graph,config,tmp_path,"health_insurance_certificate",filename="short.txt",text=SAMPLES["health_insurance_certificate"].replace("2027-07-01","2027-01-01"))
    new=next(d for d in state["documents"] if d.status=="replacement_pending")
    state=run(graph,config,"replace_document",document_id=new.doc_id,confirmed=True)
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R015").status=="failed"
    state=run(graph,config,"delete_document",document_id=new.doc_id,confirmed=True)
    assert next(r for r in state["relevant_requirements"] if r.rule_id=="R015").status=="missing_information"
