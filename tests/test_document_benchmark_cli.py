"""Thin benchmark command: canonical evaluation, private outputs, no live models."""
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from scripts import benchmark_documents as benchmark
from src.document_processing.schemas import UnknownDocument, HealthInsurance


def test_discovery_uses_local_samples_and_excludes_educational_pdf(tmp_path):
    for relative in ("data/local_samples/alternate.pdf","data/synthetic_samples/native.pdf","data/synthetic_samples/example.pdf"):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic")
    assert {path.name for path in benchmark.discover_documents(tmp_path)}=={"alternate.pdf","native.pdf"}


@pytest.mark.parametrize("mode,expected",[("local","local-only"),("auto","auto"),("openai","cloud")])
def test_benchmark_modes_reuse_canonical_evaluator_and_hide_values(tmp_path,monkeypatch,capsys,mode,expected):
    manifest=tmp_path/"manifest.json";output=tmp_path/(mode+".json")
    manifest.write_text(json.dumps([{"path":"PRIVATE-FILE.txt","expected_type":"unknown","criticalfields":{}}]),encoding="utf-8")
    processor=Mock(return_value=UnknownDocument())
    monkeypatch.setattr(benchmark,"process_document",processor)
    benchmark.main(["--mode",mode,"--manifest",str(manifest),"--output",str(output)])
    assert processor.call_args.kwargs["mode"]==expected
    result=json.loads(output.read_text(encoding="utf-8"))
    assert result["case_count"]==1 and result["gold_classification_count"]==1
    assert result["reports"][0]["filename"]=="PRIVATE-FILE.txt"
    assert "PRIVATE-FILE" not in capsys.readouterr().out


def test_benchmark_discovery_does_not_infer_gold_from_filenames(tmp_path,monkeypatch):
    output=tmp_path/"unknown-gold.json"
    monkeypatch.setattr(benchmark,"discover_documents",lambda:[tmp_path/"insurance.pdf"])
    evaluator=Mock(return_value=[{"case_number":1,"classification_correct":None,"predicted_type":"unknown"}])
    monkeypatch.setattr(benchmark,"evaluate_documents",evaluator)
    benchmark.main(["--mode","local","--output",str(output)])
    assert evaluator.call_args.args[0][0]["expected_document_type"] is None
    assert json.loads(output.read_text(encoding="utf-8"))["classification_accuracy"] is None


def test_include_values_affects_only_private_report_not_console(tmp_path,monkeypatch,capsys):
    manifest=tmp_path/"private.json";output=tmp_path/"report.json"
    manifest.write_text(json.dumps([{"document_path":"synthetic.pdf","expected_document_type":"health_insurance_certificate","expected_critical_fields":{}}]),encoding="utf-8")
    monkeypatch.setattr(benchmark,"process_document",lambda *args,**kwargs:HealthInsurance(holder_name="Synthetic Holder"))
    benchmark.main(["--mode","local","--manifest",str(manifest),"--output",str(output),"--include-values"])
    assert "Synthetic Holder" in output.read_text(encoding="utf-8")
    assert "Synthetic Holder" not in capsys.readouterr().out
