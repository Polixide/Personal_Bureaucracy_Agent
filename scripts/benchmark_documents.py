"""Private local document benchmark; canonical processor/evaluator, safe summaries.

python scripts/benchmark_documents.py --mode local

Legacy provider modes are retained for compatibility and are unsupported by the
normal local-only installation.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import PROJECT_ROOT
from src.document_processing.processor import process_document, evaluate_documents

MODE_MAP = {"local":"local-only", "auto":"auto", "openai":"cloud"}


def discover_documents(project_root=PROJECT_ROOT):
    """Discover local administrative samples without deriving expected labels."""
    found=[]
    for relative in ("data/local_samples", "data/synthetic_samples"):
        directory=Path(project_root)/relative
        if directory.is_dir():
            found.extend(path.resolve() for path in directory.rglob("*.pdf") if path.name.casefold()!="example.pdf")
    return sorted(set(found),key=lambda path:str(path).casefold())


def _cases(manifest):
    if manifest is None:
        return [{"document_path":str(path),"expected_document_type":None,"expected_critical_fields":{}} for path in discover_documents()],PROJECT_ROOT
    payload=json.loads(manifest.read_text(encoding="utf-8-sig"))
    rows=payload.get("cases") if isinstance(payload,dict) else payload
    if not isinstance(rows,list):raise ValueError("The private manifest must contain a case list")
    cases=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError("Each manifest case must be an object")
        path=row.get("document_path",row.get("path"))
        if not isinstance(path,str) or not path.strip():raise ValueError("Each case needs document_path")
        cases.append({"document_path":path,
            "expected_document_type":row.get("expected_document_type",row.get("expected_type")),
            "expected_critical_fields":row.get("expected_critical_fields",row.get("critical_fields",row.get("criticalfields",row.get("expected_fields",{}))))})
    return cases,manifest.parent


def _private_output(path):
    resolved=path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        safe=[PROJECT_ROOT/".local",PROJECT_ROOT/".test-temp",PROJECT_ROOT/"data/local_samples"]
        if not any(resolved.is_relative_to(directory) for directory in safe):
            raise ValueError("Inside the project, benchmark reports must stay in ignored .local/, .test-temp/ or data/local_samples/")
    return resolved


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=list(MODE_MAP),default="local",
                        help="Selecting auto or openai authorizes sending selected document text/pages to OpenAI.")
    parser.add_argument("--manifest",type=Path,help="Private JSON case list; optional gold values remain outside version control")
    parser.add_argument("--output",type=Path,help="Private report path; defaults to ignored .local/document_benchmark_<mode>.json")
    parser.add_argument("--include-values",action="store_true",help="Store structured private field values only in the private report, never console")
    args=parser.parse_args(argv)
    try:
        cases,base_dir=_cases(args.manifest.resolve() if args.manifest else None)
        if not cases:raise ValueError("No local PDF samples found; supply --manifest with original file paths")
        output=_private_output(args.output or PROJECT_ROOT/".local"/f"document_benchmark_{args.mode}.json")
        run=lambda path:process_document(path,mode=MODE_MAP[args.mode])
        reports=evaluate_documents(cases,base_dir=base_dir,processor=run,include_values=args.include_values)
        for case,report in zip(cases,reports):
            report["filename"]=Path(case["document_path"]).name
            report["mode"]=args.mode
        classified=[row["classification_correct"] for row in reports if row.get("classification_correct") is not None]
        result={"mode":args.mode,"case_count":len(reports),"gold_classification_count":len(classified),
            "classification_accuracy":sum(classified)/len(classified) if classified else None,
            "include_values":args.include_values,"reports":reports}
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    except (OSError,ValueError) as error:
        parser.exit(1,f"Benchmark configuration error: {error}\n")
    print(f"Document benchmark: mode={args.mode} cases={len(reports)} gold_labels={len(classified)} classification_accuracy={result['classification_accuracy']}")
    for report in reports:
        print(f"case={report['case_number']} type={report.get('predicted_type')} route={report.get('extraction_route')} missing={report.get('missing_field_count')} latency={report.get('end_to_end_latency_seconds',0):.3f}s error={report.get('error_type','none')}")
    print("Private report saved. Structured values are included only when --include-values was supplied.")

if __name__=="__main__":main()
