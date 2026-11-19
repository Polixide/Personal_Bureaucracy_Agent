"""Current core readiness replaces B's obsolete R004 category-list check."""
from src.workers.rules_engine import evaluate_requirements
from src.core.state import ResidencePermitContext


def test_core_readiness_and_missing_count_are_separate_requirements():
    results={r.rule_id:r for r in evaluate_requirements({"residence_permit_context":ResidencePermitContext(all_core_documents_present=True,missing_required_documents_count=0)})}
    assert results["R020"].passed is True
    assert results["R026"].passed is True
    assert results["R004"].passed is None


def test_uploaded_category_alone_does_not_prove_legal_readiness():
    results={r.rule_id:r for r in evaluate_requirements({"documents":[]})}
    assert results["R020"].passed is None
    assert results["R026"].passed is None
