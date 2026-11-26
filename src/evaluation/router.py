"""Run the common synthetic cases through the application's actual Router."""
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from config import MAX_REFLECTION_ATTEMPTS
from src.agents.planner import canonical_candidates
from src.core.legal_stay import INSURANCE_PREPARATION_ADVISORY
from src.core.llm_diagnostics import collect_model_calls
from src.core.model_provider import AIOptions, use_ai_options
from src.evaluation.rag_eval import labeled_guidance_metrics
from src.core.state import Document, ResidencePermitContext, UserEvent, VerifiedInput, new_run
from src.document_processing.schemas import DOCUMENT_SCHEMAS
from src.evaluation.metrics import (
    complete_total, exact_action_score, outcome_assertions, ratio, retrieval_metrics, summarize_calls, trajectory_metrics,
)
from src.workers.document_analyzer import TYPE_MAP
from src.workers.verification import MILESTONE_SCOPE, grounded_values

RAG_ENABLED = True
WORKERS = {"document_analyzer", "rules_engine", "planner", "reflection", "assistant_worker"}
UNSAFE_CODES = {"PREMATURE_SUBMISSION", "COMPLETION_CLAIM", "APPROVAL_REQUIRED",
                "UNSUPPORTED_APPROVAL_CLAIM", "MISSTATED_RULE_SCOPE"}


@dataclass
class GateVisit:
    accepted: bool
    materialized: bool
    origin: str | None
    plan_hash: str | None
    errors: tuple[str, ...]
    unsafe_actions: int


@dataclass
class WorkflowTrace:
    nodes: Counter = field(default_factory=Counter)
    gates: list[GateVisit] = field(default_factory=list)
    fallback_used: bool = False
    _plan_origin: str | None = None
    _materialized: bool = False
    _plan_hash: str | None = None

    def observe(self, node, update):
        self.nodes[node] += 1
        if node == "planner":
            self._plan_origin = update.get("plan_origin")
            self._materialized = update.get("candidate_plan") is not None and not update.get("validation_errors")
            self._plan_hash = update.get("last_model_plan_hash")
            self.fallback_used |= self._plan_origin == "deterministic_fallback"
        if node == "gate_1":
            errors = update.get("validation_errors", ())
            self.gates.append(GateVisit(
                not errors and update["run_status"] == "VALIDATED", self._materialized,
                self._plan_origin, self._plan_hash, tuple(sorted({error.code for error in errors})),
                len({error.item for error in errors if error.item not in {None, "plan_summary"} and error.code in UNSAFE_CODES}),
            ))

    def metrics(self, state):
        rejected = [gate for gate in self.gates if not gate.accepted]
        repairs = state.get("retry_count", 0)
        final_gate = self.gates[-1] if self.gates else None
        recovered = bool(repairs and final_gate and final_gate.accepted)
        changed, unchanged, unclassified = 0, 0, 0
        for before, after in zip(self.gates, self.gates[1:]):
            if not after.accepted:
                if "UNCHANGED_INVALID_PLAN" in after.errors:
                    unchanged += 1
                elif before.plan_hash and after.plan_hash and before.plan_hash != after.plan_hash:
                    changed += 1
                else:
                    unclassified += 1
        human_review = state["run_status"] == "HUMAN_REVIEW_REQUIRED"
        errors = {error.code for error in state.get("validation_errors", [])}
        outcomes = state.get("tool_outcomes", [])
        tools = trajectory_metrics(outcomes)
        return {
            "plan_valid": final_gate.accepted if final_gate else None,
            "validated_plan_origin": final_gate.origin if final_gate and final_gate.accepted else None,
            "gate_submissions": len(self.gates), "gate_acceptances": len(self.gates) - len(rejected),
            "materialized_gate_submissions": sum(gate.materialized for gate in self.gates),
            "gate_rejections": len(rejected), "gate_error_codes": [list(gate.errors) for gate in rejected],
            "initial_rejection": bool(self.gates and not self.gates[0].accepted),
            "reflection_attempts": repairs, "unchanged_invalid_replans": unchanged,
            "changed_invalid_replans": changed, "unclassified_invalid_replans": unclassified,
            "retry_exhausted": human_review and repairs >= MAX_REFLECTION_ATTEMPTS and "UNCHANGED_INVALID_PLAN" not in errors,
            "recovered_by_llm_repair": recovered and final_gate.origin in {"ollama", "openai"},
            "recovered_by_fallback": recovered and final_gate.origin == "deterministic_fallback",
            "fallback_used": self.fallback_used, "human_review_required": human_review,
            "terminal_issue": min(errors) if errors else None,
            "unsafe_actions": sum(gate.unsafe_actions for gate in rejected),
            "dependency_violations": sum("MISSING_PREREQUISITE" in gate.errors for gate in rejected),
            "premature_submission_rejections": sum("PREMATURE_SUBMISSION" in gate.errors for gate in rejected),
            "premature_completion_claims": sum("COMPLETION_CLAIM" in gate.errors for gate in rejected),
            "unsupported_legal_claim_rejections": sum(
                bool({"MISSTATED_RULE_SCOPE", "UNSUPPORTED_APPROVAL_CLAIM"}.intersection(gate.errors)) for gate in rejected
            ),
            "unknown_as_failure_rejections": sum("UNKNOWN_AS_FAILURE" in gate.errors for gate in rejected),
            "worker_calls": sum(count for name, count in self.nodes.items() if name in WORKERS),
            "router_node_calls": dict(sorted(self.nodes.items())),
            "graph_steps": sum(self.nodes.values()),
            "tool_calls": tools["tool_calls"],
            "failed_tool_calls": sum(row.get("status") == "FAILED" and not row.get("replayed") for row in outcomes),
            "rag_queries": sum(row.get("name") == "search_official_sources" and
                               row.get("status") in {"SUCCEEDED", "FAILED"} and not row.get("replayed") for row in outcomes),
        }


def _accepted_input(suite, spec, event):
    initial = spec["initial"]
    profile = {**suite["profile"], **initial.get("profile", {})}
    exchange = profile["student_type"] == "exchange"
    context = ResidencePermitContext(
        institution=profile["host_university"], exchange_student=exchange,
        applicant_type="incoming_exchange" if exchange else profile["student_type"],
        application_kind=profile["application_type"], permit_type=profile["permit_category"],
        age=profile["age"], planned_stay_months=profile["planned_stay_months"],
    )
    fixture_user = "synthetic-benchmark"
    state = new_run(event, fixture_user, scope=MILESTONE_SCOPE)
    records = {}

    def verified(document_id, fields):
        return {
            key: VerifiedInput(value=value, origin="user_confirmed", confirmed_by=fixture_user,
                               document_id=document_id, field_name=key, trace_id=state["trace_id"])
            for key, value in fields.items() if value is not None
        }

    declarations = {**suite["declarations"], **initial.get("declarations", {})}
    records["__declarations__"] = verified("__declarations__", declarations)
    documents = []
    for name in initial["accepted_documents"]:
        fixture = suite["documents"][name]
        kind = fixture["document_type"]
        observations = {**fixture["observations"], **initial.get("observations", {}).get(name, {})}
        info = DOCUMENT_SCHEMAS[kind].model_validate(observations)
        doc_id = "fixture-" + name
        documents.append(Document(doc_id=doc_id, file_path="synthetic-" + name + ".txt",
                                  status="analyzed", doc_type=TYPE_MAP[kind], info=info))
        records[doc_id] = verified(doc_id, {**observations, **fixture.get("attestations", {})})
    return {**state, "residence_permit_context": context,
            "documents": documents, "verification_records": records}


MOCK_ACTIONS = {
    "R001": ("Upload your student certificate", "Provide evidence of your exchange program."),
    "R006": ("Upload Passport Identity Page", "Passport validity must be checked."),
    "R008": ("Upload Turkey Entry Stamp", "Entry evidence is missing."),
    "R015": ("Review and extend the insurance coverage period", "The recorded end date needs review for this preparation check."),
    "R019": ("Upload Declaration Control Document", "The signed declaration is required."),
}
DUPLICATE_REVIEWS = (
    ("R006", "Verify Passport Expiry Date", "Passport expiry must be verified."),
    ("R008", "Confirm Passport Pages", "The required passport pages must be checked."),
)


def _mock_replies(suite, spec):
    if spec["intent"] == "review_evidence":
        fixture = suite["documents"][spec["initial"]["incoming_document"]]
        yield {"document_type": fixture["document_type"]}
        yield {}
        return
    mock = spec.get("mock")
    if mock == "timeout":
        yield TimeoutError("Synthetic Ollama timeout")
        return
    actions = []
    for key in spec["expected"]["actions"][0] if mock != "incompatible_then_repeat" else ["R006", "R008", "R019"]:
        title, reason = MOCK_ACTIONS[key]
        if spec["id"] == "C":
            title = {"R006": "Review conflicting passport evidence", "R015": "Review the conflicting holder name on the insurance certificate"}[key]
            reason = "Confirm the conflicting observations with a human reviewer."
        actions.append({"candidate_ids": [key], "title": title, "reason": reason})
    valid = {"actions": actions, "summary": "Complete the missing document requirements."}
    if mock in {"compatible_duplicates", "incompatible_then_repair", "incompatible_then_repeat"}:
        reviews = [
            {"candidate_ids": [key], "title": title, "reason": reason} for key, title, reason in DUPLICATE_REVIEWS
        ]
        if mock != "compatible_duplicates":
            reviews[1]["reason"] = "Check a different passport page for eligibility."
        first = {**valid, "actions": actions + reviews}
        yield first
        if mock == "incompatible_then_repeat":
            yield {**first, "actions": list(reversed(first["actions"])), "summary": "Review the evidence requirements."}
            return
        if mock == "compatible_duplicates":
            return
    yield valid


def _mock_transport(suite, spec):
    replies = iter(_mock_replies(suite, spec))
    last = None

    def reply(**kwargs):
        nonlocal last
        last = next(replies, last)
        if isinstance(last, Exception):
            raise last
        draft = last
        if spec["intent"] == "prepare" and last is not None:
            wire = json.loads(kwargs["messages"][1]["content"])
            by_rule = {}
            for candidate in wire["canonical_candidates"]:
                for rule_id in candidate["rule_ids"]:
                    by_rule.setdefault(rule_id, []).append(candidate["candidate_id"])
            rows = []
            for row in last["actions"]:
                selectors = []
                for rule_id in row["candidate_ids"]:
                    matches = by_rule.get(rule_id, [])
                    selectors.append(matches[0] if len(matches) == 1 else "unmapped_candidate")
                rows.append({**row, "candidate_ids": selectors})
            draft = {**last, "actions": rows}
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(draft)))

    return patch("ollama.Client.chat", side_effect=reply)


def _document_fixture(suite, spec, directory, document_format):
    key = spec["initial"]["incoming_document"]
    text = suite["documents"][key]["synthetic_text"]
    path = Path(directory) / f"{key}.{'pdf' if document_format == 'scan' else document_format}"
    if document_format == "txt":
        path.write_text(text, encoding="utf-8")
    else:
        import pymupdf

        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((72, 72), text, fontsize=11)
            if document_format == "scan":
                with pymupdf.open() as scan:
                    scan.new_page().insert_image(page.rect, pixmap=page.get_pixmap(dpi=150))
                    scan.save(path)
            else:
                pdf.save(path)
    return path


def _extraction_scores(state, suite, spec):
    expected = spec["expected"].get("extraction")
    if expected is None:
        return {
            "classification_correct": None, "extraction_precision": None, "extraction_recall": None,
            "document_source_accuracy": None, "extraction_exact_fields": None,
            "extraction_reported_fields": None, "extraction_reference_fields": None,
        }
    truth = suite["documents"][expected["fields_from"]]["observations"]
    analyzed = [doc for doc in state.get("documents", []) if doc.status == "analyzed" and doc.info]
    document = analyzed[0] if len(analyzed) == 1 else None
    matches_type = bool(document and document.info.document_type == expected["document_type"])
    observations = {}
    if document:
        info = document.info
        values = info.model_dump(mode="json")
        observations = {key: values[key] for key in info.observation_fields() if values[key] is not None}
    correct = {key for key, value in truth.items() if matches_type and observations.get(key) == value}
    grounded = grounded_values(document) if matches_type else {}
    sourced = {key for key in correct if grounded.get(key) == observations[key]
               and document.info.source_pages.get(key) == expected["source_page"]}
    return {
        "classification_correct": matches_type,
        "extraction_precision": ratio(len(correct), len(observations)) or 0.0,
        "extraction_recall": ratio(len(correct), len(truth)),
        "document_source_accuracy": ratio(len(sourced), len(truth)),
        "extraction_exact_fields": len(correct), "extraction_reported_fields": len(observations),
        "extraction_reference_fields": len(truth),
    }


def _action_keys(state, actions):
    # Legacy deterministic fallback lacks grounding_ids. Map only an unambiguous
    # canonical candidate for the referenced rule and keep unknown bindings visible.
    by_rule = {}
    if any(not action.grounding_ids for action in actions):
        for candidate in canonical_candidates(state):
            for rule in candidate["rule_ids"]:
                by_rule.setdefault(rule, []).append(candidate["candidate_id"])
    result = []
    kinds = {}
    for action in actions:
        matches = by_rule.get(action.related_rule_id, [])
        keys = action.grounding_ids or (matches if len(matches) == 1 else ["unmapped"])
        result.extend(keys)
        kinds.update((key, action.action_type) for key in keys)
    return result, kinds


def _case_outcome(state, trace, call_rows, suite, spec, approval_reached):
    workflow = trace.metrics(state)
    calls = summarize_calls(call_rows)
    raw = [call for call in call_rows if call["proposed_actions"] is not None]
    replans = [call for call in call_rows if call["task"] == "reflection_replan"]
    actions = (state.get("candidate_plan") or []) if workflow["plan_valid"] else []
    candidate_ids, kinds = _action_keys(state, actions)
    assessment = {item.rule_id: item for item in state.get("relevant_requirements") or []}
    rule_names = {"passed": "PASS", "failed": "FAIL", "missing_information": "UNKNOWN"}
    actual_rules = {key: rule_names[item.status] for key, item in assessment.items()}
    expected = spec["expected"]
    labeled_basis = {key: suite["rule_basis"][key] for key in expected["rule_outcomes"] if key in suite["rule_basis"]}
    unknown = [key for key, value in expected["rule_outcomes"].items() if value == "UNKNOWN"]
    context = state.get("residence_permit_context")
    labeled_conflicts = expected.get("withhold_context", [])
    withheld = [key for key in labeled_conflicts if key in state.get("verification_conflicts", [])
                and getattr(context, key, None) is None]
    contradicted = [key for key in labeled_conflicts if getattr(context, key, None) is not None]
    safe = (state.get("approval_decision") is None and not state.get("approved_plan") and not trace.nodes["state_update"]
            and all(a.requires_approval and a.tool_call is None and a.execution_type in {"information", "prepare"} for a in actions)
            and not workflow["tool_calls"] and state.get("tool_approval_decision") is None
            and (not approval_reached or workflow["plan_valid"] is True))
    status = state["run_status"]
    stop = "human_review" if status == "HUMAN_REVIEW_REQUIRED" else "other"
    if status == "INVALID_INPUT" and any(
        error.source == "gate_0" and error.message.startswith("Outside Sabanc")
        for error in state.get("validation_errors", [])
    ):
        stop = "unsupported_profile"
    if approval_reached:
        stop = "approval_required"
    elif spec["intent"] == "review_evidence" and status in {"INFORMATION_NEEDS_CONFIRMATION", "DOCUMENT_ANALYZED"}:
        stop = "evidence_review"
    action_score = exact_action_score(candidate_ids, expected.get("actions"))
    observed = {
        **workflow, **calls, **_extraction_scores(state, suite, spec), **retrieval_metrics(RAG_ENABLED),
        "stop": stop, "human_approval_reached": approval_reached, "safety_boundary_intact": bool(safe),
        "rule_outcomes": {key: actual_rules.get(key) for key in expected["rule_outcomes"]},
        "applied_rule_count": len(actual_rules), "planning_requested": spec["intent"] == "prepare",
        "preparation_complete": workflow["plan_valid"] is True and state.get("plan_origin") == "milestone_completion",
        "validated_actions": len(actions), "validated_candidate_ids": candidate_ids, "action_types": kinds,
        "validated_duplicate_references": len(candidate_ids) - len(set(candidate_ids)),
        "blocker_rule_ids": sorted(state["blockers"]) if "blockers" in state else None,
        "remaining_blockers": len(state["blockers"]) if "blockers" in state else None,
        "advisories": ["insurance_coverage_shortfall"] if any(INSURANCE_PREPARATION_ADVISORY in step.reason for step in actions) else [],
        "withheld_context": withheld, "contradiction_rate": ratio(len(contradicted), len(labeled_conflicts)),
        "unknown_preservation_rate": ratio(sum(actual_rules.get(key) == "UNKNOWN" for key in unknown), len(unknown)),
        "source_attribution_accuracy": ratio(sum(key in assessment and set(assessment[key].source_ids) == set(label["source_ids"])
                                              for key, label in labeled_basis.items()), len(labeled_basis)),
        "scope_accuracy": None,  # This branch has no per-result scope_layer.
        "grounded_claim_rate": None, "unsupported_claim_rate": None,
        "raw_proposed_actions": complete_total(raw, "proposed_actions"),
        **{key: complete_total(raw, key) for key in (
            "unsupported_selector_actions", "duplicate_candidate_references", "unknown_candidate_references",
            "supplied_rule_references", "proposed_rule_references", "unsupported_rule_wording_actions")},
        "schema_valid": calls["schema_valid_outputs"] == calls["schema_outputs_evaluated"] if calls["schema_outputs_evaluated"] else None,
        "reflection_calls": len(replans),
        "reflection_latency_seconds": complete_total(replans, "wall_seconds"),
        **{f"reflection_{key}": complete_total(replans, key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        **(action_score or {key: None for key in ("action_precision", "action_recall", "action_f1", "missing_required_actions", "extra_actions")}),
        "validated_action_coverage": action_score["action_recall"] if action_score else None,
    }
    checks = outcome_assertions(spec, observed, action_score)
    return {**observed, "assertions": checks, "task_success": all(checks.values())}


def _guidance_outcome(state, trace, calls, spec):
    workflow = trace.metrics(state)
    diagnostics = state.get("assistant_diagnostics", {})
    reference = spec["expected"]
    relevant = reference.get("relevant_source_ids", [])
    rows = [event for event in state.get("assistant_events", [])
            if event.get("request_id") == diagnostics.get("request_id") and event.get("name") == "search_official_sources"]
    retrieved = [result for row in rows for result in row.get("data", {}).get("results", [])]
    rag = diagnostics.get("rag", {})
    queries = [{**row, "relevant": relevant, "retrieved": row["retrieved_source_ids"] if not row["failed"] else None,
                "source_coverage": ratio(len(set(row["retrieved_source_ids"]) & set(relevant)), len(set(relevant)))}
               for row in rag.get("queries", [])]
    k = queries[0]["k"] if queries else None
    scores = retrieval_metrics(rag.get("enabled", True), queries, k=k)
    counts = [row["result_count"] for row in queries if row["result_count"] is not None]
    scores["retrieved_chunk_count"] = sum(counts) / len(counts) if counts else None
    reply = next((row for row in reversed(state.get("assistant_messages", [])) if row.get("role") == "assistant"), {})
    answer = labeled_guidance_metrics(reply, retrieved, reference)
    checks = {"answer": answer["answer_correct"] is True,
              "retrieved_labeled_source": not relevant or scores["retrieval_hit_rate_at_k"] == 1.0,
              "no_planning_or_external_action": not workflow["gate_submissions"] and not state.get("tool_proposals"),
              "request_budget": len([row for row in calls if not row["task"].startswith("rag_embedding_")]) <= spec["budget"]["model_calls"]}
    return {**workflow, **summarize_calls(calls), **scores, **answer,
            "retrieval_configuration": {"mode": queries[0].get("mode") if queries else None,
                                        "embedding_model": queries[0].get("embedding_model") if queries else None, "k": k,
                                        "scope": queries[0].get("scope") if queries else None,
                                        "corpus_status": queries[0].get("corpus_status") if queries else None,
                                        "vector_status": queries[0].get("vector_status") if queries else None,
                                        "indexed_vectors": queries[0].get("indexed_vectors") if queries else None},
            "index_status": queries[0].get("vector_status") if queries else None,
            "indexed_chunks": queries[0].get("indexed_chunks") if queries else None,
            "rag_queries": len(queries), "planning_requested": False, "human_approval_reached": False,
            "safety_boundary_intact": checks["no_planning_or_external_action"],
            "assertions": checks, "task_success": all(checks.values())}


def _mock_guidance_transport():
    def reply(**kwargs):
        # Deterministic data selection. Offline quality scores assess retrieval
        # and attribution, independent of the quality of a live LLM.
        indices = kwargs["format"]["properties"]["source_indices"]["items"]["enum"]
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps({"source_indices": indices[:1]})))

    return patch("ollama.Client.chat", side_effect=reply)


def run_case(suite, spec, mode, artifact_directory, *, document_format="txt", ai=None):
    """Use a fresh synthetic in-memory thread and stop at the first human interrupt."""
    if mode not in spec["modes"] or not suite.get("synthetic_only"):
        raise ValueError("Scenario is unavailable in the selected synthetic mode")
    offline = mode in {"offline", "document-offline", "rag-offline"}
    documents = spec["intent"] == "review_evidence"
    guidance = spec["intent"] == "guidance"
    selected = ai or AIOptions()
    if offline and selected.provider != "ollama":
        raise ValueError("Offline benchmarks use the scripted transport")
    if documents and document_format not in {"pdf", "txt", "scan"}:
        raise ValueError("Use a designated native PDF, image-only PDF or TXT format")
    from langsmith import tracing_context
    from src.core.graph import build_graph

    with TemporaryDirectory(dir=artifact_directory, prefix="synthetic-") as directory, ExitStack() as stack:
        stack.enter_context(patch("config.TWIN_RECORD_PATH", Path(directory) / "synthetic-lessons.db"))
        stack.enter_context(patch("config.DOCUMENT_CACHE_ENABLED", False))
        stack.enter_context(patch("src.document_processing.processor.DOCUMENT_AI_PROVIDER", "local-only"))
        stack.enter_context(tracing_context(enabled=False))
        if offline:
            stack.enter_context(_mock_guidance_transport() if guidance else _mock_transport(suite, spec))
        if documents:
            path = str(_document_fixture(suite, spec, directory, document_format))
            event = UserEvent(kind="analyze_document", file_path=path)
        elif guidance:
            event = UserEvent(kind="assistant_message", text=spec["question"])
        else:
            event = UserEvent(kind="generate_plan")
        from dataclasses import replace

        scoped = replace(selected, cloud_document=("analyze_document", path) if documents and selected.provider == "openai" else None,
                         embedding_model="" if offline else selected.embedding_model)
        stack.enter_context(use_ai_options(scoped))
        initial = _accepted_input(suite, spec, event)
        graph = build_graph()
        config = {"configurable": {"thread_id": "synthetic-" + uuid4().hex}}
        trace = WorkflowTrace()
        with collect_model_calls() as calls:
            start = perf_counter()
            for event_update in graph.stream(initial, config, stream_mode="updates"):
                for node, update in event_update.items():
                    if node != "__interrupt__":
                        trace.observe(node, update or {})
            snapshot = graph.get_state(config)
            elapsed = perf_counter() - start
        approval = "human_approval" in snapshot.next and any(task.interrupts for task in snapshot.tasks)
        call_rows = [
            {**asdict(call), "model": "scripted" if offline else call.model}
            for call in calls
        ]
        return {
            **(_guidance_outcome(snapshot.values, trace, call_rows, spec) if guidance
               else _case_outcome(snapshot.values, trace, call_rows, suite, spec, approval)),
            "end_to_end_latency_seconds": elapsed, "model_calls": call_rows,
        }
