"""Offline stand-in for the model (used by `python -m eval.run_eval` without --live).

RuleFollowingPolicy reads the conversation and does exactly what the orchestrator prompt says:
search the procedure, evaluate, delegate the sections that are not complete, answer with claims
copied from evaluate_rules. It measures the PIPELINE (tools, workers, Gate 1, fallback), not the
quality of a language model; live runs (--live) use Ollama instead.
"""
import json
import re
import zlib

from config import MAX_NEXT_ACTIONS

OBSERVATION = re.compile(r"^OBSERVATION from (\w+): (.*)$", re.DOTALL)
EMBED_DIMS = 512


def hash_embedding(text: str, dims: int = EMBED_DIMS) -> list[float]:
    """Deterministic bag-of-words vector (offline embeddings)."""
    vector = [0.0] * dims
    for word in re.findall(r"[a-z0-9ğüşöçı]+", text.lower()):
        if len(word) > 2:
            vector[zlib.crc32(word.encode("utf-8")) % dims] += 1.0
    return vector


def decision(action: str, args: dict, thought: str) -> str:
    return json.dumps({"thought": thought, "action": action, "args": args})


def observations(messages: list[dict]) -> dict[str, list[dict]]:
    """Parsed tool observations by tool name (cut or error observations are skipped)."""
    found: dict[str, list[dict]] = {}
    for message in messages:
        match = OBSERVATION.match(message["content"]) if message["role"] == "user" else None
        if not match:
            continue
        try:
            data = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if "error" not in data:
            found.setdefault(match.group(1), []).append(data)
    return found


def brief(messages: list[dict]) -> dict:
    """The PROCEDURE line of the first user message."""
    for line in messages[1]["content"].splitlines():
        if line.startswith("PROCEDURE: "):
            return json.loads(line[len("PROCEDURE: "):])
    return {}


class RuleFollowingPolicy:
    """Scripted orchestrator + worker that never guesses (see module docstring)."""
    embed_model = "hash-bow-offline"

    def __init__(self, max_delegations: int = 3):
        self.max_delegations = max_delegations

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(text) for text in texts]

    def chat(self, messages: list[dict], schema: dict, component: str) -> str:
        if component == "worker":
            return decision("finish", {"summary": "", "suspected_mistakes": []}, "Section checked.")
        return self._orchestrate(messages)

    def _orchestrate(self, messages: list[dict]) -> str:
        seen = observations(messages)
        info = brief(messages)
        sections, status = info.get("sections", []), info.get("status")
        if not info.get("active_procedure"):
            step = self._identify(messages, seen)
            if step is not None:
                return step
            search = seen["search_procedures"][-1]
            sections, status = search["sections"], self._status(search)
        if "evaluate_rules" not in seen:
            return decision("evaluate_rules", {}, "Check every rule first.")
        evaluation = seen["evaluate_rules"][-1]
        pending = self._sections_to_delegate(sections, evaluation, seen)
        if pending:
            return decision("delegate_section", {"section_id": pending[0]}, "Delegate the next open section.")
        return self._final(evaluation, status)

    @staticmethod
    def _status(search: dict) -> str:
        chosen = next((m for m in search["matches"] if m["procedure_id"] == search["selected"]), {})
        return chosen.get("status", "verified")

    @staticmethod
    def _identify(messages: list[dict], seen: dict) -> str | None:
        if "search_procedures" not in seen:
            goal = messages[1]["content"].splitlines()[0].removeprefix("GOAL: ")
            return decision("search_procedures", {"query": goal}, "Find the procedure first.")
        search = seen["search_procedures"][-1]
        if search["selected"]:
            return None
        if search["matches"]:
            titles = "; ".join(m["title"] for m in search["matches"])
            return decision("ask_user", {"question": f"Which procedure do you mean: {titles}?"}, "The goal is ambiguous.")
        return decision("final_answer", {"summary": "No verified procedure is available for this request.",
                                         "claims": [], "next_actions": []}, "Nothing matches.")

    def _sections_to_delegate(self, sections: list[dict], evaluation: dict, seen: dict) -> list[str]:
        open_rules = set(evaluation["failing"]) | set(evaluation["unknown"])
        done = [item["section_id"] for item in seen.get("delegate_section", [])]
        if len(done) >= self.max_delegations:
            return []
        return [s["id"] for s in sections
                if s.get("delegate", True) and s["id"] not in done and open_rules & set(s["rules"])]

    @staticmethod
    def _final(evaluation: dict, status: str | None) -> str:
        claims = [{"rule_id": r, "status": "fail"} for r in evaluation["failing"]]
        claims += [{"rule_id": r, "status": "unknown"} for r in evaluation["unknown"]]
        blocking_unknown = [r for r in evaluation["unknown"] if r not in evaluation["not_blocking"]]
        actions = [{"title": f"Correct the problem reported for {r}", "ref": r} for r in evaluation["failing"]]
        actions += [{"title": f"Provide the missing information for {r}", "ref": r} for r in blocking_unknown]
        counts = evaluation["counts"]
        summary = (f"{counts['pass']} rules pass, {counts['fail']} fail and {counts['unknown']} need more information. "
                   "I only prepare; you decide what to submit.")
        if status == "unverified" or evaluation["unverified_rules"]:
            summary += " Some rules are unverified: check them with the official office."
        return decision("final_answer", {"summary": summary, "claims": claims,
                                         "next_actions": actions[:MAX_NEXT_ACTIONS]}, "Synthesize the findings.")
