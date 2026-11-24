"""Read-only MCP server over the Procedure Packs (ported from Marco's demonstrator)."""
import json
import subprocess
import sys

import pytest

from config import PROJECT_ROOT
from src.integrations.mcp.server import PROTOCOL_VERSION, McpServer, serve
from src.pack import load_all_packs
from tests.fakes import ScriptedBackend

PACKS = load_all_packs()


@pytest.fixture
def server(store):
    return McpServer(PACKS, lambda: store, ScriptedBackend)


def call(server, method, params=None, request_id=1):
    return server.handle({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})


def tool(server, name, arguments):
    result = call(server, "tools/call", {"name": name, "arguments": arguments})["result"]
    return json.loads(result["content"][0]["text"]), result["isError"]


def test_handshake_and_ping(server):
    assert call(server, "initialize")["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert call(server, "ping")["result"] == {}


def test_only_read_only_tools_are_listed(server):
    tools = call(server, "tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == {"find_procedure", "search_official_sources"}
    assert all(t["annotations"]["readOnlyHint"] and not t["annotations"]["destructiveHint"] for t in tools)
    assert all(t["inputSchema"]["type"] == "object" for t in tools)


def test_every_pack_is_a_resource_with_its_rules(server):
    uris = {r["uri"] for r in call(server, "resources/list")["result"]["resources"]}
    assert uris == {f"procedure://{pack_id}" for pack_id in PACKS}
    text = call(server, "resources/read", {"uri": "procedure://tc_kimlik_first"})["result"]["contents"][0]["text"]
    summary = json.loads(text)
    assert summary["rules"] and all(rule["status"] == "unverified" for rule in summary["rules"])
    known = {source["source_id"] for source in summary["sources"]}
    assert all(set(rule["source_ids"]) <= known for rule in summary["rules"])


@pytest.mark.parametrize("goal, expected", [
    ("I lost my TC kimlik card", "tc_kimlik_renewal"),
    ("first Turkish ID card after citizenship", "tc_kimlik_first"),
    ("I want to apply for İkamet", "ikamet_first"),               # eval/goal_scenarios.json G01
    ("I want to open a bank account", None),
])
def test_find_procedure_routes_kimlik_and_ikamet(server, goal, expected):
    data, is_error = tool(server, "find_procedure", {"query": goal})
    assert not is_error and data["selected"] == expected


def test_find_procedure_does_not_touch_any_twin(server, store):
    tool(server, "find_procedure", {"query": "I lost my TC kimlik card"})
    assert store._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_search_returns_cited_passages_from_the_chosen_pack(server):
    data, is_error = tool(server, "search_official_sources",
                          {"procedure_id": "tc_kimlik_renewal", "query": "lost identity card appointment", "k": 2})
    assert not is_error and len(data["hits"]) == 2
    pack_sources = {s.source_id for s in PACKS["tc_kimlik_renewal"].sources}
    assert all(hit["source_id"] in pack_sources for hit in data["hits"])
    assert data["status"].startswith("UNVERIFIED")


@pytest.mark.parametrize("arguments", [{"procedure_id": "ikamet_first", "query": "x"},
                                       {"procedure_id": "ikamet_first", "query": "fee", "k": 9}])
def test_invalid_arguments_are_a_tool_error_not_a_crash(server, arguments):
    data, is_error = tool(server, "search_official_sources", arguments)
    assert is_error and data["error"].startswith("Invalid arguments")


def test_unknown_procedure_tool_and_method_are_reported(server):
    data, is_error = tool(server, "search_official_sources", {"procedure_id": "visa", "query": "fee amount"})
    assert is_error and "find_procedure" in data["error"]
    assert call(server, "tools/call", {"name": "draft_email", "arguments": {}})["error"]["code"] == -32602
    assert call(server, "resources/read", {"uri": "ikamet://x"})["error"]["code"] == -32602
    assert call(server, "shutdown_everything")["error"]["code"] == -32601


def test_bad_json_and_notifications(server):
    out = []
    serve(server, ["not json\n", json.dumps({"jsonrpc": "2.0", "method": "ping"}) + "\n",
                   json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n"], out.append)
    assert [json.loads(line).get("id") for line in out] == [None, 7]          # notification gets no reply
    assert json.loads(out[0])["error"]["code"] == -32700


def test_stdio_subprocess_answers_handshake_and_lists():
    """The real entry point over stdin/stdout; no model is needed for these methods."""
    requests = [{"jsonrpc": "2.0", "id": i, "method": m}
                for i, m in enumerate(["initialize", "tools/list", "resources/list"], 1)]
    done = subprocess.run([sys.executable, "-m", "src.integrations.mcp.server"], cwd=PROJECT_ROOT, timeout=60,
                          input="".join(json.dumps(r) + "\n" for r in requests), capture_output=True, text=True)
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2, 3], done.stderr
    assert len(replies[2]["result"]["resources"]) == len(PACKS)
