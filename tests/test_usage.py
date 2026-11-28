"""Provider choice at run time, the usage log, the cloud request cap and the cost estimate (src/llm.py).

No network: fake clients answer like the SDKs (a `usage` object for OpenAI and Anthropic, eval counts for Ollama).
"""
from types import SimpleNamespace

import pytest

import src.llm as llm
from src.agent.chat import review_enabled
from src.llm import LLMError, ModelCall, OpenAIBackend, UsageLog, estimated_cost, token_counts, usage_summary


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeOpenAI:
    """Replies like chat.completions.create, with token usage."""

    def __init__(self, text='{"ok": true}'):
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        message = SimpleNamespace(content=self.text, refusal=None)
        usage = SimpleNamespace(prompt_tokens=1200, completion_tokens=300)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


@pytest.fixture()
def fresh_usage(monkeypatch):
    log = UsageLog()
    monkeypatch.setattr(llm, "USAGE", log)
    return log


# ------------------------------------------------------------ provider --
@pytest.mark.parametrize("provider,kind", [("openai", llm.OpenAIBackend), ("anthropic", llm.AnthropicBackend),
                                           ("ollama", llm.OllamaBackend)])
def test_the_app_choice_wins_over_the_config_setting(monkeypatch, provider, kind):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    assert isinstance(llm.agent_backend("team", provider), kind)
    assert llm.is_cloud(provider) == (provider != "ollama")
    assert isinstance(llm.agent_backend("team"), llm.OllamaBackend)        # no choice: config.LLM_PROVIDER


def test_documents_go_to_the_cloud_only_with_the_switch_and_a_cloud_model(monkeypatch):
    monkeypatch.setattr(llm, "DOCUMENT_PROVIDER", "local")
    assert not llm.documents_in_cloud("openai")                             # the switch is off by default
    assert llm.documents_in_cloud("openai", read_in_cloud=True)
    assert not llm.documents_in_cloud("ollama", read_in_cloud=True)        # no cloud model to read them
    assert llm.is_cloud_backend(llm.document_backend("openai", True))
    assert not llm.is_cloud_backend(llm.document_backend("openai", False))


def test_a_cloud_provider_needs_its_key_in_the_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "OPENAI_API_KEY" in llm.missing_key("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.missing_key("openai") == ""
    assert llm.missing_key("ollama") == ""                                  # the local model needs no key


def test_self_review_follows_the_model_in_use(monkeypatch):
    import src.agent.chat as chat
    monkeypatch.setattr(chat, "SELF_REVIEW", "auto")
    assert review_enabled(True) and not review_enabled(False)
    monkeypatch.setattr(chat, "SELF_REVIEW", "1")
    assert review_enabled(False)


# --------------------------------------------------------------- usage --
def test_token_counts_read_every_provider_format():
    assert token_counts(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))) == (10, 3)
    assert token_counts(SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=2))) == (7, 2)
    assert token_counts(SimpleNamespace(prompt_eval_count=50, eval_count=9)) == (50, 9)       # Ollama
    assert token_counts(SimpleNamespace()) == (None, None)                                     # a test double


def test_a_cloud_call_is_logged_with_its_tokens(fresh_usage):
    backend = OpenAIBackend(model="gpt-6.1-terra", client=FakeOpenAI(), embedder=FakeEmbedder())
    backend.chat([{"role": "user", "content": "hi"}], {"type": "object"}, "chat")
    [call] = fresh_usage.since(0)
    assert call.cloud and call.ok and (call.input_tokens, call.output_tokens) == (1200, 300)
    assert call.component == "chat" and call.model == "gpt-6.1-terra"


def test_the_cloud_cap_refuses_the_next_request(fresh_usage, monkeypatch):
    monkeypatch.setattr(llm, "CLOUD_MAX_REQUESTS", 1)
    backend = OpenAIBackend(model="gpt-6.1-terra", client=FakeOpenAI(), embedder=FakeEmbedder())
    backend.chat([{"role": "user", "content": "hi"}], {"type": "object"}, "chat")
    with pytest.raises(LLMError):                       # an LLMError: the agent falls back to the plan by code
        backend.chat([{"role": "user", "content": "again"}], {"type": "object"}, "chat")
    assert fresh_usage.cloud_requests() == 1


def test_the_usage_log_measures_one_run_from_a_mark(fresh_usage):
    fresh_usage.add(ModelCall(component="a", model="m", cloud=False, ok=True, seconds=1.0))
    mark = fresh_usage.mark()
    fresh_usage.add(ModelCall(component="b", model="m", cloud=False, ok=True, seconds=2.0))
    assert [call.component for call in fresh_usage.since(mark)] == ["b"]


# ---------------------------------------------------------------- cost --
def test_cost_uses_the_price_list_and_ignores_local_and_failed_calls(monkeypatch):
    monkeypatch.setattr(llm, "CLOUD_PRICES", {"gpt-6.1-terra": (1.0, 5.0)})
    calls = [ModelCall(component="c", model="gpt-6.1-terra", cloud=True, ok=True, seconds=1.0,
                       input_tokens=1_000_000, output_tokens=200_000),
             ModelCall(component="c", model="qwen3:4b", cloud=False, ok=True, seconds=9.0, input_tokens=5, output_tokens=5),
             ModelCall(component="c", model="gpt-6.1-terra", cloud=True, ok=False, seconds=0.5)]
    assert estimated_cost(calls) == pytest.approx(2.0)                      # 1 M in x $1 + 0.2 M out x $5
    summary = usage_summary(calls)
    assert summary["requests"] == 3 and summary["cloud_requests"] == 2 and summary["failures"] == 1
    assert summary["input_tokens"] == 1_000_005 and summary["cost_usd"] == pytest.approx(2.0)


def test_a_model_without_a_price_has_no_cost(monkeypatch):
    monkeypatch.setattr(llm, "CLOUD_PRICES", {})
    calls = [ModelCall(component="c", model="claude-x", cloud=True, ok=True, seconds=1.0, input_tokens=10,
                       output_tokens=10)]
    assert estimated_cost(calls) is None and usage_summary(calls)["input_tokens"] == 10
