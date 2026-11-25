"""Cloud backends (src/llm.py): Claude and GPT through their official SDKs, with fake clients (no network).

What is checked: the schema sent as a constraint, the request shape, the fallbacks when a provider refuses the
schema, refusals and transport errors becoming LLMError (so the agent falls back instead of crashing), the
provider switch, and that embeddings stay local.
"""
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest

import src.llm as llm
from src.agent.answer import Decision, FinalAnswer
from src.llm import (AnthropicBackend, LLMError, OpenAIBackend, ask_structured, split_system, strict_schema)

REQUEST = httpx2.Request("POST", "https://api.example.invalid/v1")


def bad_request(kind):
    return kind("schema not accepted", response=httpx2.Response(400, request=REQUEST), body=None)


class FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeAnthropic:
    """Records each messages.create request; replies with scripted results (text, or an exception)."""

    def __init__(self, *results):
        self.results = list(results)
        self.requests = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        stop = "refusal" if result is None else "end_turn"
        return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=result or "")])


class FakeOpenAI:
    def __init__(self, *results):
        self.results = list(results)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        message = SimpleNamespace(content=result, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


ANSWER = '{"summary": "Upload your passport.", "claims": [], "next_actions": []}'
MESSAGES = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "STATE: {}"}]


# ------------------------------------------------------------- schemas --
def test_strict_schema_closes_every_object_and_drops_unsupported_limits():
    schema = strict_schema(FinalAnswer.model_json_schema())
    assert schema["additionalProperties"] is False
    assert all(definition["additionalProperties"] is False for definition in schema["$defs"].values())
    text = str(schema)
    assert "maxLength" not in text and "minLength" not in text and "maxItems" not in text


def test_a_free_form_object_cannot_be_a_strict_schema():
    assert strict_schema(Decision.model_json_schema()) is None      # args: any keys


def test_split_system_moves_system_text_out_and_ends_with_the_user():
    system, turns = split_system(MESSAGES + [{"role": "assistant", "content": "{}"}])
    assert system == "Rules." and turns[0]["role"] == "user" and turns[-1]["role"] == "user"


# ------------------------------------------------------------- Claude --
def test_claude_gets_the_schema_as_structured_output_and_its_reply_is_validated():
    client = FakeAnthropic(ANSWER)
    result = ask_structured(AnthropicBackend(model="claude-test", client=client, embedder=FakeEmbedder()),
                            MESSAGES, FinalAnswer, "chat")
    assert result.ok and result.value.summary == "Upload your passport."
    request = client.requests[0]
    assert request["model"] == "claude-test" and request["system"] == "Rules."
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in request


def test_a_schema_claude_cannot_constrain_is_asked_for_in_words():
    client = FakeAnthropic('{"thought": "t", "action": "final_answer", "args": {}}')
    backend = AnthropicBackend(client=client, embedder=FakeEmbedder())
    backend.chat(MESSAGES, Decision.model_json_schema(), "chat")
    request = client.requests[0]
    assert "format" not in request.get("output_config", {}) and "JSON schema" in request["system"]


def test_a_refused_schema_is_retried_without_the_constraint():
    client = FakeAnthropic(bad_request(anthropic.BadRequestError), ANSWER)
    text = AnthropicBackend(client=client, embedder=FakeEmbedder()).chat(MESSAGES, FinalAnswer.model_json_schema(), "c")
    assert text == ANSWER and "format" in client.requests[0]["output_config"]
    assert "format" not in client.requests[1].get("output_config", {})


def test_a_refusal_or_a_network_error_is_an_llm_error_so_the_agent_falls_back():
    refused = AnthropicBackend(client=FakeAnthropic(None), embedder=FakeEmbedder())
    with pytest.raises(LLMError):
        refused.chat(MESSAGES, FinalAnswer.model_json_schema(), "c")
    error = anthropic.APIConnectionError(request=REQUEST)
    offline = AnthropicBackend(client=FakeAnthropic(error, error), embedder=FakeEmbedder())   # first try + retry
    assert ask_structured(offline, MESSAGES, FinalAnswer, "c").ok is False


def test_embeddings_stay_on_the_local_model():
    assert AnthropicBackend(client=FakeAnthropic(), embedder=FakeEmbedder()).embed(["a"]) == [[1.0, 0.0]]
    assert OpenAIBackend(client=FakeOpenAI(), embedder=FakeEmbedder()).embed(["a"]) == [[1.0, 0.0]]


def test_the_cloud_backends_cannot_see_images():
    assert not hasattr(AnthropicBackend(client=FakeAnthropic(), embedder=FakeEmbedder()), "see")


# ---------------------------------------------------------------- GPT --
def test_gpt_gets_the_schema_in_json_schema_mode():
    client = FakeOpenAI(ANSWER)
    result = ask_structured(OpenAIBackend(model="gpt-test", client=client, embedder=FakeEmbedder()),
                            MESSAGES, FinalAnswer, "chat")
    assert result.ok
    request = client.requests[0]
    assert request["model"] == "gpt-test" and request["response_format"]["type"] == "json_schema"
    assert request["messages"][0] == {"role": "system", "content": "Rules."}


def test_gpt_falls_back_to_json_mode_when_the_schema_is_refused():
    client = FakeOpenAI(bad_request(openai.BadRequestError), ANSWER)
    text = OpenAIBackend(client=client, embedder=FakeEmbedder()).chat(MESSAGES, Decision.model_json_schema(), "c")
    assert text == ANSWER and client.requests[1]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in client.requests[1]["messages"][0]["content"]


def test_gpt_refusing_twice_is_an_llm_error():
    client = FakeOpenAI(bad_request(openai.BadRequestError), bad_request(openai.BadRequestError))
    with pytest.raises(LLMError):
        OpenAIBackend(client=client, embedder=FakeEmbedder()).chat(MESSAGES, Decision.model_json_schema(), "c")


# ------------------------------------------------------------- switch --
@pytest.mark.parametrize("provider,kind", [("anthropic", AnthropicBackend), ("openai", OpenAIBackend),
                                           ("ollama", llm.OllamaBackend)])
def test_the_provider_setting_chooses_the_agent_backend(monkeypatch, provider, kind):
    monkeypatch.setattr(llm, "LLM_PROVIDER", provider)
    assert isinstance(llm.agent_backend("chat"), kind)
    assert llm.is_cloud() == (provider != "ollama")
    assert isinstance(llm.local_backend(), llm.OllamaBackend)      # documents always stay local
