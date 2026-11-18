"""src/llm.py: schema-constrained calls, one retry, deterministic failure, Ollama wiring."""
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel

from config import EMBED_MODEL, LLM_MODEL
from fakes import ScriptedBackend
from src.llm import LLMError, OllamaBackend, ask_structured, clean_reply


class Answer(BaseModel):
    city: str
    year: int


MESSAGES = [{"role": "user", "content": "Where and when?"}]


def test_valid_reply_is_parsed_on_first_attempt():
    backend = ScriptedBackend([{"city": "Istanbul", "year": 2026}])
    result = ask_structured(backend, MESSAGES, Answer, "test")
    assert result.ok and result.value == Answer(city="Istanbul", year=2026) and result.attempts == 1
    assert backend.calls[0]["schema"] == Answer.model_json_schema()


def test_invalid_reply_gets_one_retry_with_the_error():
    backend = ScriptedBackend(['{"city": "Istanbul"}', {"city": "Istanbul", "year": 2026}])
    result = ask_structured(backend, MESSAGES, Answer, "test")
    assert result.ok and result.attempts == 2
    retry = backend.calls[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": '{"city": "Istanbul"}'}
    assert "year" in retry[-1]["content"] and "invalid" in retry[-1]["content"]


def test_two_invalid_replies_return_a_failure_instead_of_raising():
    backend = ScriptedBackend(["not json", '{"city": 5, "year": "x"}'])
    result = ask_structured(backend, MESSAGES, Answer, "test")
    assert not result.ok and result.attempts == 2 and result.value is None and result.error


def test_transport_errors_never_escape():
    backend = ScriptedBackend([LLMError("down"), LLMError("still down")])
    result = ask_structured(backend, MESSAGES, Answer, "test")
    assert not result.ok and "down" in result.error


def test_code_fences_are_removed():
    assert clean_reply('```json\n{"a": 1}\n```') == '{"a": 1}'
    backend = ScriptedBackend(['```json\n{"city": "Ankara", "year": 2027}\n```'])
    assert ask_structured(backend, MESSAGES, Answer, "test").value.city == "Ankara"


def test_ollama_backend_sends_schema_model_and_no_thinking():
    with patch("src.llm.Client") as client:
        client.return_value.chat.return_value = SimpleNamespace(message=SimpleNamespace(content='{"x":1}'))
        reply = OllamaBackend().chat(MESSAGES, {"type": "object"}, "unit")
    kwargs = client.return_value.chat.call_args.kwargs
    assert reply == '{"x":1}'
    assert kwargs["model"] == LLM_MODEL and kwargs["format"] == {"type": "object"}
    assert kwargs["think"] is False and kwargs["options"]["temperature"] == 0.0


@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), TimeoutError("slow"), ConnectionError("x")])
def test_ollama_transport_errors_become_llm_errors(error):
    with patch("src.llm.Client") as client:
        client.return_value.chat.side_effect = error
        with pytest.raises(LLMError):
            OllamaBackend().chat(MESSAGES, {}, "unit")


def test_ollama_embeddings_use_the_embedding_model():
    with patch("src.llm.Client") as client:
        client.return_value.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        vectors = OllamaBackend().embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.return_value.embed.call_args.kwargs["model"] == EMBED_MODEL


def test_autouse_guard_blocks_real_ollama_calls():
    with pytest.raises(AssertionError, match="must mock"):
        OllamaBackend().chat(MESSAGES, {}, "unit")
    with pytest.raises(AssertionError, match="must mock"):
        OllamaBackend().embed(["text"])


def test_logs_never_contain_prompt_text(capsys):
    with patch("src.llm.Client") as client:
        client.return_value.chat.return_value = SimpleNamespace(message=SimpleNamespace(content="{}"))
        OllamaBackend().chat([{"role": "user", "content": "SECRET PASSPORT P1234567"}], {}, "unit")
    err = capsys.readouterr().err
    assert "[LLM] unit model=" in err and "SECRET" not in err
