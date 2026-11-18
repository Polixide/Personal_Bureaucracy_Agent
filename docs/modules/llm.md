# `src/llm.py` — the only door to the language model

## What it does
Every chat and embedding call goes through this module. Structured calls send a JSON schema, validate the
reply with Pydantic, retry **once** with the validation error, and otherwise return `ok=False` so the caller
can use its deterministic fallback.

## Where it sits in the flow
- **`ask_structured(backend, messages, OutputModel, component)`** is used by the agent's orchestrator and
  workers (one `Decision` per step) and by generic document extraction.
- **`OllamaBackend`** is the real backend: `chat()` and `embed()` with the models and host from `config.py`.
- **`logged_chat(client, component=..., **kwargs)`** is used directly by the milestone-1 document processor.

## Inputs and outputs
```python
result = ask_structured(backend, [{"role": "user", "content": "..."}], Decision, "orchestrator")
result.ok        # True if the reply validated (attempt 1 or 2)
result.value     # the Decision instance
result.error     # "action: Input should be 'read_twin', ...", without echoing input values
result.attempts  # 1 or 2
```
Ollama call settings: `format=<schema>`, `temperature=0`, `think=False`, `stream=False`.

## Why it is built this way
- **The app must never crash because of the model.** Transport errors become `LLMError`, and
  `ask_structured` never raises. Callers decide the fallback; the agent switches to its deterministic workflow
  after 2 failed decisions in a row.
- **Backends are swappable.** Tests use `tests/fakes.py:ScriptedBackend`, which returns a fixed list of replies.
  That is how agent behaviour (invalid arguments, loops, wrong claims) is tested without a model.
- **Document extraction keeps its own validation.** It calls `logged_chat` without the retry, because its 118
  existing tests pin the exact number of model calls. Its fallback is "no facts, user confirms".

## How to test it
```
python -m pytest -q tests/test_llm.py
```
An autouse fixture in `tests/conftest.py` makes any real `ollama` chat or embed call fail the test.

## Known limits
- One retry only. A 4B model that fails twice gives up for that step; that is intended.
- The logs contain only component, model, outcome and duration, never prompts or replies.
