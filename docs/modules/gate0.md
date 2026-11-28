# `src/agent/gate0.py` — Gate 0: checking what the user sends

## What it does
Gate 0 checks, with plain code and no model, everything the user sends before it can cost a model call or reach the
Digital Twin: chat messages, uploaded files and the profile name. A refusal is a list of `Issue`s (a code and a
sentence the user understands); an empty list means "pass".

## Where it sits in the flow
```
user ─► Gate 0 ─► agent (ReAct: decide → act → observe) ─► Gate 1 ─► answer
```
- **`run_chat`** (`src/agent/chat.py`) calls `check_message` first. On a refusal it returns a `ChatReply` with
  `status="invalid_input"` at once: no model call, no chat message or run stored. The app shows it as a warning
  ("Gate 0: …").
- **`save_upload`** (`app.py`) calls `check_upload` before writing the file, for uploads in the Documents view, in
  the Now box and in the chat.
- **`sidebar`** (`app.py`) calls `check_profile`; an invalid name falls back to `demo-user`.
- The model has the same kind of gate at the tool boundary: `run_tool` (`src/tools/registry.py`) validates every
  tool call with Pydantic, and an invalid call becomes an error observation instead of running.

## Inputs and outputs
Input: a string, or a file name plus its bytes. Output: `list[Issue]` (`src/models.py`).
```python
check_message("")                       # [Issue(code="EMPTY_MESSAGE", message="The message is empty: ...")]
check_upload("passport.pdf", b"hello")  # [Issue(code="WRONG_CONTENT", item="pdf",
                                        #        message="This file is not a real .pdf (it may have been renamed)...")]
check_upload("scan.png", png_bytes)     # []
```
Codes: `EMPTY_MESSAGE`, `MESSAGE_TOO_LONG`, `UNSUPPORTED_FILE`, `EMPTY_FILE`, `FILE_TOO_LARGE`, `WRONG_CONTENT`,
`TOO_MANY_PAGES`, `INVALID_PROFILE`. Limits live in `config.py`: `UPLOAD_TYPES`, `GATE0_MAX_MESSAGE_CHARS` (4000),
`GATE0_MAX_UPLOAD_MB` (20), `GATE0_MAX_PDF_PAGES` (60).

## Why it is built this way
- **It mirrors Gate 1.** The milestone-1 design has two deterministic gates: Gate 0 on the input, Gate 1 on the
  model's answer. Both return the same `Issue` shape, so the app shows them the same way.
- **It only checks what can be decided without reading the document.** The type is checked by the first bytes
  (`%PDF`, PNG and JPEG signatures), not by the name alone. A PDF that PyMuPDF cannot open is *not* refused here:
  the document reader gives the precise reason (password, no text, damaged), so the user gets one clear message.

## How to test it
```
python -m pytest -q tests/test_gate0.py
```

## Known limits
- It does not judge meaning: a message that is long but off-topic passes (the agent and Gate 1 handle it).
- A TXT file is not checked for its encoding; the reader handles that.
- It is not a security scanner: it does not look for malware inside a PDF.
