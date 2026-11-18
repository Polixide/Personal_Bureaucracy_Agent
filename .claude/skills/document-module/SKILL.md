---
name: document-module
description: Use after implementing or changing a module in src/ so teammates can understand it. Writes a one-page plain-language explanation in docs/modules/.
---

# Document a module

Create or update `docs/modules/<module_name>.md` (max one page) with:

1. **What it does** in two sentences.
2. **Where it sits in the flow**: which node/function calls it, what it returns.
3. **Inputs and outputs**: the Pydantic models involved, with one small example of real data.
4. **Why it is built this way**: the one or two design decisions a teammate might question.
5. **How to test it**: the exact pytest command.
6. **Known limits**.

Use plain words, no jargon without a short explanation. Do not paste the whole source code.
Language: English (switch if the user asks).
