"""fill_pdf tool (artifact): propose a PDF form filled from Twin facts (PyMuPDF form fields).

Values come only from facts; fields whose fact is unknown stay empty and are listed as missing.
"""
from pathlib import Path

import pymupdf
from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.tools.proposals import ProposalOutput, propose


class FillPdfInput(BaseModel):
    form_id: str = Field(description="A form id of the procedure")


def form_values(form_fields: dict[str, str], values: dict) -> tuple[dict[str, str], list[str]]:
    """PDF field -> text for known facts, and the facts still missing."""
    filled, missing = {}, []
    for pdf_field, fact in form_fields.items():
        value = values.get(fact)
        if value is None:
            missing.append(fact)
        else:
            filled[pdf_field] = value.isoformat() if hasattr(value, "isoformat") else str(value)
    return filled, missing


def fill_form(template: Path, values: dict[str, str], output: Path) -> Path:
    """Write a copy of `template` with its text widgets filled."""
    with pymupdf.open(template) as document:
        for page in document:
            for widget in page.widgets() or []:
                if widget.field_name in values:
                    widget.field_value = values[widget.field_name]
                    widget.update()
        output.parent.mkdir(parents=True, exist_ok=True)
        document.save(output)
    return output


def run(args: FillPdfInput, ctx: ToolContext) -> ProposalOutput:
    pack = require_pack(ctx)
    form = next((f for f in pack.procedure.forms if f.form_id == args.form_id), None)
    if form is None:
        forms = ", ".join(f.form_id for f in pack.procedure.forms) or "none"
        raise ToolError(f"Unknown form {args.form_id!r}. Forms: {forms}")
    filled, missing = form_values(form.fields, ctx.view().values)
    payload = {"form_id": form.form_id, "template": form.template, "values": filled, "missing": missing}
    preview = f"{len(filled)} fields filled, missing: {', '.join(missing) or 'none'}"
    return propose(ctx, pack, "fill_pdf", f"Filled form: {form.title}", payload, [], form.stage_id or None, preview)


SPEC = ToolSpec(
    name="fill_pdf",
    description="Propose a PDF form of the procedure filled with the known facts (unknown fields stay empty).",
    input_model=FillPdfInput, output_model=ProposalOutput, fn=run, kind="artifact", roles=("orchestrator",))
