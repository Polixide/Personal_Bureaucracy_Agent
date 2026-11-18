"""draft_email tool (artifact): propose an e-mail DRAFT to a contact of the procedure. Never sent.

Recipients come from procedure.json `contacts`; an address is used only if the pack has one.
The text goes through Gate 1's text checks before it can even be proposed.
"""
from pydantic import BaseModel, Field

from src.tools.common import ToolContext, ToolError, ToolSpec, require_pack
from src.tools.proposals import ProposalOutput, check_rule_ids, check_stage, check_text, propose

UNKNOWN_ADDRESS = "(address not in the procedure sources: fill it in yourself)"


class DraftEmailInput(BaseModel):
    to_role: str = Field(description="A contact role of the procedure (e.g. IRO)")
    subject: str = Field(min_length=3, max_length=150)
    body: str = Field(min_length=10, max_length=3000, description="Polite, factual text; no promises, no invented dates")
    rule_ids: list[str] = Field(default_factory=list, description="Rules the e-mail is about")
    stage_id: str | None = Field(default=None, description="If the e-mail starts an official stage, its stage id")


def run(args: DraftEmailInput, ctx: ToolContext) -> ProposalOutput:
    pack = require_pack(ctx)
    contact = next((c for c in pack.procedure.contacts if c.role.casefold() == args.to_role.casefold()), None)
    if contact is None:
        roles = ", ".join(c.role for c in pack.procedure.contacts) or "none"
        raise ToolError(f"Unknown contact role {args.to_role!r}. Contacts: {roles}")
    check_rule_ids(pack, args.rule_ids)
    check_stage(pack, args.stage_id)
    check_text(ctx, f"{args.subject}\n{args.body}")
    payload = {"to_name": contact.name, "to_email": contact.email or UNKNOWN_ADDRESS,
               "subject": args.subject, "body": args.body}
    return propose(ctx, pack, "draft_email", f"E-mail draft to {contact.name}: {args.subject}", payload,
                   args.rule_ids, args.stage_id, args.body)


SPEC = ToolSpec(
    name="draft_email",
    description="Propose an e-mail draft to a contact of the procedure (for example IRO). It is saved as a draft "
                "file after approval and never sent by the agent.",
    input_model=DraftEmailInput, output_model=ProposalOutput, fn=run, kind="artifact", roles=("orchestrator",))
