"""Grounded local email drafts and an optional explicitly approved Gmail adapter."""
import base64
from email.message import EmailMessage
import json
from hashlib import sha256
from pathlib import Path
from config import PROJECT_ROOT
import os
from urllib.request import Request, urlopen
from src.core.procedure import assess_document_collection
from src.tools.schemas import EmailDraftInput, EmailResult


def prepare_draft(state, args, call_id):
    package=None
    if args.purpose=='package_review':
        from src.core.procedure import iro_package_readiness
        package=iro_package_readiness(state)
        if not package['ready']:raise ValueError('Complete the IRO package and fingerprint/GI progress before preparing a package-review email')
        if set(args.attachment_ids)-set(package['attachment_ids']):raise ValueError('Select only accepted package attachments')
    recipient = args.recipient or os.getenv("IRO_EMAIL") or "irolegal@sabanciuniv.edu"
    # Validate operator configuration by the same input boundary as user intent.
    if recipient:
        EmailDraftInput(recipient=recipient)
    available = {getattr(doc, "doc_id", None): doc for doc in state.get("documents", [])
                 if getattr(doc, "status", None) == "analyzed"}
    if any(doc_id not in available for doc_id in args.attachment_ids):
        raise ValueError("An attachment must reference an existing analyzed document")
    attachments = [{"document_id": doc_id, "document_type": available[doc_id].doc_type,
                    "file_path": available[doc_id].file_path} for doc_id in dict.fromkeys(args.attachment_ids)]
    for attachment in attachments:
        path = Path(attachment["file_path"]).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError("Selected attachment is unavailable in the local project")
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Attachment exceeds the local 20 MB email safety limit")
        attachment["content_sha256"] = sha256(path.read_bytes()).hexdigest()
    if sum(Path(item["file_path"]).stat().st_size for item in attachments) > 20 * 1024 * 1024:
        raise ValueError("Selected attachments exceed the 20 MB total email safety limit")
    procedure = state.get("procedure")
    stage = procedure.get("current_stage") if isinstance(procedure, dict) else getattr(procedure, "current_stage", None)
    assessment = assess_document_collection(state)
    ready_count = sum(value == "READY" for value in assessment.prerequisites.values())
    body = ("Dear International Relations Office,\n\n"
        "I am preparing my first short-term residence-permit application as an incoming exchange student.\n"
        f"My assistant records the preparation stage as {stage.replace('_', ' ').title() if stage else 'not established'} and {ready_count} of 6 document prerequisites as ready.\n"
        "Could you please advise on the current review arrangements and any unresolved preparation information?\n\n"
        "This draft does not claim official eligibility, application submission or permit approval.\n")
    if package is not None:
        body=("Dear International Relations Office,\n\nI have prepared the supported document package for my first short-term residence-permit application as an incoming exchange student. "
            "Please review the selected document scans and advise on an appointment after your verification.\n"
            "This is a request for IRO package review, not physical submission or residence-permit approval.\n")
    if attachments:
        body += "\nDocuments selected for attachment: " + ", ".join(item["document_type"].replace("_", " ") for item in attachments) + ".\n"
    else:
        body += "\nNo attachments have been selected.\n"
    body += "\nKind regards\n"
    return EmailResult(purpose=args.purpose,draft_id=call_id, recipient=recipient, recipient_source_ref="SAB_EXCHANGE_CURRENT" if not args.recipient and not os.getenv("IRO_EMAIL") else "user_or_operator_input", subject=args.subject, body=body, attachments=attachments)


class GmailAdapter:
    def __init__(self, token, *, transport=urlopen):
        self.token, self.transport = token, transport

    def send(self, draft):
        from pathlib import Path
        from config import PROJECT_ROOT
        message = EmailMessage()
        message["To"], message["Subject"] = draft.recipient, draft.subject
        message["Message-ID"] = "<" + draft.draft_id + "@ikamet-local>"
        message.set_content(draft.body)
        for attachment in draft.attachments:
            path = Path(attachment["file_path"]).resolve()
            if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
                raise ValueError("Selected attachment is unavailable or outside the local project")
            content = path.read_bytes()
            if sha256(content).hexdigest() != attachment.get("content_sha256"):
                raise ValueError("Attachment content changed after draft preparation")
            subtype = "pdf" if path.suffix.lower() == ".pdf" else "octet-stream"
            message.add_attachment(content, maintype="application", subtype=subtype, filename=path.name)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        request = Request("https://gmail.googleapis.com/gmail/v1/users/me/messages/send", data=json.dumps({"raw": raw}).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}, method="POST")
        with self.transport(request, timeout=30) as response:
            return str(json.load(response)["id"])


def configured_email_adapter():
    token = os.getenv("GMAIL_ACCESS_TOKEN")
    return GmailAdapter(token) if token else None


def email_app_link(draft):
    """Open the user's email composer, never send or contact an email API."""
    from urllib.parse import urlencode,quote
    data=draft.model_dump(mode='json') if hasattr(draft,'model_dump') else draft
    EmailDraftInput(recipient=data['recipient'],subject=data['subject'])
    if not data['recipient']:raise ValueError('Confirm a recipient before opening the email app')
    return 'mailto:'+quote(data['recipient'],safe='@')+'?'+urlencode({'subject':data['subject'],'body':data['body']},quote_via=quote)


def email_text(draft):
    data=draft.model_dump(mode='json') if hasattr(draft,'model_dump') else draft
    return 'To: '+str(data.get('recipient') or '')+'\nSubject: '+str(data.get('subject') or '')+'\n\n'+str(data.get('body') or '')


def email_file(draft):
    """Portable EML draft; attachments stay source/hash-checked, never auto-send."""
    from email.policy import SMTP
    data=EmailResult.model_validate(draft)
    message=EmailMessage(policy=SMTP)
    message['To']=data.recipient or '';message['Subject']=data.subject
    message['Message-ID']='<'+data.draft_id+'@ikamet-local>'
    message['X-Unsent']='1';message.set_content(data.body)
    total=0
    for attachment in data.attachments:
        path=Path(attachment['file_path']).resolve()
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file() or path.suffix.lower() not in {'.pdf','.txt'}:raise ValueError('An attachment is unavailable; prepare a fresh draft')
        content=path.read_bytes();total+=len(content)
        if total>20*1024*1024 or sha256(content).hexdigest()!=attachment.get('content_sha256'):raise ValueError('An attachment changed; prepare a fresh draft')
        message.add_attachment(content,maintype='application',subtype='pdf' if path.suffix.lower()=='.pdf' else 'octet-stream',filename=path.name)
    return message.as_bytes()


def gmail_compose_link(draft):
    """User-controlled web composer. No Gmail API, OAuth, attachments or send."""
    from urllib.parse import urlencode,quote
    data=draft.model_dump(mode='json') if hasattr(draft,'model_dump') else draft
    EmailDraftInput(recipient=data['recipient'],subject=data['subject'])
    if not data['recipient']:raise ValueError('Confirm a recipient before opening Gmail')
    return 'https://mail.google.com/mail/?'+urlencode({'view':'cm','fs':'1','to':data['recipient'],'su':data['subject'],'body':data['body']},quote_via=quote)
