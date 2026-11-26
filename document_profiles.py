"""Three small extraction contracts and contextual labels; no orchestration code."""

from dataclasses import dataclass
import re

from pydantic import BaseModel, ConfigDict, Field

from schemas import REQUIRED_FIELDS


def concise_extraction_schema(schema: dict) -> None:
    """Require nullable field keys; bound optional quotes and put values first."""
    properties = schema.get("properties", {})
    fields = [name for name in properties if name not in {"source_snippets", "ambiguous_fields"}]
    schema["properties"] = {**{name: properties[name] for name in fields},
                            **{name: properties[name] for name in ("source_snippets", "ambiguous_fields") if name in properties}}
    schema["required"] = fields
    if "source_snippets" in properties:
        properties["source_snippets"].update(additionalProperties=False,
            properties={name: {"type": "string", "maxLength": 160} for name in fields})
    if "ambiguous_fields" in properties:
        properties["ambiguous_fields"].update(items={"type": "string", "enum": fields}, maxItems=len(fields))


class ExtractionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, json_schema_extra=concise_extraction_schema)
    source_snippets: dict[str, str] = Field(default_factory=dict)
    ambiguous_fields: list[str] = Field(default_factory=list)


class EnrollmentExtraction(ExtractionMetadata):
    holder_name: str | None = None
    student_id: str | None = None
    institution: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    enrollment_status: str | None = None


class InsuranceExtraction(ExtractionMetadata):
    holder_name: str | None = None
    insurer: str | None = None
    policy_number: str | None = None
    issue_date: str | None = None
    expiry_date: str | None = None
    coverage_start_date: str | None = None
    coverage_end_date: str | None = None


class ApplicationExtraction(ExtractionMetadata):
    applicant_name: str | None = None
    application_number: str | None = None
    passport_number: str | None = None
    application_date: str | None = None
    permit_type: str | None = None


@dataclass(frozen=True)
class DocumentProfile:
    schema: type[BaseModel]
    expected: tuple[str, ...]
    labels: dict[str, str]
    instructions: str
    resolve_unique_evidence: bool = True

    def required(self, document_type: str) -> tuple[str, ...]:
        return REQUIRED_FIELDS[document_type]


COMMON_DATES = {"issue_date": r"^(?:issue date\b|date issued\b|belge tarihi|duzenlenme tarihi)",
                "expiry_date": r"\bexpiry date\b|\bexpiration date\b|son gecerlilik tarihi"}
INSURANCE_LABELS = {
    **COMMON_DATES,
    "coverage_period": r"baslangic.*bitis.*tarih|\b(?:coverage|travel) period\b",
    "issue_date": r"police (?:onay|tanzim|duzenleme) tarihi|policy (?:issue|approval) date|\bissue date\b",
    "document_issue_date": r"duzenleme tarihi|tanzim tarihi|document issuance date",
    "coverage_start_date": r"(?:sigorta|teminat|police) baslangic tarihi|coverage start date",
    "coverage_end_date": r"(?:sigorta|teminat|police) bitis tarihi|coverage end date",
    "policy_number": r"\b(?:police\s+(?:no(?:su)?|numarasi)|policy\s+(?:number|no))\b",
}
REGISTRY = {
    "student_enrollment_certificate": DocumentProfile(EnrollmentExtraction, ("enrollment_status",), COMMON_DATES,
        "Extract enrollment facts only. Student identifier means student_id. Do not extract insurance fields.", False),
    "health_insurance_certificate": DocumentProfile(InsuranceExtraction, ("coverage_start_date", "coverage_end_date"),
        INSURANCE_LABELS, """insurer means insurance provider (normalized as institution too).
SAĞLIK SİGORTASI POLİÇESİ is a health policy. POLİÇE ONAY TARİHİ is policy issue date;
holder_name is the insured person's Sigortalı Adı Soyadı (possibly split across lines).
insurer is the underwriting Sigorta Şirketi, not the selling agent/acente.
DÜZENLEME TARİHİ is document issuance, a fallback only without a policy issue label.
SEYAHAT BAŞLANGIÇ-BİTİŞ TARİHİ is coverage start/end, not issue date.
Preserve coverage dates and explicit expiry separately. Never use tax/identity numbers as policy IDs."""),
    "ikamet_application_form": DocumentProfile(ApplicationExtraction, ("passport_number", "permit_type"), {
        "application_date": r"application date|date of application|basvuru tarihi",
        "application_number": r"application (?:number|no)|basvuru (?:numarasi|no)",
        "passport_number": r"passport (?:number|no)|pasaport (?:numarasi|no)",
    }, "Extract applicant and application fields only. A blank form is incomplete, not an approved permit."),
    "unknown": DocumentProfile(ExtractionMetadata, (), {}, "No supported type established; do not invent fields."),
}

INDICATORS = {
    "student_enrollment_certificate": r"student enrollment certificate|ogrenci belgesi|enrollment confirmation|certificate of (?:enrolment|registration)",
    "health_insurance_certificate": r"saglik sigorta(?:si)? policesi|health insurance (?:certificate|policy)",
    "ikamet_application_form": r"(?:residence permit|ikamet).*(?:application|basvuru).*(?:form|belgesi)|e-ikamet basvuru",
}
CLASSIFICATION_SIGNALS = {
    "student_enrollment_certificate": (r"student|ogrenci|enrol|registr|learner", r"university|universite|school|campus|education"),
    "health_insurance_certificate": (r"insurance|sigorta|policy|police", r"health|saglik|coverage|medical|teminat"),
    "ikamet_application_form": (r"residence|ikamet|permit|oturum", r"application|basvuru|form|request"),
}

# Classification concerns document identity, not extraction fields or legal rules.
IDENTITY_PURPOSES = {
    "student_enrollment_certificate": r"(?:this (?:certificate|document)|hereby).*(?:certif|confirm|attest).*(?:enrol|registr|student)",
    "health_insurance_certificate": r"this (?:insurance )?policy.*(?:provides|covers|insures|coverage)|isbu police.*(?:teminat|sigorta)",
    "ikamet_application_form": r"this form.*(?:shows|confirms|certifies|records).*(?:residence|permit|application)|bu form.*(?:ikamet|basvuru)",
}
IDENTITY_FIELDS = {
    "student_enrollment_certificate": {
        "student_id": r"student (?:id|identifier|number)|ogrenci (?:no|numarasi)",
        "enrollment": r"enrollment status|registration status|ogrencilik durumu",
        "institution": r"university|institution|universite",
    },
    "health_insurance_certificate": {
        "policy_id": INSURANCE_LABELS["policy_number"],
        "insured": r"policy holder|insured(?: name)?|sigortali",
        "coverage": r"coverage (?:start|end|period)|seyahat baslangic.*bitis tarihi",
    },
    "ikamet_application_form": {
        "application_id": r"application (?:number|no)|basvuru (?:no|numarasi)",
        "application_date": r"application date|basvuru tarihi",
        "passport": r"passport (?:number|no)|pasaport (?:no|numarasi)",
    },
}


def indicator_types(normalized_text: str) -> list[str]:
    return [kind for kind, pattern in INDICATORS.items() if re.search(pattern, normalized_text)]


def extraction_payload(payload: dict, document_type: str) -> dict:
    """Accept legacy normalized aliases, then validate only applicable schema fields."""
    data = dict(payload)
    if document_type == "health_insurance_certificate":
        data.setdefault("insurer", data.get("institution"))
    if document_type == "ikamet_application_form":
        data.setdefault("applicant_name", data.get("holder_name"))
    profile = REGISTRY[document_type]
    allowed = set(profile.schema.model_fields)
    data = {key: value for key, value in data.items() if key in allowed}
    # Shared normalized schema performs strict date type/calendar checks after grounding.
    validated = profile.schema.model_validate(data).model_dump()
    if document_type == "health_insurance_certificate":
        validated["institution"] = validated["insurer"]
    if document_type == "ikamet_application_form":
        validated["holder_name"] = validated["applicant_name"]
    validated["document_type"] = document_type
    relevant = allowed | {"institution", "holder_name"}
    validated["ambiguous_fields"] = [field for field in validated["ambiguous_fields"] if field in relevant]
    for first, second in (("insurer", "institution"), ("applicant_name", "holder_name")):
        if first in allowed and (first in validated["ambiguous_fields"] or second in validated["ambiguous_fields"]):
            validated["ambiguous_fields"] = list(dict.fromkeys(validated["ambiguous_fields"] + [first, second]))
    validated["source_snippets"] = {key: value for key, value in validated["source_snippets"].items() if key in relevant}
    return validated
