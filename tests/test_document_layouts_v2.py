"""Sanitized replicas of layout classes; never private student values."""
from datetime import date
from pathlib import Path
import pytest
from unittest.mock import patch
from src.document_processing.extractor import TextPage
from src.document_processing.processor import complete_source_projection,cached_process_document
from src.document_processing.evidence import source_candidates,field_candidates
from src.document_processing.schemas import HealthInsurance,UnknownDocument


def test_turkish_date_labels_establish_day_first_without_relaxing_unknown_english_dates():
    c,_=source_candidates([TextPage(1,"Baslangic Tarihi:01/02/2030\nBitis Tarihi:01/02/2031")])
    assert field_candidates("coverage_start_date",HealthInsurance,c)[0]["normalized_value"]=="2030-02-01"
    c,_=source_candidates([TextPage(1,"Coverage start date: 03/04/2030")])
    assert field_candidates("coverage_start_date",HealthInsurance,c)[0]["normalized_value"] is None


def test_labelled_policy_identifier_survives_other_labelled_columns():
    c,_=source_candidates([TextPage(1,"Musteri No:CUST123  Police No:POL-2030-004  Zeyil No:0  Yenileme No:0")])
    assert field_candidates("policy_number",HealthInsurance,c)[0]["normalized_value"]=="POL-2030-004"
    c,_=source_candidates([TextPage(1,"Policy number: ABC123 DEF456")])
    assert field_candidates("policy_number",HealthInsurance,c)[0]["normalized_value"] is None


def test_bilingual_ocr_reordered_names_and_dates_produce_complete_source_form():
    text=("Declaration for Short Term Residence Permit Registration Form\n"
        "Kayit Numarasi (Registration\n2030-11-1234567\nNumber)\n"
        "Talep Edilen Izin Baslangic Tarihi\n18/9/2030\nTalep Edilen Bitis Tarihi\n18/9/2031\n"
        "(Requested Start Date)\n(Requested End Date)\n"
        "Soyadi\nEXAMPLE\nDiger Uyrugu\nSumame\nOther Citizenship\n"
        "Adi\nTAYLOR\nDogumdaki Uyrugu\nName\nNationality in Born\nPassport number: X12345678")
    info=complete_source_projection([TextPage(1,text,ocr_used=True)],"ikamet_application_form")
    assert info is not None and info.applicant_name=="TAYLOR EXAMPLE"
    assert info.requested_start_date==date(2030,9,18) and info.requested_end_date==date(2031,9,18)
    assert info.application_number=="2030-11-1234567"
    assert info.source_pages["applicant_name"]==1


def test_native_two_column_period_uses_roles_not_birth_dates():
    text=("Declaration for Short Term Residence Permit Registration Form\n"
        "Talep Edilen Izin Baslangic Tarihi  Talep Edilen Bitis Tarihi  18 / 9 / 2030  18 / 9 / 2031\n"
        "Birth date: 01/01/2000\nApplicant: Taylor Example\nPassport number: X12345678")
    info=complete_source_projection([TextPage(1,text)],"ikamet_application_form")
    assert info is not None and info.requested_end_date==date(2031,9,18)


def test_provider_prose_and_association_are_not_company_names():
    text=("Health Insurance Certificate\nInsurer: Example Insurance Ltd.\n"
        "Primlerden tazminat tutari dusulerek sigorta ettirene iade edilir.\n"
        "Turkiye Sigorta Birligi")
    c,_=source_candidates([TextPage(1,text)])
    options=field_candidates("insurer",HealthInsurance,c)
    assert {item["normalized_value"] for item in options}=={"Example Insurance Ltd."}


def test_cache_reuses_observations_and_invalidates_changed_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr("config.DOCUMENT_CACHE_ENABLED",True)
    monkeypatch.setattr("config.PROJECT_ROOT",tmp_path)
    path=tmp_path/"synthetic.txt";path.write_text("one",encoding="utf-8")
    with patch("src.document_processing.processor.process_document",side_effect=lambda *a,**k:UnknownDocument()) as processor:
        cached_process_document(path,mode="local-only")
        second=cached_process_document(path,mode="local-only")
        assert second.extraction_diagnostics["cache_hit"] is True and processor.call_count==1
        path.write_text("two",encoding="utf-8")
        cached_process_document(path,mode="local-only")
        assert processor.call_count==2


def test_turkish_student_status_certificate_heading_does_not_depend_on_model_guess():
    from src.document_processing.processor import checked_classification
    pages=[TextPage(1,"OGRENCI DURUM BELGESI\nOgrenci No: 12345678\nOgrencilik Durumu: Aktif")]
    assert checked_classification("unknown",pages)=="student_enrollment_certificate"
    assert checked_classification("unknown",[TextPage(1,"My application refers to an ogrenci durum belgesi")])=="unknown"


def test_explicit_turkish_person_label_recovers_null_student_model_value():
    from src.document_processing.evidence import ground_fields
    from src.document_processing.schemas import StudentCertificate
    pages=[TextPage(1,"OGRENCI DURUM BELGESI\nAdi Soyadi: Taylor Example\nAnne Adi: Other Parent")]
    candidates,issues=source_candidates(pages)
    payload={"holder_name":None}
    ground_fields(payload,StudentCertificate,candidates,issues,pages)
    assert payload["holder_name"]=="Taylor Example"
    assert payload["source_pages"]["holder_name"]==1
    assert payload["source_snippets"]["holder_name"]=="Adi Soyadi: Taylor Example"
    assert any("holder_name" in note for note in payload["correction_notes"])


def test_mrz_nationality_ocr_correction_requires_structural_and_country_validation():
    from src.document_processing.evidence import mrz_candidates,ground_fields
    from src.document_processing.schemas import PassportIdentity
    def digit(text):
        values=[0 if c=="<" else ord(c)-55 if c.isalpha() else int(c) for c in text]
        return str(sum(v*(7,3,1)[i%3] for i,v in enumerate(values))%10)
    a="P<UTOEXAMPLE<<TAYLOR".ljust(44,"<")
    number="P1234567<";dob="000101";expiry="350101";personal="<"*14
    b=number+digit(number)+"1TA"+dob+digit(dob)+"F"+expiry+digit(expiry)+personal+digit(personal)
    b+=digit(b[:10]+b[13:20]+b[21:43])
    pages=[TextPage(1,a+"\n"+b)]
    candidates=mrz_candidates(pages)
    assert next(c for c in candidates if c["field"]=="passport_number")["normalized_value"]=="P1234567"
    assert next(c for c in candidates if c["field"]=="nationality")["normalized_value"] == "ITA"
    payload={};issues=[]
    ground_fields(payload,PassportIdentity,candidates,issues,pages)
    assert payload["nationality"] == "ITA" and "nationality" not in payload["ambiguous_fields"]
    assert any("MRZ source code" in note for note in payload["correction_notes"])
    assert not issues



def _checked_passport_mrz(nationality='1TA'):
    def digit(text):
        return str(sum((0 if c=='<' else ord(c)-55 if c.isalpha() else int(c))*(7,3,1)[i%3] for i,c in enumerate(text))%10)
    top='P<ITAEXAMPLE<<TAYLOR'.ljust(44,'<')
    number='P1234567<';dob='000101';expiry='350101';personal='<'*14
    bottom=number+digit(number)+nationality+dob+digit(dob)+'F'+expiry+digit(expiry)+personal+digit(personal)
    return top,bottom+digit(bottom[:10]+bottom[13:20]+bottom[21:43])


def test_passport_controlled_correction_retains_exact_original_mrz_page_and_note():
    from src.document_processing.evidence import mrz_candidates,ground_fields,ISO_ALPHA3_CODES
    from src.document_processing.schemas import PassportIdentity
    top,bottom=_checked_passport_mrz()
    pages=[TextPage(3,top+'\n'+bottom)]
    candidates=mrz_candidates(pages)
    item=next(c for c in candidates if c['field']=='nationality')
    assert item['original_value']=='1TA' and item['normalized_value']=='ITA' and item['page_number']==3
    assert item['snippet']==top+'\n'+bottom and item['validated_mrz']
    assert len(ISO_ALPHA3_CODES)==249
    payload={};issues=[];ground_fields(payload,PassportIdentity,candidates,issues,pages)
    assert payload['source_pages']['nationality']==3 and payload['source_snippets']['nationality']==top+'\n'+bottom
    assert any('1TA' in note and 'ITA' in note for note in payload['correction_notes'])


@pytest.mark.parametrize('damage',['passport_check','expiry_check','composite_check','short_line'])
def test_mrz_nationality_recovery_never_repairs_identity_or_date_check_failures(damage):
    from src.document_processing.evidence import mrz_candidates
    top,bottom=_checked_passport_mrz()
    if damage=='short_line':bottom=bottom[:-1]
    else:
        position={'passport_check':9,'expiry_check':27,'composite_check':43}[damage]
        bottom=bottom[:position]+str((int(bottom[position])+1)%10)+bottom[position+1:]
    assert mrz_candidates([TextPage(1,top+'\n'+bottom)])==[]


@pytest.mark.parametrize('code',['ZZZ','1TX','0XX'])
def test_mrz_unknown_country_codes_are_not_invented(code):
    from src.document_processing.evidence import mrz_candidates
    top,bottom=_checked_passport_mrz(code)
    candidate=next(c for c in mrz_candidates([TextPage(1,top+'\n'+bottom)]) if c['field']=='nationality')
    assert candidate['normalized_value'] is None and not candidate.get('mrz_correction')


def test_valid_conflicting_nationality_evidence_still_requires_review():
    from src.document_processing.evidence import ground_fields
    from src.document_processing.schemas import PassportIdentity
    top,bottom=_checked_passport_mrz('ITA')
    pages=[TextPage(1,'Passport identity page\nNationality: France\n'+top+'\n'+bottom)]
    candidates,issues=source_candidates(pages);payload={}
    ground_fields(payload,PassportIdentity,candidates,issues,pages)
    assert payload['nationality'] is None and 'nationality' in payload['ambiguous_fields']
