# `src/tools/extract_document.py` — documents to proposed facts

## What it does
It reads an uploaded file (PDF, TXT, or a photo in JPG/PNG) and returns what each document in it *shows*:
values backed by an exact quote and page, mapped to the procedure's fact names with `procedure.json`
`documents[].extracts`. One file can hold several documents (a passport page, a certificate and a policy
scanned together): each becomes its own document in the Twin.

## The steps, in order
1. **Text** (`src/document_processing/extractor.py`). The text layer of each page; OCR (RapidOCR, on the CPU)
   when the page is a scan: no text, or only a watermark/header over a full-page image. A photo is turned into
   a one-page PDF first. Each page remembers whether its text came from OCR (`TextPage.ocr`).
2. **Which pages are which document** (`src/document_processing/segment.py`, code only).
   - A page **opens** a document when it has a machine-readable zone of that document's `mrz` kind (checked by its
     digits), or one of its `titles` in its top lines. A line with a value ("Passport number: X1") is not a title.
   - Every other page **continues** the document before it (page 2 of a policy, the instructions of a form).
   - A document without title is named by its `keywords`; the model only breaks a tie, choosing among the tied types.
3. **Reading each document.**
   - The seven milestone-1 kinds keep their **typed processor** (typed schema, label evidence, MRZ, date roles:
     the regression oracle). A file that is one typed document keeps the processor's own classification, as before.
   - Every other pack document uses the **generic reader** (`generic.py`): MRZ values first (code, check digits),
     then the model proposes `{field, value, page, quote}` batch by batch over **all pages** (before, only 3,400
     characters were read), asking only for the fields still missing.
   - **Second pass for typed documents.** Values the typed processor left out (not those it found ambiguous) are
     asked to the generic reader; they are stored as values **the user must confirm**.
4. **Pictures** (`vision.py`, vision model `qwen3-vl:2b-instruct`). Only for pages that are images:
   - nothing names the page (a border stamp has no title; its OCR text is "ISTANBUL 26.02.22"): the vision model
     chooses one of the pack's documents; every value of a document named this way waits for confirmation;
   - values still missing: the vision model reads them, and code compares each one with the OCR text of the page.
     Only a date or a number found in the OCR text counts at once; anything else waits for confirmation.
5. The registered tool stores the observations in the Twin (`service.store_extraction`): facts with
   `provenance=document`, confirmed unless the pack, the second pass or the vision step says otherwise.

## Inputs and outputs
- **Input:** a file path, the `Pack`, optional `doc_type` (the user's choice: the whole file is that document),
  optional `only_pages` (read one part of a split file again), and a backend (model, vision model).
- **Output:** `ExtractionResult{observations: [DocumentObservation], error}`. Each observation now also says
  `pages` (where it is in the file), `found_by` (mrz, title, keywords, model, vision, user) and `notes`.

## Why it is built this way
- **Measured before and after** with `python -m eval.document_eval` on 13 files (synthetic text PDFs, scans,
  a photo, files with two documents, and the public passport and stamp specimens), see `docs/eval_documents.md`:
  documents found 8/17 → 16/17 (17/17 with the vision model), values read correctly 46% → 75% (89% with the
  second pass and the vision model, 8 of them to confirm), values believed wrong 0.
- **The model never decides alone.** Titles, MRZ check digits and keywords name the documents; quotes, pages and
  types are checked by code; what code cannot check is shown to the user to confirm.
- **New procedures need no Python:** a pack lists `titles`, `keywords` and `mrz` for its documents.

## How to test it
```
python -m pytest -q tests/test_extract_document.py tests/test_segment.py tests/test_vision_and_crawl.py \
    tests/test_processor.py tests/test_extractor.py
python -m eval.document_eval --label "my change" --out docs/eval_documents.md    # live, needs Ollama
```

## Known limits
- **Speed on a 4 GB GPU.** About 20–40 s per document; the vision model adds about 35 s for a stamp and up to
  2–3 minutes for a full-page photo, so it runs only on pictures, only when something is missing.
- **Two documents of the same type in one file** (two passports) are read as one; their values then disagree and
  are shown as a conflict.
- **Names on photos.** OCR often splits a name over two lines; such names are proposed for confirmation.
- **Replies of the vision model are not quoted text**: a value it reads that the OCR text does not show is always
  marked "read from the image, not found in its OCR text".
