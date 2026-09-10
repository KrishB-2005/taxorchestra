"""Agent 1 — Extraction: source documents to typed records.

Two strategies behind one call:

  **Text-layer** (default, free, deterministic). Pull the PDF's text and read
  the labelled box values with anchored patterns. Works on any document whose
  text layer is intact and whose boxes are labelled the conventional way — the
  bundled samples, and a good share of employer-issued PDFs.

  **Model** (`--provider anthropic`). Hand Claude the PDF itself and parse the
  response straight into the W2 / 1099 schema. This is what handles scans,
  unusual layouts and anything the patterns miss.

The text-layer strategy runs first and the model is the fallback, not the
default, because the cheapest correct answer should win. `confidence` records
which one produced the record so a reviewer can weigh it.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pypdf import PdfReader

from taxorchestra.llm.client import LLMClient, LLMError
from taxorchestra.models import (
    W2,
    DocumentKind,
    ExtractionResult,
    Form1099DIV,
    Form1099INT,
    Form1099NEC,
    SourceDocument,
)

# Amounts print with thousands separators and an optional currency symbol.
_MONEY = r"\$?\s*([0-9][0-9,]*\.?[0-9]*)"


def _money(match: re.Match | None) -> Decimal:
    if match is None:
        return Decimal("0")
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return Decimal("0")


def _find(pattern: str, text: str) -> re.Match | None:
    return re.search(pattern, text, re.IGNORECASE)


def _text_of(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


SYSTEM_PROMPT = """\
You read United States tax source documents and return their contents as \
structured data.

Report the amounts exactly as printed on the document. Do not compute, round or \
reconcile anything — if box 3 disagrees with box 1, report both as printed; \
downstream validation exists to catch that. Amounts are plain decimal numbers \
with no currency symbol or thousands separator. Use 0 for a box that is blank \
or absent.
"""


class ExtractionAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def extract(self, path: str | Path) -> ExtractionResult:
        source = Path(path)
        text = _text_of(source)
        kind = self._classify(text)

        record = self._from_text(kind, text)
        if record is not None:
            return ExtractionResult(
                document=record,
                source_path=str(source),
                confidence=0.95,
                notes="read from the PDF text layer",
            )

        if self.llm is None:
            raise ValueError(
                f"{source.name}: the text layer did not yield a {kind.value} and no "
                "model provider is configured — rerun with --provider anthropic"
            )

        return self._from_model(kind, source)

    def extract_all(self, paths: list[str | Path]) -> list[ExtractionResult]:
        return [self.extract(p) for p in paths]

    # -- classification --------------------------------------------------

    @staticmethod
    def _classify(text: str) -> DocumentKind:
        lowered = text.lower()
        if "1099-nec" in lowered or "nonemployee compensation" in lowered:
            return DocumentKind.NEC_1099
        if "1099-div" in lowered or "ordinary dividends" in lowered:
            return DocumentKind.DIV_1099
        if "1099-int" in lowered or "interest income" in lowered:
            return DocumentKind.INT_1099
        return DocumentKind.W2

    # -- text-layer strategy ---------------------------------------------

    def _from_text(self, kind: DocumentKind, text: str) -> SourceDocument | None:
        try:
            if kind is DocumentKind.W2:
                return self._w2_from_text(text)
            if kind is DocumentKind.INT_1099:
                return self._int_from_text(text)
            if kind is DocumentKind.DIV_1099:
                return self._div_from_text(text)
            if kind is DocumentKind.NEC_1099:
                return self._nec_from_text(text)
        except (ValueError, InvalidOperation):
            return None
        return None

    @staticmethod
    def _w2_from_text(text: str) -> W2 | None:
        wages = _find(rf"box\s*1\b[^\n:]*:\s*{_MONEY}", text)
        withheld = _find(rf"box\s*2\b[^\n:]*:\s*{_MONEY}", text)
        if wages is None or withheld is None:
            return None
        employer = _find(r"employer:\s*(.+)", text)
        ein = _find(r"employer\s*ein:\s*([\d-]+)", text)
        ssn = _find(r"employee\s*ssn:\s*([\d-]+)", text)
        if employer is None or ein is None or ssn is None:
            return None
        return W2(
            employer_name=employer.group(1).strip(),
            employer_ein=ein.group(1),
            employee_ssn=ssn.group(1),
            wages=_money(wages),
            federal_income_tax_withheld=_money(withheld),
            social_security_wages=_money(_find(rf"box\s*3\b[^\n:]*:\s*{_MONEY}", text)),
            social_security_tax_withheld=_money(
                _find(rf"box\s*4\b[^\n:]*:\s*{_MONEY}", text)
            ),
            medicare_wages=_money(_find(rf"box\s*5\b[^\n:]*:\s*{_MONEY}", text)),
            medicare_tax_withheld=_money(_find(rf"box\s*6\b[^\n:]*:\s*{_MONEY}", text)),
        )

    @staticmethod
    def _payer(text: str) -> tuple[str, str] | None:
        name = _find(r"payer:\s*(.+)", text)
        tin = _find(r"payer\s*tin:\s*([\d-]+)", text)
        if name is None or tin is None:
            return None
        return name.group(1).strip(), tin.group(1)

    def _int_from_text(self, text: str) -> Form1099INT | None:
        payer = self._payer(text)
        box1 = _find(rf"box\s*1\b[^\n:]*:\s*{_MONEY}", text)
        if payer is None or box1 is None:
            return None
        return Form1099INT(
            payer_name=payer[0],
            payer_tin=payer[1],
            interest_income=_money(box1),
            federal_income_tax_withheld=_money(
                _find(rf"box\s*4\b[^\n:]*:\s*{_MONEY}", text)
            ),
            tax_exempt_interest=_money(_find(rf"box\s*8\b[^\n:]*:\s*{_MONEY}", text)),
        )

    def _div_from_text(self, text: str) -> Form1099DIV | None:
        payer = self._payer(text)
        box1a = _find(rf"box\s*1a\b[^\n:]*:\s*{_MONEY}", text)
        if payer is None or box1a is None:
            return None
        return Form1099DIV(
            payer_name=payer[0],
            payer_tin=payer[1],
            ordinary_dividends=_money(box1a),
            qualified_dividends=_money(_find(rf"box\s*1b\b[^\n:]*:\s*{_MONEY}", text)),
            federal_income_tax_withheld=_money(
                _find(rf"box\s*4\b[^\n:]*:\s*{_MONEY}", text)
            ),
        )

    def _nec_from_text(self, text: str) -> Form1099NEC | None:
        payer = self._payer(text)
        box1 = _find(rf"box\s*1\b[^\n:]*:\s*{_MONEY}", text)
        if payer is None or box1 is None:
            return None
        return Form1099NEC(
            payer_name=payer[0],
            payer_tin=payer[1],
            nonemployee_compensation=_money(box1),
            federal_income_tax_withheld=_money(
                _find(rf"box\s*4\b[^\n:]*:\s*{_MONEY}", text)
            ),
        )

    # -- model strategy --------------------------------------------------

    _SCHEMAS: dict[DocumentKind, type] = {
        DocumentKind.W2: W2,
        DocumentKind.INT_1099: Form1099INT,
        DocumentKind.DIV_1099: Form1099DIV,
        DocumentKind.NEC_1099: Form1099NEC,
    }

    def _from_model(self, kind: DocumentKind, source: Path) -> ExtractionResult:
        schema = self._SCHEMAS[kind]
        assert self.llm is not None
        try:
            record = self.llm.parse(
                system=SYSTEM_PROMPT,
                user=(
                    f"This is a {kind.value.upper()}. Return its contents as "
                    "structured data."
                ),
                schema=schema,
                pdf=source.read_bytes(),
            )
        except LLMError as exc:
            raise ValueError(f"{source.name}: extraction failed — {exc}") from exc

        return ExtractionResult(
            document=record,
            source_path=str(source),
            confidence=0.85,
            notes=f"read by {self.llm.name}; text layer did not match",
        )
