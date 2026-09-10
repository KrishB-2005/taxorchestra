"""Agent 4 — Orchestrator: run the pipeline and write the return.

Sequence:

    extract → compute → validate → map → render → fill

Extraction and mapping are independent, so mapping can be warmed from cache
while documents are read. Validation runs *before* anything is written: a
return that fails an arithmetic identity is not filed, it is reported.

The renderer is the last mile and the fiddliest part — a checkbox is not set
with `True` but with the name of one of its appearance streams, and that name
differs per field.
"""

from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path

from taxorchestra.agents.extraction import ExtractionAgent
from taxorchestra.agents.mapping import FormMappingAgent
from taxorchestra.agents.validation import ValidationAgent
from taxorchestra.forms import acroform
from taxorchestra.models import (
    FieldCatalog,
    FieldType,
    FilingResult,
    FilingStatus,
    Form1040Return,
    SourceDocument,
    Taxpayer,
)
from taxorchestra.returns import compute_return

_STATUS_KEY: dict[FilingStatus, str] = {
    FilingStatus.SINGLE: "filing_status_single",
    FilingStatus.MARRIED_JOINTLY: "filing_status_married_jointly",
    FilingStatus.MARRIED_SEPARATELY: "filing_status_married_separately",
    FilingStatus.HEAD_OF_HOUSEHOLD: "filing_status_head_of_household",
    FilingStatus.QUALIFYING_SURVIVING_SPOUSE: "filing_status_qualifying_surviving_spouse",
}


class Orchestrator:
    def __init__(
        self,
        extraction: ExtractionAgent,
        mapping: FormMappingAgent,
        validation: ValidationAgent,
    ) -> None:
        self.extraction = extraction
        self.mapping = mapping
        self.validation = validation

    def file_return(
        self,
        *,
        taxpayer: Taxpayer,
        document_paths: list[str | Path],
        template_pdf: str | Path,
        output_pdf: str | Path,
        use_cache: bool = True,
        allow_invalid: bool = False,
    ) -> FilingResult:
        extracted = self.extraction.extract_all(document_paths)
        return self.file_documents(
            taxpayer=taxpayer,
            documents=[e.document for e in extracted],
            template_pdf=template_pdf,
            output_pdf=output_pdf,
            use_cache=use_cache,
            allow_invalid=allow_invalid,
        )

    def file_documents(
        self,
        *,
        taxpayer: Taxpayer,
        documents: list[SourceDocument],
        template_pdf: str | Path,
        output_pdf: str | Path,
        use_cache: bool = True,
        allow_invalid: bool = False,
    ) -> FilingResult:
        """Same pipeline, for callers that already hold extracted records.

        The interview collects documents by asking rather than by reading files,
        so it joins here instead of at `file_return`.
        """
        started = time.perf_counter()

        return_ = compute_return(taxpayer, documents)
        report = self.validation.validate(return_, documents)

        if not report.ok and not allow_invalid:
            detail = "; ".join(i.message for i in report.errors)
            raise ValueError(f"refusing to write an invalid return: {detail}")

        catalog = self.mapping.build_catalog(str(template_pdf), use_cache=use_cache)
        values = self.render(return_, catalog, str(template_pdf))

        written = acroform.fill(template_pdf, output_pdf, values)

        return FilingResult(
            tax_year=return_.tax_year,
            **{"return": return_},
            validation=report,
            output_pdf=str(output_pdf),
            fields_written=written,
            fields_attempted=len(values),
            elapsed_seconds=round(time.perf_counter() - started, 4),
            cache_hit=self.mapping.stats.cache_hit,
        )

    # -- rendering -------------------------------------------------------

    def render(
        self,
        return_: Form1040Return,
        catalog: FieldCatalog,
        template_pdf: str,
    ) -> dict[str, str]:
        """Map computed values onto AcroForm names, formatted per field type."""
        by_key = catalog.by_key()
        on_states = {
            w.acro_name: w.on_states for w in acroform.load_widgets(template_pdf)
        }
        values: dict[str, str] = {}

        def put(semantic_key: str, raw: object) -> None:
            mapping = by_key.get(semantic_key)
            if mapping is None:
                return  # unresolved field: leave the box empty rather than guess
            rendered = self._format(mapping.field_type, raw, on_states.get(mapping.acro_name, ()))
            if rendered is not None:
                values[mapping.acro_name] = rendered

        taxpayer = return_.taxpayer
        put("taxpayer_first_name", taxpayer.first_name)
        put("taxpayer_last_name", taxpayer.last_name)
        put("taxpayer_ssn", taxpayer.ssn)
        put("address_street", taxpayer.address.street)
        put("address_apartment", taxpayer.address.apartment)
        put("address_city", taxpayer.address.city)
        put("address_state", taxpayer.address.state)
        put("address_zip", taxpayer.address.zip_code)

        if taxpayer.spouse_first_name:
            put("spouse_first_name", taxpayer.spouse_first_name)
        if taxpayer.spouse_ssn:
            put("spouse_ssn", taxpayer.spouse_ssn)

        put(_STATUS_KEY[taxpayer.filing_status], True)

        for line in return_.lines:
            put(line.semantic_key, line.value)

        return values

    @staticmethod
    def _format(
        field_type: FieldType,
        raw: object,
        on_states: tuple[str, ...],
    ) -> str | None:
        if raw is None:
            return None

        if field_type is FieldType.CHECKBOX:
            if not raw:
                return None
            # A checkbox takes the name of an appearance stream ("/1", "/Yes"),
            # not a boolean. Which name is per-field, so read it off the widget.
            return on_states[0] if on_states else "/Yes"

        if field_type is FieldType.SSN:
            # The SSN boxes are 9-cell comb fields with /MaxLen 9, so they take
            # bare digits. pypdf will happily write "900-12-3456" into them and
            # the result renders as overflow; pdf-lib refuses outright, which is
            # how this surfaced.
            digits = "".join(ch for ch in str(raw) if ch.isdigit())
            return digits if len(digits) == 9 else None

        if field_type is FieldType.MONEY:
            amount = raw if isinstance(raw, Decimal) else Decimal(str(raw))
            # The 1040 is filed in whole dollars. A zero line is left blank —
            # writing "0" everywhere makes a filled form unreadable.
            if amount == 0:
                return None
            return f"{amount.quantize(Decimal('1')):,}"

        text = str(raw).strip()
        return text or None
