"""End-to-end accuracy and cache-latency measurement.

Accuracy here means: reopen the written PDF and check that every value we
intended to write is actually stored in the field we intended to write it to.
Counting successful `update_page_form_field_values` calls would be measuring
our own optimism — a field can be written and still not land, because the name
did not resolve to a widget or the value was silently coerced.

The cold/warm split is the honest form of the caching claim: cold pays for
field resolution, warm reads the catalog back and only does the arithmetic and
the fill.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from taxorchestra.agents.extraction import ExtractionAgent
from taxorchestra.agents.mapping import FormMappingAgent
from taxorchestra.agents.orchestrator import Orchestrator
from taxorchestra.agents.validation import ValidationAgent
from taxorchestra.cache.store import MemoryCache
from taxorchestra.forms import acroform
from taxorchestra.knowledge.store import BM25Store
from taxorchestra.llm.client import build_client
from taxorchestra.models import Address, FilingStatus, Taxpayer
from taxorchestra.samples import SAMPLE_SSN, write_sample_documents

BENCH_TAXPAYER = Taxpayer(
    first_name="Dana",
    last_name="Whitfield",
    ssn=SAMPLE_SSN,
    address=Address(
        street="418 Larkspur Lane",
        city="Silver Spring",
        state="MD",
        zip_code="20910",
    ),
    filing_status=FilingStatus.SINGLE,
)


def run_benchmark(
    *,
    template: str,
    provider: str | None = None,
    warm_runs: int = 5,
) -> dict:
    llm = build_client(provider)
    cache = MemoryCache()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        documents = [str(p) for p in write_sample_documents(tmp_path / "docs")]
        output = tmp_path / "f1040-filled.pdf"

        mapping = FormMappingAgent(llm, cache)
        orchestrator = Orchestrator(
            extraction=ExtractionAgent(llm),
            mapping=mapping,
            validation=ValidationAgent(BM25Store()),
        )

        # ---- cold: field catalog resolved from scratch ------------------
        cold_start = time.perf_counter()
        result = orchestrator.file_return(
            taxpayer=BENCH_TAXPAYER,
            document_paths=documents,
            template_pdf=template,
            output_pdf=output,
        )
        cold_seconds = time.perf_counter() - cold_start
        model_calls_cold = mapping.stats.llm_calls

        # ---- verify by reading the written PDF back ---------------------
        intended = orchestrator.render(
            result.return_,
            mapping.build_catalog(template),
            template,
        )
        stored = acroform.read_back(output)

        # A checkbox's stored value is the bare appearance-state name; the value
        # we hand pypdf carries the PDF name slash. Same value, two spellings.
        def norm(value: str | None) -> str | None:
            return value.lstrip("/") if value is not None else None

        correct = sum(
            1
            for name, value in intended.items()
            if norm(stored.get(name)) == norm(value)
        )
        mismatches = [
            {"field": name, "expected": value, "stored": stored.get(name)}
            for name, value in intended.items()
            if norm(stored.get(name)) != norm(value)
        ]

        # ---- warm: catalog served from cache ----------------------------
        warm_times: list[float] = []
        for _ in range(warm_runs):
            warm_mapping = FormMappingAgent(llm, cache)
            warm_orchestrator = Orchestrator(
                extraction=ExtractionAgent(llm),
                mapping=warm_mapping,
                validation=ValidationAgent(BM25Store()),
            )
            start = time.perf_counter()
            warm_orchestrator.file_return(
                taxpayer=BENCH_TAXPAYER,
                document_paths=documents,
                template_pdf=template,
                output_pdf=output,
            )
            warm_times.append(time.perf_counter() - start)

        # ---- catalog lookup alone, without the fill ---------------------
        lookup_times: list[float] = []
        for _ in range(warm_runs):
            agent = FormMappingAgent(llm, cache)
            start = time.perf_counter()
            agent.build_catalog(template)
            lookup_times.append(time.perf_counter() - start)

    return {
        "provider": llm.name,
        "model": getattr(llm, "model", None),
        "form": {
            "path": template,
            "tax_year": result.return_.tax_year,
            "total_fields": len(mapping.build_catalog(template).mappings),
        },
        "fill": {
            "fields_intended": len(intended),
            "fields_verified_in_pdf": correct,
            "accuracy": round(correct / len(intended), 4) if intended else 0.0,
            "mismatches": mismatches,
        },
        "validation": {
            "ok": result.validation.ok,
            "issues": len(result.validation.issues),
        },
        "latency_seconds": {
            "cold_end_to_end": round(cold_seconds, 4),
            "warm_end_to_end_mean": round(sum(warm_times) / len(warm_times), 4),
            "warm_end_to_end_min": round(min(warm_times), 4),
            "cached_catalog_lookup_mean": round(
                sum(lookup_times) / len(lookup_times), 4
            ),
            "cached_catalog_lookup_min": round(min(lookup_times), 4),
        },
        "model_calls": {
            "cold": model_calls_cold,
            "warm": 0,
        },
        "return_summary": {
            line.semantic_key: str(line.value)
            for line in result.return_.lines
            if line.value
        },
    }
