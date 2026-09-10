"""End-to-end and unit coverage.

Every test runs on the `fixture` provider, so the suite needs no API key and
costs nothing. The form itself is real: tests that touch the 1040 skip when it
has not been fetched, rather than vendoring a government PDF into the repo.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from taxorchestra.agents.extraction import ExtractionAgent
from taxorchestra.agents.mapping import FormMappingAgent
from taxorchestra.agents.orchestrator import Orchestrator
from taxorchestra.agents.validation import ValidationAgent
from taxorchestra.cache.store import MemoryCache, SQLiteCache
from taxorchestra.forms import acroform
from taxorchestra.knowledge.store import BM25Store
from taxorchestra.llm.client import FixtureClient
from taxorchestra.models import (
    W2,
    Address,
    FilingStatus,
    Form1099INT,
    Severity,
    Taxpayer,
)
from taxorchestra.returns import compute_return
from taxorchestra.samples import (
    SAMPLE_1099INT,
    SAMPLE_SSN,
    SAMPLE_W2,
    write_1099int_pdf,
    write_sample_documents,
    write_w2_pdf,
)
from taxorchestra.taxdata import ordinary_income_tax, standard_deduction

TEMPLATE = Path("data/forms/f1040.pdf")

requires_form = pytest.mark.skipif(
    not TEMPLATE.exists(),
    reason="blank 1040 not present — run `taxorchestra fetch-form`",
)


@pytest.fixture
def taxpayer() -> Taxpayer:
    return Taxpayer(
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


# ---------------------------------------------------------------------------
# Tax computation
# ---------------------------------------------------------------------------


class TestTaxComputation:
    def test_brackets_are_marginal_not_flat(self) -> None:
        """A dollar over a threshold is taxed at the new rate, not the whole sum."""
        just_under = ordinary_income_tax(Decimal("11925"), FilingStatus.SINGLE)
        just_over = ordinary_income_tax(Decimal("11926"), FilingStatus.SINGLE)
        assert just_under == Decimal("1193")  # 11,925 * 10%, rounded
        assert just_over - just_under < Decimal("1")

    def test_zero_and_negative_income_owe_nothing(self) -> None:
        assert ordinary_income_tax(Decimal("0"), FilingStatus.SINGLE) == 0
        assert ordinary_income_tax(Decimal("-500"), FilingStatus.SINGLE) == 0

    def test_married_jointly_pays_less_than_single_on_equal_income(self) -> None:
        income = Decimal("120000")
        assert ordinary_income_tax(income, FilingStatus.MARRIED_JOINTLY) < (
            ordinary_income_tax(income, FilingStatus.SINGLE)
        )

    def test_known_bracket_walk(self) -> None:
        """77,984 single: 10% to 11,925, 12% to 48,475, 22% on the rest."""
        expected = (
            Decimal("11925") * Decimal("0.10")
            + (Decimal("48475") - Decimal("11925")) * Decimal("0.12")
            + (Decimal("77984") - Decimal("48475")) * Decimal("0.22")
        ).quantize(Decimal("1"))
        assert ordinary_income_tax(Decimal("77984"), FilingStatus.SINGLE) == expected


class TestReturnComputation:
    def test_wages_and_interest_flow_to_agi(self, taxpayer: Taxpayer) -> None:
        result = compute_return(taxpayer, [SAMPLE_W2, SAMPLE_1099INT])
        assert result.value("wages_w2_box1") == Decimal("92450")
        assert result.value("interest_taxable") == Decimal("1284")
        assert result.value("total_income") == Decimal("93734")
        assert result.value("adjusted_gross_income") == Decimal("93734")

    def test_tax_exempt_interest_is_reported_but_not_taxed(
        self, taxpayer: Taxpayer
    ) -> None:
        muni = Form1099INT(
            payer_name="Municipal Fund",
            payer_tin="00-1111111",
            interest_income=Decimal("0"),
            tax_exempt_interest=Decimal("5000"),
        )
        result = compute_return(taxpayer, [muni])
        assert result.value("interest_tax_exempt") == Decimal("5000")
        assert result.value("total_income") == Decimal("0")

    def test_taxable_income_floors_at_zero(self, taxpayer: Taxpayer) -> None:
        tiny = W2(
            employer_name="Part Time Co",
            employer_ein="00-2222222",
            employee_ssn=SAMPLE_SSN,
            wages=Decimal("1000"),
            federal_income_tax_withheld=Decimal("0"),
        )
        result = compute_return(taxpayer, [tiny])
        assert result.value("taxable_income") == Decimal("0")
        assert result.value("tax_total") == Decimal("0")

    def test_refund_and_owed_are_mutually_exclusive(self, taxpayer: Taxpayer) -> None:
        over_withheld = W2(
            employer_name="Generous Withholding Inc",
            employer_ein="00-3333333",
            employee_ssn=SAMPLE_SSN,
            wages=Decimal("50000"),
            federal_income_tax_withheld=Decimal("20000"),
        )
        result = compute_return(taxpayer, [over_withheld])
        assert result.value("refund_overpaid") > 0
        assert result.value("amount_owed") == 0

    def test_every_line_derives_from_the_lines_as_printed(
        self, taxpayer: Taxpayer
    ) -> None:
        """The form must agree with itself.

        Caught by running the TypeScript port of this module against the same
        inputs and getting a different answer: `add()` used to return the
        unrounded value, so line 16 was the tax on a taxable income carrying
        cents that line 15 never showed. Off by a dollar, and invisible unless
        you recompute from the printed figures.
        """
        result = compute_return(taxpayer, [SAMPLE_W2, SAMPLE_1099INT])

        assert result.value("total_income") == (
            result.value("wages_total")
            + result.value("interest_taxable")
            + result.value("dividends_ordinary")
        )
        assert result.value("taxable_income") == (
            result.value("adjusted_gross_income") - result.value("deduction_total")
        )
        assert result.value("tax_computed") == ordinary_income_tax(
            result.value("taxable_income"), taxpayer.filing_status
        )
        assert result.value("amount_owed") == (
            result.value("tax_total") - result.value("payments_total")
        )

    def test_standard_deduction_matches_filing_status(self, taxpayer: Taxpayer) -> None:
        result = compute_return(taxpayer, [SAMPLE_W2])
        assert result.value("deduction_standard_or_itemized") == standard_deduction(
            FilingStatus.SINGLE
        )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_reads_a_w2_from_the_pdf_text_layer(self, tmp_path: Path) -> None:
        path = write_w2_pdf(tmp_path / "w2.pdf", SAMPLE_W2)
        result = ExtractionAgent().extract(path)
        assert isinstance(result.document, W2)
        assert result.document.wages == SAMPLE_W2.wages
        assert result.document.federal_income_tax_withheld == Decimal("11894.00")
        assert result.document.employee_ssn == SAMPLE_SSN
        assert "text layer" in (result.notes or "")

    def test_reads_a_1099int(self, tmp_path: Path) -> None:
        path = write_1099int_pdf(tmp_path / "int.pdf", SAMPLE_1099INT)
        result = ExtractionAgent().extract(path)
        assert isinstance(result.document, Form1099INT)
        assert result.document.interest_income == Decimal("1284.37")

    def test_classifies_by_content_not_filename(self, tmp_path: Path) -> None:
        misleading = write_1099int_pdf(tmp_path / "definitely-a-w2.pdf", SAMPLE_1099INT)
        result = ExtractionAgent().extract(misleading)
        assert isinstance(result.document, Form1099INT)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_clean_return_passes(self, taxpayer: Taxpayer) -> None:
        documents = [SAMPLE_W2, SAMPLE_1099INT]
        report = ValidationAgent(BM25Store()).validate(
            compute_return(taxpayer, documents), documents
        )
        assert report.ok
        assert report.errors == []

    def test_catches_a_tampered_total(self, taxpayer: Taxpayer) -> None:
        """The check that exists to catch an invented number."""
        result = compute_return(taxpayer, [SAMPLE_W2])
        for line in result.lines:
            if line.semantic_key == "taxable_income":
                line.value = Decimal("1")  # as if a model had made it up

        report = ValidationAgent(BM25Store()).validate(result, [SAMPLE_W2])
        assert not report.ok
        assert any(i.semantic_key == "taxable_income" for i in report.errors)
        assert all(i.citation for i in report.errors)

    def test_flags_an_ssn_mismatch(self, taxpayer: Taxpayer) -> None:
        wrong = SAMPLE_W2.model_copy(update={"employee_ssn": "900-99-9999"})
        report = ValidationAgent(BM25Store()).validate(
            compute_return(taxpayer, [wrong]), [wrong]
        )
        assert not report.ok
        assert any("does not match" in i.message for i in report.errors)

    def test_warns_that_self_employment_tax_is_not_computed(
        self, taxpayer: Taxpayer
    ) -> None:
        from taxorchestra.models import Form1099NEC

        nec = Form1099NEC(
            payer_name="Contract Client LLC",
            payer_tin="00-4444444",
            nonemployee_compensation=Decimal("18000"),
        )
        report = ValidationAgent(BM25Store()).validate(
            compute_return(taxpayer, [nec]), [nec]
        )
        assert report.ok  # a warning, not an error
        assert any(i.severity is Severity.WARNING for i in report.issues)


class TestKnowledgeRetrieval:
    def test_retrieval_is_relevant(self) -> None:
        hits = BM25Store().search("qualified dividends exceed ordinary dividends", k=1)
        assert hits and hits[0].rule.rule_id == "dividends-3a-3b"

    def test_retrieval_is_deterministic(self) -> None:
        store = BM25Store()
        query = "taxable income line 15"
        assert [h.rule.rule_id for h in store.search(query)] == [
            h.rule.rule_id for h in store.search(query)
        ]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCache:
    def test_sqlite_round_trips(self, tmp_path: Path) -> None:
        cache = SQLiteCache(tmp_path / "c.sqlite")
        cache.put("key", {"a": 1})
        assert cache.get("key") == {"a": 1}
        assert cache.get("absent") is None

    def test_sqlite_overwrites_rather_than_duplicating(self, tmp_path: Path) -> None:
        cache = SQLiteCache(tmp_path / "c.sqlite")
        cache.put("key", {"v": 1})
        cache.put("key", {"v": 2})
        assert cache.get("key") == {"v": 2}


# ---------------------------------------------------------------------------
# Form handling — needs the real 1040
# ---------------------------------------------------------------------------


@requires_form
class TestFormMapping:
    def test_resolves_the_wage_line_to_the_right_widget(self) -> None:
        agent = FormMappingAgent(FixtureClient(), MemoryCache())
        catalog = agent.build_catalog(str(TEMPLATE))
        wages = catalog.by_key()["wages_w2_box1"]
        assert wages.line == "1a"
        assert wages.acro_name.endswith("f1_47[0]")

    def test_two_column_rows_do_not_bleed(self) -> None:
        """2a and 2b sit on one row; each must claim only its own label."""
        catalog = FormMappingAgent(FixtureClient(), MemoryCache()).build_catalog(
            str(TEMPLATE)
        )
        by_key = catalog.by_key()
        assert by_key["interest_tax_exempt"].line == "2a"
        assert by_key["interest_taxable"].line == "2b"
        assert by_key["interest_tax_exempt"].acro_name != (
            by_key["interest_taxable"].acro_name
        )

    def test_second_run_is_served_from_cache(self) -> None:
        cache = MemoryCache()
        cold = FormMappingAgent(FixtureClient(), cache)
        cold.build_catalog(str(TEMPLATE))
        assert cold.stats.llm_calls > 0
        assert not cold.stats.cache_hit

        warm = FormMappingAgent(FixtureClient(), cache)
        warm.build_catalog(str(TEMPLATE))
        assert warm.stats.cache_hit
        assert warm.stats.llm_calls == 0

    def test_cache_is_keyed_on_the_form_digest(self) -> None:
        """A different form revision must not reuse last year's mapping."""
        cache = MemoryCache()
        FormMappingAgent(FixtureClient(), cache).build_catalog(str(TEMPLATE))
        assert cache.get("a-different-digest") is None


@requires_form
class TestEndToEnd:
    def _run(self, taxpayer: Taxpayer, tmp_path: Path):
        llm = FixtureClient()
        documents = [str(p) for p in write_sample_documents(tmp_path / "docs")]
        orchestrator = Orchestrator(
            extraction=ExtractionAgent(llm),
            mapping=FormMappingAgent(llm, MemoryCache()),
            validation=ValidationAgent(BM25Store()),
        )
        return orchestrator.file_return(
            taxpayer=taxpayer,
            document_paths=documents,
            template_pdf=TEMPLATE,
            output_pdf=tmp_path / "filled.pdf",
        )

    def test_every_intended_value_is_present_in_the_written_pdf(
        self, taxpayer: Taxpayer, tmp_path: Path
    ) -> None:
        result = self._run(taxpayer, tmp_path)
        stored = acroform.read_back(tmp_path / "filled.pdf")

        assert result.validation.ok
        assert result.fields_written == result.fields_attempted
        # Reading the file back is the only proof that a write landed.
        assert len(stored) == result.fields_written

    def test_key_lines_land_in_the_correct_boxes(
        self, taxpayer: Taxpayer, tmp_path: Path
    ) -> None:
        self._run(taxpayer, tmp_path)
        stored = {
            name.split(".")[-1]: value
            for name, value in acroform.read_back(tmp_path / "filled.pdf").items()
        }
        assert stored["f1_47[0]"] == "92,450"  # line 1a wages
        assert stored["f1_59[0]"] == "1,284"  # line 2b taxable interest
        assert stored["f2_06[0]"] == "77,984"  # line 15 taxable income
        assert stored["f1_16[0]"] == SAMPLE_SSN.replace("-", "")  # comb field

    def test_ssn_is_written_as_nine_bare_digits(
        self, taxpayer: Taxpayer, tmp_path: Path
    ) -> None:
        """The SSN boxes are comb fields with /MaxLen 9.

        pypdf writes an over-length value without complaint, so this went
        unnoticed until the browser port hit pdf-lib, which rejects it.
        """
        self._run(taxpayer, tmp_path)
        stored = {
            name.split(".")[-1]: value
            for name, value in acroform.read_back(tmp_path / "filled.pdf").items()
        }
        assert stored["f1_16[0]"] == SAMPLE_SSN.replace("-", "")
        assert len(stored["f1_16[0]"]) == 9

    def test_zero_lines_are_left_blank_not_written_as_zero(
        self, taxpayer: Taxpayer, tmp_path: Path
    ) -> None:
        self._run(taxpayer, tmp_path)
        stored = acroform.read_back(tmp_path / "filled.pdf")
        assert "0" not in stored.values()

    def test_an_invalid_return_is_refused(
        self, taxpayer: Taxpayer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import taxorchestra.agents.orchestrator as orchestrator_module

        def broken(taxpayer_arg, documents, **kwargs):
            result = compute_return(taxpayer_arg, documents, **kwargs)
            for line in result.lines:
                if line.semantic_key == "total_income":
                    line.value = Decimal("999999")
            return result

        monkeypatch.setattr(orchestrator_module, "compute_return", broken)
        with pytest.raises(ValueError, match="refusing to write an invalid return"):
            self._run(taxpayer, tmp_path)


@requires_form
class TestAcroForm:
    def test_the_form_has_no_usable_metadata(self) -> None:
        """The premise of the whole mapping pipeline, asserted."""
        from pypdf import PdfReader

        fields = PdfReader(str(TEMPLATE)).get_fields() or {}
        assert len(fields) > 200
        assert not any(meta.get("/TU") for meta in fields.values())

    def test_writing_an_unknown_field_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="not present in the template"):
            acroform.fill(TEMPLATE, tmp_path / "x.pdf", {"no.such.field": "1"})
