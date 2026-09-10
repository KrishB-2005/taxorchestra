"""Typed contracts shared by every agent in the pipeline.

Everything that crosses an agent boundary is a Pydantic model. That is the
main defence against the usual failure mode of LLM pipelines: a model returns
prose where a number was expected, three stages later something concatenates a
string onto a dollar amount, and the bug surfaces as a wrong refund.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Money = Decimal


class FilingStatus(StrEnum):
    SINGLE = "single"
    MARRIED_JOINTLY = "married_filing_jointly"
    MARRIED_SEPARATELY = "married_filing_separately"
    HEAD_OF_HOUSEHOLD = "head_of_household"
    QUALIFYING_SURVIVING_SPOUSE = "qualifying_surviving_spouse"


class DocumentKind(StrEnum):
    W2 = "w2"
    INT_1099 = "1099-int"
    DIV_1099 = "1099-div"
    NEC_1099 = "1099-nec"


class StrictModel(BaseModel):
    """Reject unknown keys rather than silently dropping them.

    An LLM that invents an extra field is telling us the prompt drifted; we
    want that to fail loudly at the boundary instead of vanishing.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# --------------------------------------------------------------------------
# Taxpayer identity
# --------------------------------------------------------------------------


class Address(StrictModel):
    street: str
    apartment: str | None = None
    city: str
    state: str = Field(min_length=2, max_length=2)
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")


class Taxpayer(StrictModel):
    first_name: str
    last_name: str
    ssn: str = Field(pattern=r"^\d{3}-?\d{2}-?\d{4}$")
    address: Address
    filing_status: FilingStatus
    spouse_first_name: str | None = None
    spouse_last_name: str | None = None
    spouse_ssn: str | None = Field(default=None, pattern=r"^\d{3}-?\d{2}-?\d{4}$")

    @field_validator("ssn", "spouse_ssn")
    @classmethod
    def _normalise_ssn(cls, v: str | None) -> str | None:
        if v is None:
            return None
        digits = v.replace("-", "")
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"


# --------------------------------------------------------------------------
# Source documents (agent 1 output)
# --------------------------------------------------------------------------


class W2(StrictModel):
    kind: Literal[DocumentKind.W2] = DocumentKind.W2
    employer_name: str
    employer_ein: str
    employee_ssn: str
    wages: Money = Field(description="Box 1 — wages, tips, other compensation")
    federal_income_tax_withheld: Money = Field(description="Box 2")
    social_security_wages: Money | None = Field(default=None, description="Box 3")
    social_security_tax_withheld: Money | None = Field(default=None, description="Box 4")
    medicare_wages: Money | None = Field(default=None, description="Box 5")
    medicare_tax_withheld: Money | None = Field(default=None, description="Box 6")


class Form1099INT(StrictModel):
    kind: Literal[DocumentKind.INT_1099] = DocumentKind.INT_1099
    payer_name: str
    payer_tin: str
    interest_income: Money = Field(description="Box 1")
    tax_exempt_interest: Money = Field(default=Decimal("0"), description="Box 8")
    federal_income_tax_withheld: Money = Field(default=Decimal("0"), description="Box 4")


class Form1099DIV(StrictModel):
    kind: Literal[DocumentKind.DIV_1099] = DocumentKind.DIV_1099
    payer_name: str
    payer_tin: str
    ordinary_dividends: Money = Field(description="Box 1a")
    qualified_dividends: Money = Field(default=Decimal("0"), description="Box 1b")
    federal_income_tax_withheld: Money = Field(default=Decimal("0"), description="Box 4")


class Form1099NEC(StrictModel):
    kind: Literal[DocumentKind.NEC_1099] = DocumentKind.NEC_1099
    payer_name: str
    payer_tin: str
    nonemployee_compensation: Money = Field(description="Box 1")
    federal_income_tax_withheld: Money = Field(default=Decimal("0"), description="Box 4")


SourceDocument = W2 | Form1099INT | Form1099DIV | Form1099NEC


class ExtractionResult(StrictModel):
    """One extracted document plus the provenance needed to audit it."""

    document: SourceDocument
    source_path: str
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None


# --------------------------------------------------------------------------
# Field mapping (agent 2 output)
# --------------------------------------------------------------------------


class FieldType(StrEnum):
    TEXT = "text"
    MONEY = "money"
    CHECKBOX = "checkbox"
    SSN = "ssn"
    UNKNOWN = "unknown"


class FieldMapping(StrictModel):
    """A cryptic AcroForm field resolved to something a human can reason about.

    `topmostSubform[0].Page1[0].f1_47[0]` carries no meaning on its own. After
    mapping it becomes line "1a", semantic key `wages_w2_box1`, type money.
    """

    acro_name: str = Field(description="Raw AcroForm field name from the PDF")
    page: int
    line: str | None = Field(default=None, description="Form 1040 line label, e.g. '1a'")
    semantic_key: str | None = Field(
        default=None, description="Stable snake_case key, e.g. 'wages_w2_box1'"
    )
    field_type: FieldType = FieldType.UNKNOWN
    label_text: str = Field(default="", description="Label text recovered near the widget")
    section: str | None = Field(default=None, description="Enclosing section, e.g. 'Income'")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    phase: int = Field(
        default=0,
        description=(
            "Which phase recovered this field's label "
            "(1 geometric, 2 structural, 3 caption)"
        ),
    )

    @property
    def resolved(self) -> bool:
        return self.semantic_key is not None


class FieldCatalog(StrictModel):
    form_id: str
    form_year: int
    pdf_sha256: str
    mappings: list[FieldMapping]

    def by_key(self) -> dict[str, FieldMapping]:
        return {m.semantic_key: m for m in self.mappings if m.semantic_key}

    def by_acro_name(self) -> dict[str, FieldMapping]:
        return {m.acro_name: m for m in self.mappings}

    @property
    def resolved_count(self) -> int:
        return sum(1 for m in self.mappings if m.resolved)


# --------------------------------------------------------------------------
# Return computation (agent 4 output)
# --------------------------------------------------------------------------


class Form1040Line(StrictModel):
    line: str
    semantic_key: str
    value: Money
    derivation: str = Field(description="Plain-English account of where the number came from")


class Form1040Return(StrictModel):
    tax_year: int
    taxpayer: Taxpayer
    lines: list[Form1040Line]

    def value(self, semantic_key: str) -> Money:
        for line in self.lines:
            if line.semantic_key == semantic_key:
                return line.value
        return Decimal("0")

    def as_dict(self) -> dict[str, Money]:
        return {line.semantic_key: line.value for line in self.lines}


# --------------------------------------------------------------------------
# Validation (agent 3 output)
# --------------------------------------------------------------------------


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(StrictModel):
    severity: Severity
    semantic_key: str | None = None
    message: str
    citation: str | None = Field(
        default=None, description="Retrieved IRS rule this judgement was grounded in"
    )


class ValidationReport(StrictModel):
    issues: list[ValidationIssue]

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]


# --------------------------------------------------------------------------
# Final artefact
# --------------------------------------------------------------------------


class FilingResult(StrictModel):
    tax_year: int
    return_: Form1040Return = Field(alias="return")
    validation: ValidationReport
    output_pdf: str
    fields_written: int
    fields_attempted: int
    elapsed_seconds: float
    cache_hit: bool

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @property
    def fill_rate(self) -> float:
        if self.fields_attempted == 0:
            return 0.0
        return self.fields_written / self.fields_attempted
