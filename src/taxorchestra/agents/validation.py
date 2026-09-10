"""Agent 3 — Validation: check the computed return against the IRS rules.

The checks themselves are arithmetic and deterministic. That is the point: the
failure mode this agent exists to catch is a language model inventing a number,
and you do not catch invented numbers by asking another language model whether
they look right. You re-derive them.

Retrieval's job here is *grounding*, not judgement. Each finding is attached to
the rule it violates so a reviewer gets "line 15 disagrees with line 11 − 14,
per Form 1040 line 15" rather than "the validator flagged line 15".
"""

from __future__ import annotations

from decimal import Decimal

from taxorchestra.knowledge.store import KnowledgeStore
from taxorchestra.models import (
    W2,
    Form1040Return,
    Form1099NEC,
    Severity,
    SourceDocument,
    ValidationIssue,
    ValidationReport,
)

ZERO = Decimal("0")


class ValidationAgent:
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def validate(
        self,
        return_: Form1040Return,
        documents: list[SourceDocument],
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        value = return_.value

        issues.extend(self._check_arithmetic(value))
        issues.extend(self._check_sanity(return_, value))
        issues.extend(self._check_documents(return_, documents))

        return ValidationReport(issues=issues)

    # -- helpers ---------------------------------------------------------

    def _cite(self, query: str) -> str | None:
        hits = self.store.search(query, k=1)
        return hits[0].rule.citation if hits else None

    def _error(self, key: str, message: str, query: str) -> ValidationIssue:
        return ValidationIssue(
            severity=Severity.ERROR,
            semantic_key=key,
            message=message,
            citation=self._cite(query),
        )

    def _warn(self, key: str | None, message: str, query: str) -> ValidationIssue:
        return ValidationIssue(
            severity=Severity.WARNING,
            semantic_key=key,
            message=message,
            citation=self._cite(query),
        )

    # -- checks ----------------------------------------------------------

    def _check_arithmetic(self, value) -> list[ValidationIssue]:
        """Re-derive every total from its components."""
        issues: list[ValidationIssue] = []

        identities: list[tuple[str, Decimal, Decimal, str, str]] = [
            (
                "total_income",
                value("total_income"),
                value("wages_total")
                + value("interest_taxable")
                + value("dividends_ordinary")
                + value("other_income_schedule1"),
                "line 9 must equal lines 1z + 2b + 3b + 8",
                "total income line 9 sum",
            ),
            (
                "taxable_income",
                value("taxable_income"),
                max(ZERO, value("adjusted_gross_income") - value("deduction_total")),
                "line 15 must equal line 11 less line 14, floored at zero",
                "taxable income line 15 subtract",
            ),
            (
                "withholding_total",
                value("withholding_total"),
                value("withholding_w2") + value("withholding_1099"),
                "line 25d must equal lines 25a + 25b",
                "withholding lines 25a 25b 25d",
            ),
            (
                "payments_total",
                value("payments_total"),
                value("withholding_total"),
                "line 33 must equal line 25d for a return with no other payments",
                "total payments line 33",
            ),
        ]

        for key, actual, expected, message, query in identities:
            if actual != expected:
                issues.append(
                    self._error(
                        key,
                        f"{message}; computed {expected} but the return says {actual}",
                        query,
                    )
                )

        # Exactly one of refund / owed may be non-zero.
        refund = value("refund_overpaid")
        owed = value("amount_owed")
        if refund > ZERO and owed > ZERO:
            issues.append(
                self._error(
                    "amount_owed",
                    f"both a refund ({refund}) and a balance due ({owed}) are set; "
                    "exactly one can be non-zero",
                    "overpayment line 34 amount owed line 37",
                )
            )

        settlement = value("payments_total") - value("tax_total")
        if settlement > ZERO and refund != settlement:
            issues.append(
                self._error(
                    "refund_overpaid",
                    f"payments exceed tax by {settlement} but line 34 says {refund}",
                    "overpayment line 34",
                )
            )
        if settlement < ZERO and owed != -settlement:
            issues.append(
                self._error(
                    "amount_owed",
                    f"tax exceeds payments by {-settlement} but line 37 says {owed}",
                    "amount owed line 37",
                )
            )

        return issues

    def _check_sanity(self, return_: Form1040Return, value) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for line in return_.lines:
            if line.value < ZERO:
                issues.append(
                    self._error(
                        line.semantic_key,
                        f"line {line.line} is negative ({line.value}); "
                        "1040 dollar entries are non-negative here",
                        "negative amounts dollar entries",
                    )
                )

        qualified = value("dividends_qualified")
        ordinary = value("dividends_ordinary")
        if qualified > ordinary:
            issues.append(
                self._error(
                    "dividends_qualified",
                    f"qualified dividends ({qualified}) exceed ordinary dividends "
                    f"({ordinary}); qualified are a subset of ordinary",
                    "qualified dividends subset ordinary line 3a 3b",
                )
            )

        return issues

    def _check_documents(
        self,
        return_: Form1040Return,
        documents: list[SourceDocument],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        expected_ssn = return_.taxpayer.ssn.replace("-", "")
        for doc in documents:
            if isinstance(doc, W2):
                if doc.employee_ssn.replace("-", "") != expected_ssn:
                    issues.append(
                        self._error(
                            "taxpayer_ssn",
                            f"W-2 from {doc.employer_name} shows SSN "
                            f"{doc.employee_ssn}, which does not match the return",
                            "social security number must match W-2",
                        )
                    )

        if any(isinstance(d, Form1099NEC) for d in documents):
            issues.append(
                self._warn(
                    "other_income_schedule1",
                    "1099-NEC income is present. It is carried into total income "
                    "gross; Schedule C and self-employment tax on Schedule SE are "
                    "not computed by this pipeline",
                    "nonemployee compensation schedule C self-employment tax",
                )
            )

        return issues
