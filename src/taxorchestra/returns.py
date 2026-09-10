"""Turn extracted documents into Form 1040 line values.

Pure and deterministic — no model calls here. Given the same documents this
always produces the same return, which is what makes the pipeline testable and
what lets the validation agent check arithmetic rather than vibes.

Every line carries a `derivation` string. When a reviewer asks why line 15 says
what it says, the answer travels with the number.
"""

from __future__ import annotations

from decimal import Decimal

from taxorchestra import taxdata
from taxorchestra.models import (
    W2,
    Form1040Line,
    Form1040Return,
    Form1099DIV,
    Form1099INT,
    Form1099NEC,
    SourceDocument,
    Taxpayer,
)

ZERO = Decimal("0")


def _sum(values: list[Decimal]) -> Decimal:
    return sum(values, ZERO)


def compute_return(
    taxpayer: Taxpayer,
    documents: list[SourceDocument],
    *,
    tax_year: int = taxdata.TAX_YEAR,
) -> Form1040Return:
    w2s = [d for d in documents if isinstance(d, W2)]
    ints = [d for d in documents if isinstance(d, Form1099INT)]
    divs = [d for d in documents if isinstance(d, Form1099DIV)]
    necs = [d for d in documents if isinstance(d, Form1099NEC)]

    lines: list[Form1040Line] = []

    def add(line: str, key: str, value: Decimal, derivation: str) -> Decimal:
        """Record a line in whole dollars and hand back the *rounded* figure.

        Returning the unrounded value would let later lines compute from cents
        while the form prints whole dollars — line 16 would then be the tax on a
        taxable income that does not match line 15 as printed. Pub 17 has it the
        other way round: include cents when totalling source documents, then
        round at the line, and derive each later line from the entered values.
        """
        rounded = taxdata.to_dollars(value)
        lines.append(
            Form1040Line(
                line=line,
                semantic_key=key,
                value=rounded,
                derivation=derivation,
            )
        )
        return rounded

    # ---- income --------------------------------------------------------
    wages = add(
        "1a",
        "wages_w2_box1",
        _sum([w.wages for w in w2s]),
        f"Box 1 of {len(w2s)} Form W-2(s)",
    )
    wages_total = add(
        "1z",
        "wages_total",
        wages,
        "Line 1a; lines 1b–1h are zero for this return",
    )

    add(
        "2a",
        "interest_tax_exempt",
        _sum([i.tax_exempt_interest for i in ints]),
        f"Box 8 of {len(ints)} Form 1099-INT(s); informational, not taxed",
    )
    taxable_interest = add(
        "2b",
        "interest_taxable",
        _sum([i.interest_income for i in ints]),
        f"Box 1 of {len(ints)} Form 1099-INT(s)",
    )

    add(
        "3a",
        "dividends_qualified",
        _sum([d.qualified_dividends for d in divs]),
        f"Box 1b of {len(divs)} Form 1099-DIV(s)",
    )
    ordinary_div = add(
        "3b",
        "dividends_ordinary",
        _sum([d.ordinary_dividends for d in divs]),
        f"Box 1a of {len(divs)} Form 1099-DIV(s)",
    )

    # 1099-NEC is self-employment income. A complete return routes it through
    # Schedule C and Schedule SE; this pipeline carries the gross amount into
    # total income and leaves the validation agent to flag the missing SE tax
    # rather than quietly under-report it.
    nec_total = _sum([n.nonemployee_compensation for n in necs])
    if necs:
        add(
            "8",
            "other_income_schedule1",
            nec_total,
            f"Box 1 of {len(necs)} Form 1099-NEC(s), carried gross — "
            "Schedule C/SE not computed",
        )

    total_income = add(
        "9",
        "total_income",
        wages_total + taxable_interest + ordinary_div + nec_total,
        "Lines 1z + 2b + 3b + 8",
    )

    agi = add(
        "11",
        "adjusted_gross_income",
        total_income,
        "Line 9 less line 10 adjustments (none for this return)",
    )

    # ---- deductions ----------------------------------------------------
    deduction = add(
        "12",
        "deduction_standard_or_itemized",
        taxdata.standard_deduction(taxpayer.filing_status),
        f"{tax_year} standard deduction for {taxpayer.filing_status.value}",
    )
    deduction_total = add(
        "14",
        "deduction_total",
        deduction,
        "Line 12 plus line 13 QBI deduction (zero for this return)",
    )

    taxable_income = add(
        "15",
        "taxable_income",
        max(ZERO, agi - deduction_total),
        "Line 11 less line 14, floored at zero",
    )

    # ---- tax -----------------------------------------------------------
    tax = add(
        "16",
        "tax_computed",
        taxdata.ordinary_income_tax(taxable_income, taxpayer.filing_status),
        f"Ordinary-income brackets for {taxpayer.filing_status.value}",
    )
    tax_total = add("24", "tax_total", tax, "Line 16; no additional taxes on this return")

    # ---- payments ------------------------------------------------------
    withholding_w2 = add(
        "25a",
        "withholding_w2",
        _sum([w.federal_income_tax_withheld for w in w2s]),
        "Box 2 of the Form W-2(s)",
    )
    withholding_1099 = add(
        "25b",
        "withholding_1099",
        _sum(
            [i.federal_income_tax_withheld for i in ints]
            + [d.federal_income_tax_withheld for d in divs]
            + [n.federal_income_tax_withheld for n in necs]
        ),
        "Box 4 of the Form 1099(s)",
    )
    withholding_total = add(
        "25d",
        "withholding_total",
        withholding_w2 + withholding_1099,
        "Lines 25a + 25b",
    )
    payments = add(
        "33",
        "payments_total",
        withholding_total,
        "Line 25d; no estimated payments or refundable credits",
    )

    # ---- settlement ----------------------------------------------------
    add(
        "34",
        "refund_overpaid",
        max(ZERO, payments - tax_total),
        "Line 33 less line 24 when payments exceed tax",
    )
    add(
        "35a",
        "refund_requested",
        max(ZERO, payments - tax_total),
        "Entire overpayment refunded; none applied forward",
    )
    add(
        "37",
        "amount_owed",
        max(ZERO, tax_total - payments),
        "Line 24 less line 33 when tax exceeds payments",
    )

    return Form1040Return(tax_year=tax_year, taxpayer=taxpayer, lines=lines)
