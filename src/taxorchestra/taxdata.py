"""Tax-year constants and the ordinary-income tax computation.

⚠️  These figures are transcribed from published 2025 inflation-adjustment
    figures. They are held here as data, versioned by year, precisely because
    they change annually and must be re-verified against the IRS revenue
    procedure for the year in question before anyone relies on them.

    TaxOrchestra is a document-automation demonstration, not tax software. It
    computes the ordinary-income case only: no capital-gains worksheet, no
    AMT, no credits beyond what is passed in, no state return.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from taxorchestra.models import FilingStatus

TAX_YEAR = 2025

# Standard deduction by filing status.
STANDARD_DEDUCTION: dict[FilingStatus, Decimal] = {
    FilingStatus.SINGLE: Decimal("15750"),
    FilingStatus.MARRIED_JOINTLY: Decimal("31500"),
    FilingStatus.MARRIED_SEPARATELY: Decimal("15750"),
    FilingStatus.HEAD_OF_HOUSEHOLD: Decimal("23625"),
    FilingStatus.QUALIFYING_SURVIVING_SPOUSE: Decimal("31500"),
}

# (upper bound of bracket, marginal rate). The final bracket is unbounded.
_Bracket = tuple[Decimal | None, Decimal]

BRACKETS: dict[FilingStatus, list[_Bracket]] = {
    FilingStatus.SINGLE: [
        (Decimal("11925"), Decimal("0.10")),
        (Decimal("48475"), Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250525"), Decimal("0.32")),
        (Decimal("626350"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
    FilingStatus.MARRIED_JOINTLY: [
        (Decimal("23850"), Decimal("0.10")),
        (Decimal("96950"), Decimal("0.12")),
        (Decimal("206700"), Decimal("0.22")),
        (Decimal("394600"), Decimal("0.24")),
        (Decimal("501050"), Decimal("0.32")),
        (Decimal("751600"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
    FilingStatus.MARRIED_SEPARATELY: [
        (Decimal("11925"), Decimal("0.10")),
        (Decimal("48475"), Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250525"), Decimal("0.32")),
        (Decimal("375800"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
    FilingStatus.HEAD_OF_HOUSEHOLD: [
        (Decimal("17000"), Decimal("0.10")),
        (Decimal("64850"), Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250500"), Decimal("0.32")),
        (Decimal("626350"), Decimal("0.35")),
        (None, Decimal("0.37")),
    ],
}
BRACKETS[FilingStatus.QUALIFYING_SURVIVING_SPOUSE] = BRACKETS[FilingStatus.MARRIED_JOINTLY]

CENT = Decimal("0.01")
DOLLAR = Decimal("1")


def to_dollars(amount: Decimal) -> Decimal:
    """Round to whole dollars the way the IRS instructions specify.

    Drop amounts under 50 cents, raise 50 cents and over to the next dollar.
    Decimal's default is ROUND_HALF_EVEN, which sends $1,192.50 down to $1,192 —
    correct for accounting, wrong for a tax return.
    """
    return amount.quantize(DOLLAR, rounding=ROUND_HALF_UP)


def standard_deduction(status: FilingStatus) -> Decimal:
    return STANDARD_DEDUCTION[status]


def ordinary_income_tax(taxable_income: Decimal, status: FilingStatus) -> Decimal:
    """Marginal-bracket tax on ordinary income.

    Walks the brackets accumulating tax on each slice. Returns whole dollars,
    which is how the 1040 is filed.
    """
    if taxable_income <= 0:
        return Decimal("0")

    tax = Decimal("0")
    lower = Decimal("0")
    for upper, rate in BRACKETS[status]:
        if upper is None or taxable_income <= upper:
            tax += (taxable_income - lower) * rate
            break
        tax += (upper - lower) * rate
        lower = upper

    return to_dollars(tax)
