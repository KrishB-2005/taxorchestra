"""A small corpus of Form 1040 rules, each with a citation.

Deliberately small and hand-written rather than scraped. The validation agent
needs statements it can quote back at a reviewer, and a compact curated corpus
retrieves far more precisely than the full instructions PDF chunked blindly.

Every entry cites the line or publication it came from so a finding can be
traced. Verify against the current-year instructions before relying on any of
it — see the warning in `taxdata.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    citation: str
    text: str


RULES: list[Rule] = [
    Rule(
        "wages-1a",
        "Form 1040 (2025), line 1a",
        "Line 1a reports the total amount from box 1 of all Forms W-2. Box 1 is "
        "wages, tips and other compensation, and is the figure that flows to the "
        "return — not box 3 or box 5, which are the Social Security and Medicare "
        "wage bases and may differ.",
    ),
    Rule(
        "wages-1z",
        "Form 1040 (2025), line 1z",
        "Line 1z is the sum of lines 1a through 1h. It is the total wage figure "
        "carried into total income on line 9.",
    ),
    Rule(
        "interest-2a-2b",
        "Form 1040 (2025), lines 2a and 2b",
        "Line 2a reports tax-exempt interest and line 2b reports taxable interest. "
        "Tax-exempt interest is reported for informational purposes and is not "
        "included in total income; only line 2b flows into line 9.",
    ),
    Rule(
        "dividends-3a-3b",
        "Form 1040 (2025), lines 3a and 3b",
        "Line 3a reports qualified dividends and line 3b reports ordinary "
        "dividends. Qualified dividends are a subset of ordinary dividends, so "
        "line 3a can never exceed line 3b. Only line 3b is added into total income.",
    ),
    Rule(
        "total-income-9",
        "Form 1040 (2025), line 9",
        "Line 9, total income, is the sum of lines 1z, 2b, 3b, 4b, 5b, 6b, 7 and 8.",
    ),
    Rule(
        "agi-11",
        "Form 1040 (2025), line 11",
        "Line 11, adjusted gross income, is line 9 less the adjustments to income "
        "on line 10, which come from Schedule 1.",
    ),
    Rule(
        "deduction-12",
        "Form 1040 (2025), line 12",
        "Line 12 is the greater of the standard deduction for the filing status or "
        "total itemized deductions from Schedule A. A taxpayer may not claim both.",
    ),
    Rule(
        "taxable-income-15",
        "Form 1040 (2025), line 15",
        "Line 15, taxable income, is line 11 less line 14. If the result is zero or "
        "less, enter -0-. Taxable income is never negative.",
    ),
    Rule(
        "withholding-25",
        "Form 1040 (2025), lines 25a–25d",
        "Line 25a reports federal income tax withheld shown on Forms W-2 (box 2), "
        "line 25b that shown on Forms 1099 (box 4), and line 25d is their total "
        "plus line 25c. Withholding is a payment, not a deduction.",
    ),
    Rule(
        "settlement-34-37",
        "Form 1040 (2025), lines 34 and 37",
        "If line 33 (total payments) exceeds line 24 (total tax), the difference is "
        "the overpayment on line 34. If line 24 exceeds line 33, the difference is "
        "the amount owed on line 37. Exactly one of the two is non-zero.",
    ),
    Rule(
        "nec-self-employment",
        "Schedule C and Schedule SE",
        "Nonemployee compensation reported in box 1 of Form 1099-NEC is self-"
        "employment income. It is reported on Schedule C, and net earnings of $400 "
        "or more require Schedule SE and self-employment tax.",
    ),
    Rule(
        "ssn-consistency",
        "Form 1040 (2025), identity section",
        "The social security number entered on the return must match the SSN shown "
        "on the taxpayer's Forms W-2 and 1099. A mismatch will cause the return to "
        "be rejected on e-file.",
    ),
    Rule(
        "filing-status-exclusive",
        "Form 1040 (2025), filing status",
        "Exactly one filing status box is checked: single, married filing jointly, "
        "married filing separately, head of household, or qualifying surviving "
        "spouse.",
    ),
    Rule(
        "negative-amounts",
        "Form 1040 (2025), general instructions",
        "Dollar entries on the return are non-negative except where the form "
        "explicitly provides for a loss. Wages, interest, dividends, withholding "
        "and totals are never negative.",
    ),
]
