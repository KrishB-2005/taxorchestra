"""Interactive interview: ask for what's needed, fill the form, file nothing.

The CLI's `file` command assumes you already have PDFs and know the flags. This
asks instead - identity, filing status, then each income document in turn,
either read from a PDF or typed in by hand.

Nothing here transmits anything. The only side effects are a filled PDF on your
disk and, if you ask for it, a saved answers file so you don't retype next time.
"""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import typer

from taxorchestra.agents.extraction import ExtractionAgent
from taxorchestra.models import (
    W2,
    Address,
    FilingStatus,
    Form1099DIV,
    Form1099INT,
    Form1099NEC,
    SourceDocument,
    Taxpayer,
)

BOLD = typer.colors.BRIGHT_WHITE
DIM = typer.colors.BRIGHT_BLACK
ACCENT = typer.colors.CYAN


def rule(title: str = "") -> None:
    typer.echo("")
    if title:
        typer.secho(f"-- {title} " + "-" * max(0, 58 - len(title)), fg=ACCENT)
    else:
        typer.secho("-" * 62, fg=DIM)


def note(text: str) -> None:
    typer.secho(f"   {text}", fg=DIM)


def mask_ssn(ssn: str) -> str:
    digits = ssn.replace("-", "")
    return f"***-**-{digits[-4:]}" if len(digits) >= 4 else "***-**-****"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def ask_text(label: str, default: str | None = None, *, allow_blank: bool = False) -> str:
    while True:
        value = typer.prompt(f"  {label}", default=default or "", show_default=bool(default))
        value = value.strip()
        if value or allow_blank:
            return value
        typer.secho("   > required", fg=typer.colors.YELLOW)


def ask_money(label: str, default: Decimal | None = None) -> Decimal:
    """Accept 92450, 92,450, $92,450.00 - however it is printed on the form."""
    while True:
        raw = typer.prompt(
            f"  {label}",
            default=str(default) if default is not None else "0",
            show_default=True,
        )
        cleaned = raw.strip().replace("$", "").replace(",", "").replace(" ", "")
        if not cleaned:
            return Decimal("0")
        try:
            amount = Decimal(cleaned)
        except InvalidOperation:
            typer.secho(f"   > '{raw}' is not an amount", fg=typer.colors.YELLOW)
            continue
        if amount < 0:
            typer.secho("   > amounts on these boxes are not negative", fg=typer.colors.YELLOW)
            continue
        return amount


def ask_choice(label: str, options: list[str], default: int = 1) -> int:
    """One-based menu. Returns the chosen index."""
    typer.secho(f"  {label}", fg=BOLD)
    for index, option in enumerate(options, start=1):
        typer.echo(f"    {index}. {option}")
    while True:
        raw = typer.prompt("  choice", default=str(default))
        try:
            choice = int(raw)
        except ValueError:
            typer.secho("   > enter a number", fg=typer.colors.YELLOW)
            continue
        if 1 <= choice <= len(options):
            return choice
        typer.secho(f"   > pick 1-{len(options)}", fg=typer.colors.YELLOW)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

_STATUS_OPTIONS: list[tuple[str, FilingStatus]] = [
    ("Single", FilingStatus.SINGLE),
    ("Married filing jointly", FilingStatus.MARRIED_JOINTLY),
    ("Married filing separately", FilingStatus.MARRIED_SEPARATELY),
    ("Head of household", FilingStatus.HEAD_OF_HOUSEHOLD),
    ("Qualifying surviving spouse", FilingStatus.QUALIFYING_SURVIVING_SPOUSE),
]


def ask_taxpayer(prefill: dict[str, Any] | None = None) -> Taxpayer:
    prefill = prefill or {}
    address_prefill = prefill.get("address", {})

    rule("About you")
    note("This stays on your machine. Nothing is uploaded or transmitted.")
    typer.echo("")

    first = ask_text("First name and middle initial", prefill.get("first_name"))
    last = ask_text("Last name", prefill.get("last_name"))

    while True:
        ssn = ask_text("Social security number (###-##-####)", prefill.get("ssn"))
        digits = ssn.replace("-", "").replace(" ", "")
        if len(digits) == 9 and digits.isdigit():
            ssn = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
            break
        typer.secho("   > needs to be 9 digits", fg=typer.colors.YELLOW)

    rule("Where you live")
    street = ask_text("Street address", address_prefill.get("street"))
    apartment = ask_text(
        "Apartment / unit (blank if none)",
        address_prefill.get("apartment"),
        allow_blank=True,
    )
    city = ask_text("City or town", address_prefill.get("city"))

    while True:
        state = ask_text("State (2 letters)", address_prefill.get("state")).upper()
        if len(state) == 2 and state.isalpha():
            break
        typer.secho("   > two letters, e.g. MD", fg=typer.colors.YELLOW)

    while True:
        zip_code = ask_text("ZIP code", address_prefill.get("zip_code"))
        bare = zip_code.replace("-", "")
        if bare.isdigit() and len(bare) in (5, 9):
            zip_code = bare[:5] if len(bare) == 5 else f"{bare[:5]}-{bare[5:]}"
            break
        typer.secho("   > 5 or 9 digits", fg=typer.colors.YELLOW)

    rule("Filing status")
    note("If you are unsure, the IRS Interactive Tax Assistant can tell you.")
    typer.echo("")
    default_index = 1
    if prefill.get("filing_status"):
        for index, (_, status) in enumerate(_STATUS_OPTIONS, start=1):
            if status.value == prefill["filing_status"]:
                default_index = index
    chosen = ask_choice(
        "Which applies to you?",
        [label for label, _ in _STATUS_OPTIONS],
        default=default_index,
    )
    filing_status = _STATUS_OPTIONS[chosen - 1][1]

    spouse_first = spouse_last = spouse_ssn = None
    if filing_status is FilingStatus.MARRIED_JOINTLY:
        rule("Your spouse")
        spouse_first = ask_text(
            "Spouse first name and middle initial", prefill.get("spouse_first_name")
        )
        spouse_last = ask_text("Spouse last name", prefill.get("spouse_last_name") or last)
        while True:
            spouse_ssn = ask_text("Spouse social security number", prefill.get("spouse_ssn"))
            digits = spouse_ssn.replace("-", "").replace(" ", "")
            if len(digits) == 9 and digits.isdigit():
                spouse_ssn = f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
                break
            typer.secho("   > needs to be 9 digits", fg=typer.colors.YELLOW)

    return Taxpayer(
        first_name=first,
        last_name=last,
        ssn=ssn,
        address=Address(
            street=street,
            apartment=apartment or None,
            city=city,
            state=state,
            zip_code=zip_code,
        ),
        filing_status=filing_status,
        spouse_first_name=spouse_first,
        spouse_last_name=spouse_last,
        spouse_ssn=spouse_ssn,
    )


# ---------------------------------------------------------------------------
# Income documents
# ---------------------------------------------------------------------------

_DOC_MENU = [
    "W-2 - wages from an employer",
    "1099-INT - bank or savings interest",
    "1099-DIV - dividends from investments",
    "1099-NEC - contract or freelance work",
    "Nothing more to add",
]


def _read_pdf(extraction: ExtractionAgent) -> SourceDocument | None:
    while True:
        raw = ask_text("Path to the PDF (blank to go back)", allow_blank=True)
        if not raw:
            return None
        path = Path(raw.strip().strip('"').strip("'")).expanduser()
        if not path.exists():
            typer.secho(f"   > no file at {path}", fg=typer.colors.YELLOW)
            continue
        try:
            result = extraction.extract(path)
        except ValueError as exc:
            typer.secho(f"   > {exc}", fg=typer.colors.YELLOW)
            note("Choose 'type the numbers in' instead if the PDF is a scan.")
            return None
        typer.secho(
            f"   [ok] read {type(result.document).__name__} ({result.notes})",
            fg=typer.colors.GREEN,
        )
        return result.document


def _w2_by_hand() -> W2:
    note("Copy these straight off the form - the box numbers are printed on it.")
    return W2(
        employer_name=ask_text("Employer name"),
        employer_ein=ask_text("Employer EIN (##-#######)"),
        employee_ssn=ask_text("Your SSN as printed on this W-2"),
        wages=ask_money("Box 1  wages, tips, other compensation"),
        federal_income_tax_withheld=ask_money("Box 2  federal income tax withheld"),
        social_security_wages=ask_money("Box 3  social security wages"),
        social_security_tax_withheld=ask_money("Box 4  social security tax withheld"),
        medicare_wages=ask_money("Box 5  medicare wages and tips"),
        medicare_tax_withheld=ask_money("Box 6  medicare tax withheld"),
    )


def _int_by_hand() -> Form1099INT:
    return Form1099INT(
        payer_name=ask_text("Payer name (the bank)"),
        payer_tin=ask_text("Payer TIN"),
        interest_income=ask_money("Box 1  interest income"),
        federal_income_tax_withheld=ask_money("Box 4  federal income tax withheld"),
        tax_exempt_interest=ask_money("Box 8  tax-exempt interest"),
    )


def _div_by_hand() -> Form1099DIV:
    return Form1099DIV(
        payer_name=ask_text("Payer name"),
        payer_tin=ask_text("Payer TIN"),
        ordinary_dividends=ask_money("Box 1a  total ordinary dividends"),
        qualified_dividends=ask_money("Box 1b  qualified dividends"),
        federal_income_tax_withheld=ask_money("Box 4  federal income tax withheld"),
    )


def _nec_by_hand() -> Form1099NEC:
    return Form1099NEC(
        payer_name=ask_text("Payer name"),
        payer_tin=ask_text("Payer TIN"),
        nonemployee_compensation=ask_money("Box 1  nonemployee compensation"),
        federal_income_tax_withheld=ask_money("Box 4  federal income tax withheld"),
    )


_BY_HAND = {1: _w2_by_hand, 2: _int_by_hand, 3: _div_by_hand, 4: _nec_by_hand}


def describe(document: SourceDocument) -> str:
    if isinstance(document, W2):
        return f"W-2      {document.employer_name} - box 1 ${document.wages:,.2f}"
    if isinstance(document, Form1099INT):
        return f"1099-INT {document.payer_name} - box 1 ${document.interest_income:,.2f}"
    if isinstance(document, Form1099DIV):
        return f"1099-DIV {document.payer_name} - box 1a ${document.ordinary_dividends:,.2f}"
    if isinstance(document, Form1099NEC):
        return (
            f"1099-NEC {document.payer_name} - box 1 "
            f"${document.nonemployee_compensation:,.2f}"
        )
    return type(document).__name__


def ask_documents(
    extraction: ExtractionAgent,
    existing: list[SourceDocument] | None = None,
) -> list[SourceDocument]:
    documents: list[SourceDocument] = list(existing or [])

    rule("Your income documents")
    note("Add each W-2 and 1099 you received. Read them from a PDF or type them in.")
    if documents:
        typer.echo("")
        note("Already added:")
        for document in documents:
            typer.secho(f"     * {describe(document)}", fg=DIM)

    while True:
        typer.echo("")
        kind = ask_choice("What would you like to add?", _DOC_MENU, default=5 if documents else 1)
        if kind == 5:
            break

        how = ask_choice(
            "How would you like to enter it?",
            ["Read it from a PDF file", "Type the numbers in myself"],
            default=2,
        )

        document: SourceDocument | None
        if how == 1:
            document = _read_pdf(extraction)
        else:
            try:
                document = _BY_HAND[kind]()
            except Exception as exc:  # noqa: BLE001 - user typo, not a crash
                typer.secho(f"   > {exc}", fg=typer.colors.YELLOW)
                document = None

        if document is not None:
            documents.append(document)
            typer.secho(f"   [ok] added - {describe(document)}", fg=typer.colors.GREEN)

    if not documents:
        typer.secho(
            "\n  No income documents added. There is nothing to put on a return.",
            fg=typer.colors.YELLOW,
        )
    return documents


# ---------------------------------------------------------------------------
# Saved answers
# ---------------------------------------------------------------------------


def save_profile(path: Path, taxpayer: Taxpayer, documents: list[SourceDocument]) -> None:
    payload = {
        "taxpayer": taxpayer.model_dump(mode="json"),
        "documents": [d.model_dump(mode="json") for d in documents],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_DOC_TYPES = {
    "w2": W2,
    "1099-int": Form1099INT,
    "1099-div": Form1099DIV,
    "1099-nec": Form1099NEC,
}


def load_profile(path: Path) -> tuple[dict[str, Any], list[SourceDocument]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = [
        _DOC_TYPES[entry["kind"]].model_validate(entry)
        for entry in payload.get("documents", [])
    ]
    return payload.get("taxpayer", {}), documents


def summarise(taxpayer: Taxpayer, documents: list[SourceDocument]) -> None:
    rule("What I have")
    typer.echo("")
    name = f"{taxpayer.first_name} {taxpayer.last_name}"
    if taxpayer.spouse_first_name:
        name += f" and {taxpayer.spouse_first_name} {taxpayer.spouse_last_name}"
    typer.secho(f"   {name}", fg=BOLD)
    typer.echo(f"   {mask_ssn(taxpayer.ssn)}   {taxpayer.filing_status.value.replace('_', ' ')}")
    address = taxpayer.address
    line2 = f"{address.city}, {address.state} {address.zip_code}"
    typer.echo(f"   {address.street}{', ' + address.apartment if address.apartment else ''}")
    typer.echo(f"   {line2}")
    typer.echo("")
    for document in documents:
        typer.echo(f"   * {describe(document)}")
