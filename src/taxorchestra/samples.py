"""Synthetic W-2 and 1099 source documents.

Real tax documents contain real people's wages and social security numbers, so
the test corpus is generated rather than collected. Everything here is fake:
the SSNs use the 900-range the SSA never issues, and the EINs are invalid by
construction.

The PDF writer is ~60 lines of the format written by hand rather than a
dependency. A project whose entire premise is reading and writing AcroForms can
afford to know what a content stream looks like, and it keeps `pip install
taxorchestra` free of a rendering library that only the fixtures need.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from taxorchestra.models import W2, Form1099INT

PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEADING = 18
LEFT_MARGIN = 64
TOP = 720


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def write_text_pdf(path: str | Path, lines: list[tuple[str, int]]) -> Path:
    """Write a single-page PDF containing `lines` of (text, font_size).

    Builds the object table by hand and records byte offsets for the xref, which
    is the part a naive generator gets wrong — a bad xref makes the file open in
    some viewers and fail in others.
    """
    content_parts = ["BT"]
    y = TOP
    for text, size in lines:
        content_parts.append(f"/F1 {size} Tf")
        content_parts.append(f"1 0 0 1 {LEFT_MARGIN} {y} Tm")
        content_parts.append(f"({_escape(text)}) Tj")
        y -= LEADING
    content_parts.append("ET")
    content = "\n".join(content_parts).encode("latin-1")

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(out))
    return target


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def write_w2_pdf(path: str | Path, w2: W2) -> Path:
    """Render a W-2 as a labelled key/value sheet.

    Not a facsimile of the real form — a legible document with the box numbers
    the extractor keys off, which is what the pipeline actually needs.
    """
    return write_text_pdf(
        path,
        [
            ("Form W-2  Wage and Tax Statement", 14),
            ("", 10),
            (f"Employer: {w2.employer_name}", 11),
            (f"Employer EIN: {w2.employer_ein}", 11),
            (f"Employee SSN: {w2.employee_ssn}", 11),
            ("", 10),
            (f"Box 1 - Wages, tips, other compensation: {_money(w2.wages)}", 11),
            (
                f"Box 2 - Federal income tax withheld: "
                f"{_money(w2.federal_income_tax_withheld)}",
                11,
            ),
            (
                f"Box 3 - Social security wages: "
                f"{_money(w2.social_security_wages or Decimal(0))}",
                11,
            ),
            (
                f"Box 4 - Social security tax withheld: "
                f"{_money(w2.social_security_tax_withheld or Decimal(0))}",
                11,
            ),
            (f"Box 5 - Medicare wages and tips: {_money(w2.medicare_wages or Decimal(0))}", 11),
            (
                f"Box 6 - Medicare tax withheld: "
                f"{_money(w2.medicare_tax_withheld or Decimal(0))}",
                11,
            ),
        ],
    )


def write_1099int_pdf(path: str | Path, form: Form1099INT) -> Path:
    return write_text_pdf(
        path,
        [
            ("Form 1099-INT  Interest Income", 14),
            ("", 10),
            (f"Payer: {form.payer_name}", 11),
            (f"Payer TIN: {form.payer_tin}", 11),
            ("", 10),
            (f"Box 1 - Interest income: {_money(form.interest_income)}", 11),
            (
                f"Box 4 - Federal income tax withheld: "
                f"{_money(form.federal_income_tax_withheld)}",
                11,
            ),
            (f"Box 8 - Tax-exempt interest: {_money(form.tax_exempt_interest)}", 11),
        ],
    )


# --------------------------------------------------------------------------
# The bundled scenario
# --------------------------------------------------------------------------

SAMPLE_SSN = "900-12-3456"  # 900-range: never issued by the SSA

SAMPLE_W2 = W2(
    employer_name="Northwind Analytics LLC",
    employer_ein="00-1234567",
    employee_ssn=SAMPLE_SSN,
    wages=Decimal("92450.00"),
    federal_income_tax_withheld=Decimal("11894.00"),
    social_security_wages=Decimal("92450.00"),
    social_security_tax_withheld=Decimal("5731.90"),
    medicare_wages=Decimal("92450.00"),
    medicare_tax_withheld=Decimal("1340.53"),
)

SAMPLE_1099INT = Form1099INT(
    payer_name="Cedar Ridge Savings Bank",
    payer_tin="00-7654321",
    interest_income=Decimal("1284.37"),
    tax_exempt_interest=Decimal("0.00"),
    federal_income_tax_withheld=Decimal("0.00"),
)


def write_sample_documents(directory: str | Path) -> list[Path]:
    """Write the bundled scenario to `directory`, returning the file paths."""
    target = Path(directory)
    return [
        write_w2_pdf(target / "w2-northwind.pdf", SAMPLE_W2),
        write_1099int_pdf(target / "1099int-cedar-ridge.pdf", SAMPLE_1099INT),
    ]
