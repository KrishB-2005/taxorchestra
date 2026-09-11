"""Turn semantic values into AcroForm field values, for any mapped form.

The mapping agent works on any fillable IRS PDF, so placing values does too.
What does *not* generalise is knowing which values are correct — that is tax
logic, and it is written per form. This module handles the last mile: given a
catalog and a `{semantic_key: value}` dict, produce `{acro_name: text}` ready
for `acroform.fill`.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from taxorchestra.forms import acroform
from taxorchestra.models import FieldCatalog, FieldType

DOLLAR = Decimal("1")


def format_value(
    field_type: FieldType,
    raw: object,
    on_states: tuple[str, ...] = (),
) -> str | None:
    """Render one value for one field, or None to leave the box empty."""
    if raw is None:
        return None

    if field_type is FieldType.CHECKBOX:
        if not raw:
            return None
        # A checkbox takes the name of an appearance stream ("/1", "/Yes"),
        # not a boolean, and the name differs per field.
        return on_states[0] if on_states else "/Yes"

    if field_type is FieldType.SSN:
        # SSN boxes are comb fields with /MaxLen 9, so they take bare digits.
        # pypdf writes an over-length value without complaint and it renders as
        # overflow; pdf-lib refuses outright, which is how this surfaced.
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        return digits if len(digits) == 9 else None

    if field_type is FieldType.MONEY:
        try:
            amount = raw if isinstance(raw, Decimal) else Decimal(str(raw))
        except InvalidOperation:
            return None
        # Filed in whole dollars, and a zero line is left blank — writing "0"
        # on every unused line makes a filled form unreadable.
        if amount == 0:
            return None
        return f"{amount.quantize(DOLLAR):,}"

    text = str(raw).strip()
    return text or None


def render_values(
    catalog: FieldCatalog,
    values: dict[str, object],
    template_pdf: str,
) -> tuple[dict[str, str], list[str]]:
    """Map `{semantic_key: value}` onto `{acro_name: text}`.

    Returns the writable values and the keys that had no field in the catalog.
    An unknown key is reported rather than silently dropped: on a tax form, a
    value that quietly fails to land is worse than one that fails loudly.
    """
    by_key = catalog.by_key()
    on_states = {w.acro_name: w.on_states for w in acroform.load_widgets(template_pdf)}

    rendered: dict[str, str] = {}
    unknown: list[str] = []

    for key, raw in values.items():
        mapping = by_key.get(key)
        if mapping is None:
            unknown.append(key)
            continue
        text = format_value(
            mapping.field_type, raw, on_states.get(mapping.acro_name, ())
        )
        if text is not None:
            rendered[mapping.acro_name] = text

    return rendered, unknown
