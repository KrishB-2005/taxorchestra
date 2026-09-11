"""Low-level AcroForm access for the IRS Form 1040 PDF.

The 1040 ships 229 form fields named `f1_01[0]` … `f2_38[0]` with **no** `/TU`
tooltips and no other semantic metadata — verified against the 2025 form. The
only thing tying a field to its meaning is *where it sits on the page*: the
label "1a  Total amount from Form(s) W-2, box 1" is a text run to the left of
the widget rectangle on the same baseline.

This module supplies the geometry. Turning recovered label fragments into
stable semantic keys is the mapping agent's job.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject

# Vertical tolerance when deciding whether a text run shares a widget's
# baseline. The 1040 sets body copy on a ~12pt grid, so 6pt keeps us inside
# one row without bleeding into the row above or below.
BASELINE_TOLERANCE_PT = 6.0

# A label that starts further left than this is a section heading spanning the
# page ("Income", "Payments"), not this row's own label.
SECTION_COLUMN_MAX_X = 60.0

_LINE_LABEL = re.compile(r"^\d{1,2}[a-z]?$|^[a-z]$")


@dataclass(frozen=True)
class Widget:
    """One fillable field, located on the page."""

    acro_name: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    field_type: str  # raw /FT: /Tx, /Btn, /Ch
    on_states: tuple[str, ...] = ()

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class TextRun:
    x: float
    y: float
    text: str


@dataclass
class RecoveredLabel:
    """Label fragments found around a widget, split by role."""

    line: str | None = None
    section: str | None = None
    fragments: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(self.fragments).strip()


def _qualified_name(obj: DictionaryObject) -> str:
    """Walk the /Parent chain to build the fully qualified field name.

    A widget's own /T is only the leaf (`f1_47[0]`); `get_fields()` reports the
    dotted path, and the fill API keys off that path, so the two must agree.
    """
    parts: list[str] = []
    node: DictionaryObject | None = obj
    seen: set[int] = set()
    while node is not None:
        if id(node) in seen:  # defensive: malformed PDFs can self-reference
            break
        seen.add(id(node))
        title = node.get("/T")
        if title is not None:
            parts.append(str(title))
        parent = node.get("/Parent")
        node = parent.get_object() if parent is not None else None
    return ".".join(reversed(parts))


def _inherited(obj: DictionaryObject, key: str):
    node: DictionaryObject | None = obj
    while node is not None:
        if key in node:
            return node[key]
        parent = node.get("/Parent")
        node = parent.get_object() if parent is not None else None
    return None


def _on_states(obj: DictionaryObject) -> tuple[str, ...]:
    """Checkbox "on" values, read from the appearance dictionary.

    A checkbox is not set with `True` — it takes the name of one of its
    appearance streams (`/1`, `/Yes`, …), which differs per field.
    """
    ap = obj.get("/AP")
    if ap is None:
        return ()
    normal = ap.get_object().get("/N")
    if normal is None:
        return ()
    return tuple(str(k) for k in normal.get_object().keys() if str(k) != "/Off")


def load_widgets(pdf_path: str | Path) -> list[Widget]:
    """Every fillable widget in the document, with page-space rectangles."""
    reader = PdfReader(str(pdf_path))
    widgets: list[Widget] = []

    for page_index, page in enumerate(reader.pages):
        annots = page.get("/Annots")
        if annots is None:
            continue
        for ref in annots.get_object():
            obj = ref.get_object()
            if obj.get("/Subtype") != "/Widget":
                continue
            rect = obj.get("/Rect")
            if rect is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in rect)
            ftype = _inherited(obj, "/FT")
            widgets.append(
                Widget(
                    acro_name=_qualified_name(obj),
                    page=page_index,
                    x0=min(x0, x1),
                    y0=min(y0, y1),
                    x1=max(x0, x1),
                    y1=max(y0, y1),
                    field_type=str(ftype) if ftype else "",
                    on_states=_on_states(obj),
                )
            )
    return widgets


def load_text_runs(pdf_path: str | Path, page_index: int) -> list[TextRun]:
    """Text on one page, each run tagged with its device-space origin."""
    reader = PdfReader(str(pdf_path))
    runs: list[TextRun] = []

    def visitor(text, _cm, tm, _font_dict, _font_size) -> None:
        stripped = text.strip()
        if stripped:
            runs.append(TextRun(x=float(tm[4]), y=float(tm[5]), text=stripped))

    reader.pages[page_index].extract_text(visitor_text=visitor)
    return runs


def recover_label(
    widget: Widget,
    runs: list[TextRun],
    left_bound: float = 0.0,
) -> RecoveredLabel:
    """Recover the human-readable label for a widget from surrounding text.

    Text on the widget's baseline, between `left_bound` and the widget's left
    edge, ordered left to right. The leftmost fragment in the section column is
    the section heading; a short fragment such as "1a" or "b" is the line label;
    the rest is prose.

    `left_bound` matters because the 1040 puts two entries on one row —
    "2a Tax-exempt interest ___  b Taxable interest ___". Without a bound the
    right-hand box collects the left-hand box's label too. Callers pass the
    right edge of the preceding widget on the same row.
    """
    same_row = [
        r
        for r in runs
        if abs(r.y - widget.center_y) <= BASELINE_TOLERANCE_PT
        and left_bound <= r.x < widget.x0
    ]
    same_row.sort(key=lambda r: r.x)

    label = RecoveredLabel()
    for run in same_row:
        text = run.text
        if run.x <= SECTION_COLUMN_MAX_X and not _LINE_LABEL.match(text):
            label.section = text
            continue
        if _LINE_LABEL.match(text):
            # "1" and "a" arrive as separate runs and should join into "1a",
            # but only once: a row holding "2a … b …" must not build "2ab".
            if label.line is None:
                label.line = text
            elif label.line.isdigit() and text.isalpha():
                label.line += text
            continue
        # Leader dots pad the label out to the column rule; they are noise.
        label.fragments.append(text.rstrip(". ").strip())

    label.fragments = [f for f in label.fragments if f]
    return label


def row_left_bounds(widgets: list[Widget]) -> dict[str, float]:
    """Left clipping bound for each widget, from its neighbours on the row.

    Keyed by AcroForm name. A widget with nothing to its left on the row gets
    0.0, so the section column is still reachable.
    """
    bounds: dict[str, float] = {}
    for widget in widgets:
        same_row = [
            other
            for other in widgets
            if other is not widget
            and abs(other.center_y - widget.center_y) <= BASELINE_TOLERANCE_PT
            and other.x1 <= widget.x0
        ]
        bounds[widget.acro_name] = max((o.x1 for o in same_row), default=0.0)
    return bounds


def recover_caption_above(
    widget: Widget,
    runs: list[TextRun],
    max_rise_pt: float = 22.0,
    x_slack_pt: float = 40.0,
) -> str | None:
    """Fallback for widgets whose label sits *above* them, not to the left.

    The 1040's identity block is laid out as captioned boxes — "Your first name
    and middle initial" is a caption over `f1_14[0]`, with nothing to its left.
    The baseline join finds nothing for these; this finds the nearest caption
    starting within `x_slack_pt` of the widget's left edge.
    """
    candidates = [
        r
        for r in runs
        if not (r.x == 0.0 and r.y == 0.0)  # origin runs are extraction artefacts
        and widget.y1 < r.y <= widget.y1 + max_rise_pt
        and (widget.x0 - x_slack_pt) <= r.x <= widget.x1
    ]
    if not candidates:
        return None
    # Nearest vertically, then leftmost.
    candidates.sort(key=lambda r: (r.y - widget.y1, r.x))
    return candidates[0].text.rstrip(". ").strip() or None


# Two widgets belong to the same table column when their left and right edges
# agree to within this much. The 1040 family draws columns to the point.
COLUMN_EDGE_TOLERANCE_PT = 2.0

# A column needs at least this many cells before it is treated as a table
# rather than a coincidence of two boxes sharing an x position.
MIN_COLUMN_CELLS = 3


def group_columns(widgets: list[Widget]) -> list[list[Widget]]:
    """Group widgets into table columns, ordered top to bottom within each.

    Schedules B and C and Form 8812 are mostly tables: a stack of identical
    boxes with no per-row label, because the label is a column header printed
    once at the top. Row-wise geometry finds nothing for these, which is why
    Schedule B recovers a label for only 31% of its widgets before this.
    """
    buckets: dict[tuple[float, float], list[Widget]] = {}
    for widget in widgets:
        key = (
            round(widget.x0 / COLUMN_EDGE_TOLERANCE_PT),
            round(widget.x1 / COLUMN_EDGE_TOLERANCE_PT),
        )
        buckets.setdefault(key, []).append(widget)

    columns = []
    for cells in buckets.values():
        if len(cells) >= MIN_COLUMN_CELLS:
            columns.append(sorted(cells, key=lambda w: -w.center_y))
    return columns


def recover_column_header(
    column: list[Widget],
    runs: list[TextRun],
    max_rise_pt: float = 42.0,
) -> str | None:
    """The header printed above a table column, if there is one.

    Searched within the column's own horizontal span so that marginal
    instruction prose — which the 1040 schedules set at x=36, well left of any
    data column — cannot be mistaken for a header.
    """
    if not column:
        return None
    top = column[0]
    candidates = [
        r
        for r in runs
        if not (r.x == 0.0 and r.y == 0.0)
        and top.y1 < r.y <= top.y1 + max_rise_pt
        and (top.x0 - 20.0) <= r.x <= top.x1
    ]
    if not candidates:
        return None
    # Nearest above wins; ties break leftmost.
    candidates.sort(key=lambda r: (r.y - top.y1, r.x))
    header = candidates[0].text.rstrip(". ").strip()
    # A bare line number is the row marker, not a header.
    return header if header and not _LINE_LABEL.match(header) else None


def pdf_sha256(pdf_path: str | Path) -> str:
    """Digest of the blank form.

    The cache is keyed on this: the IRS reissues the 1040 every year and field
    names shift, so a mapping learned for one revision must not be reused for
    another.
    """
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()


def detect_form_year(pdf_path: str | Path) -> int | None:
    """Read the tax year off the form's masthead."""
    runs = load_text_runs(pdf_path, 0)
    header = [r for r in runs if r.y > 740.0]
    header.sort(key=lambda r: r.x)
    joined = "".join(r.text for r in header)
    match = re.search(r"20\d{2}", joined)
    return int(match.group()) if match else None


def fill(
    template_path: str | Path,
    output_path: str | Path,
    values: dict[str, str],
) -> int:
    """Write `values` (keyed by fully qualified AcroForm name) into the form.

    Returns the number of fields actually written. `NeedAppearances` tells the
    viewer to render appearance streams for the values we set, which is what
    makes them visible in Preview/Acrobat rather than only present in the data.
    """
    reader = PdfReader(str(template_path))
    writer = PdfWriter(clone_from=reader)
    writer.set_need_appearances_writer(True)

    known = set(reader.get_fields() or {})
    unknown = set(values) - known
    if unknown:
        raise KeyError(
            f"{len(unknown)} field(s) not present in the template, refusing to write: "
            + ", ".join(sorted(unknown)[:5])
        )

    written = 0
    for page in writer.pages:
        annots = page.get("/Annots")
        if annots is None:
            continue
        page_targets = {}
        for ref in annots.get_object():
            obj = ref.get_object()
            if obj.get("/Subtype") != "/Widget":
                continue
            name = _qualified_name(obj)
            if name in values:
                page_targets[name] = values[name]
        if page_targets:
            writer.update_page_form_field_values(page, page_targets, auto_regenerate=False)
            written += len(page_targets)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        writer.write(fh)
    return written


def read_back(pdf_path: str | Path) -> dict[str, str]:
    """Values actually stored in a filled PDF.

    Used by the benchmark: writing a field is not proof it landed, so the
    accuracy number is computed by reopening the output and reading it. An
    unchecked checkbox carries `/Off`, which is the absence of a value rather
    than a value, so it is excluded — otherwise a blank form would score as
    having ~70 fields filled.
    """
    reader = PdfReader(str(pdf_path))
    fields = reader.get_fields() or {}
    out: dict[str, str] = {}
    for name, meta in fields.items():
        value = meta.get("/V")
        if value is None:
            continue
        text = str(value)
        if text.startswith("/"):
            text = text[1:]
        if text in ("", "Off"):
            continue
        out[name] = text
    return out
