"""Agent 2 — FormMapping: cryptic AcroForm names to semantic keys.

The 1040 gives us 229 fields called `f1_01[0]` … `f2_38[0]`, no tooltips, no
schema. Resolution runs in four phases, cheapest first, each one narrowing the
work left for the next. Only the fourth phase costs a model call, and its
result is cached against the form's digest.

  1. geometric   Join each widget to the text on its baseline, to its left.
                 Recovers the label for most numbered money lines outright.
  2. structural  Repair what geometry cannot see: a continuation row's label is
                 the bare letter "b" because the "1" was printed one row up, and
                 section headings only appear on the row that introduces them.
  3. caption     For widgets with nothing to their left — the identity block —
                 take the caption printed above the box instead.
  4. semantic    Hand the recovered label to the model and get back a canonical
                 snake_case key and a field type.

Phases 1-3 are deterministic and free. Phase 4 runs once per distinct label and
is what the cache stores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from taxorchestra.cache.store import MappingCache
from taxorchestra.forms import acroform
from taxorchestra.llm.client import LLMClient, LLMError
from taxorchestra.models import FieldCatalog, FieldMapping, FieldType

# A widget whose semantic key the model is under this confident about is left
# unresolved rather than guessed at. A wrong mapping writes a real number into
# the wrong box on a tax form, which is worse than an empty box.
MIN_CONFIDENCE = 0.55

_MAJOR_LINE = re.compile(r"^(\d{1,2})([a-z]?)$")
_BARE_LETTER = re.compile(r"^[a-z]$")

# A handful of boxes on the 1040 have a printed caption that is simply absent
# from the PDF's text layer — "ZIP code" is drawn but produces no extractable
# text run, so no amount of geometry will find it. Rather than let the model
# guess at an empty label, the form's few known gaps are stated outright.
# Keyed by the leaf AcroForm name; verified against the 2025 revision.
CAPTION_GAPS: dict[str, str] = {
    "f1_24[0]": "ZIP code",
    # Line 16's amount box sits in the money column, but its label "16 Tax" is
    # separated from it by the Form 8814/4972 checkboxes. The column clipping
    # that stops line 2b stealing 2a's label also cuts this one off, and
    # loosening the clip would reintroduce the worse bug. Stated outright
    # instead. (The small box at f2_07 on the same row is the "other form"
    # name field, which this pipeline does not fill.)
    "f2_08[0]": "16 Tax",
}


class MappingProposal(BaseModel):
    """What the model returns for one field. Deliberately tiny."""

    semantic_key: str | None = Field(
        description=(
            "Stable snake_case identifier for what this field holds, e.g. "
            "'wages_w2_box1'. Null if the label is too ambiguous to name."
        )
    )
    field_type: Literal["text", "money", "checkbox", "ssn", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One sentence explaining the choice.")


SYSTEM_PROMPT = """\
You map fields on IRS Form 1040 to stable identifiers.

You receive the label text recovered from around a single form field, plus its \
line number and section heading when those were found. Return a snake_case \
`semantic_key` naming what the field holds.

Rules:
- The key describes the *content*, not the position: `wages_w2_box1`, not `line_1a`.
- Reuse the obvious noun. Wages are `wages_*`, interest `interest_*`, dividends \
`dividends_*`, withholding `withholding_*`, totals `*_total`.
- `field_type` is `money` for dollar amounts, `ssn` for social security numbers, \
`checkbox` for tick boxes, `text` otherwise.
- If the label is too fragmentary to name confidently, return null with a low \
confidence. A null is cheap; a wrong mapping writes a number into the wrong box.
"""


@dataclass
class MappingStats:
    """Per-phase attribution, so the pipeline can be reasoned about."""

    total: int = 0
    phase_counts: dict[int, int] = None  # type: ignore[assignment]
    llm_calls: int = 0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.phase_counts is None:
            self.phase_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    @property
    def resolved(self) -> int:
        return sum(self.phase_counts.values())


class FormMappingAgent:
    def __init__(self, llm: LLMClient, cache: MappingCache | None = None) -> None:
        self.llm = llm
        self.cache = cache
        self.stats = MappingStats()

    # -- public ----------------------------------------------------------

    def build_catalog(self, pdf_path: str, *, use_cache: bool = True) -> FieldCatalog:
        digest = acroform.pdf_sha256(pdf_path)

        if use_cache and self.cache is not None:
            cached = self.cache.get(digest)
            if cached is not None:
                self.stats.cache_hit = True
                return FieldCatalog.model_validate(cached)

        catalog = self._resolve(pdf_path, digest)

        if use_cache and self.cache is not None:
            self.cache.put(digest, catalog.model_dump(mode="json"))
        return catalog

    # -- phases ----------------------------------------------------------

    def _resolve(self, pdf_path: str, digest: str) -> FieldCatalog:
        widgets = acroform.load_widgets(pdf_path)
        runs_by_page = {
            page: acroform.load_text_runs(pdf_path, page)
            for page in sorted({w.page for w in widgets})
        }

        mappings: list[FieldMapping] = []
        for page in sorted(runs_by_page):
            page_widgets = [w for w in widgets if w.page == page]
            # Reading order: down the page, then across.
            page_widgets.sort(key=lambda w: (-w.center_y, w.x0))
            mappings.extend(self._resolve_page(page_widgets, runs_by_page[page], page))

        self.stats.total = len(mappings)
        return FieldCatalog(
            form_id="f1040",
            form_year=acroform.detect_form_year(pdf_path) or 0,
            pdf_sha256=digest,
            mappings=mappings,
        )

    def _resolve_page(
        self,
        widgets: list[acroform.Widget],
        runs: list[acroform.TextRun],
        page: int,
    ) -> list[FieldMapping]:
        drafts: list[FieldMapping] = []

        # ---- phase 1: geometric -------------------------------------
        # Clip each widget's label search at its left-hand neighbour on the
        # row, so two-column lines ("2a … b …") don't bleed into each other.
        bounds = acroform.row_left_bounds(widgets)
        for widget in widgets:
            label = acroform.recover_label(widget, runs, bounds[widget.acro_name])
            drafts.append(
                FieldMapping(
                    acro_name=widget.acro_name,
                    page=page,
                    line=label.line,
                    label_text=label.text,
                    section=label.section,
                    field_type=self._type_from_widget(widget),
                    phase=1 if label.text else 0,
                )
            )

        # ---- phase 2: structural ------------------------------------
        # Carry the last major line number and section down the page, so a
        # continuation row labelled just "b" becomes "1b" under "Income".
        last_major: str | None = None
        last_section: str | None = None
        for draft in drafts:
            if draft.line:
                match = _MAJOR_LINE.match(draft.line)
                if match and match.group(1):
                    last_major = match.group(1)
                elif _BARE_LETTER.match(draft.line) and last_major:
                    draft.line = f"{last_major}{draft.line}"
                    if draft.phase == 0:
                        draft.phase = 2
            if draft.section:
                last_section = draft.section
            elif last_section:
                draft.section = last_section

        # ---- phase 3: caption above ---------------------------------
        for widget, draft in zip(widgets, drafts, strict=True):
            if draft.label_text:
                continue
            caption = acroform.recover_caption_above(widget, runs)
            if caption is None:
                caption = CAPTION_GAPS.get(widget.acro_name.split(".")[-1])
            if caption:
                draft.label_text = caption
                draft.phase = 3

        # ---- phase 4: semantic --------------------------------------
        # One model call per *distinct* label, not per field: the 1040 repeats
        # labels across columns, and identical text has identical meaning.
        proposals: dict[str, MappingProposal] = {}
        for draft in drafts:
            if not draft.label_text:
                continue
            prompt = self._prompt_for(draft)
            if prompt not in proposals:
                proposals[prompt] = self._propose(prompt)
                self.stats.llm_calls += 1
            proposal = proposals[prompt]

            if proposal.semantic_key and proposal.confidence >= MIN_CONFIDENCE:
                draft.semantic_key = proposal.semantic_key
                draft.confidence = proposal.confidence
                if draft.field_type is FieldType.UNKNOWN:
                    draft.field_type = FieldType(proposal.field_type)

        # `phase` records which phase recovered the *label*; phase 4 then names
        # every one of them. Attributing all resolutions to phase 4 would hide
        # how much work the free deterministic phases actually do.
        for draft in drafts:
            if draft.resolved:
                self.stats.phase_counts[draft.phase] = (
                    self.stats.phase_counts.get(draft.phase, 0) + 1
                )
        return drafts

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _type_from_widget(widget: acroform.Widget) -> FieldType:
        # /Btn is unambiguous from the PDF itself; text vs money needs the label.
        return FieldType.CHECKBOX if widget.field_type == "/Btn" else FieldType.UNKNOWN

    @staticmethod
    def _prompt_for(draft: FieldMapping) -> str:
        parts = [f"Label: {draft.label_text}"]
        if draft.line:
            parts.append(f"Line: {draft.line}")
        if draft.section:
            parts.append(f"Section: {draft.section}")
        if draft.field_type is FieldType.CHECKBOX:
            parts.append("This field is a checkbox.")
        return "\n".join(parts)

    def _propose(self, prompt: str) -> MappingProposal:
        try:
            return self.llm.parse(
                system=SYSTEM_PROMPT,
                user=prompt,
                schema=MappingProposal,
                cache_system=True,
            )
        except LLMError:
            # A model failure degrades one field to unresolved; it does not
            # abort the filing. The validation agent will flag the gap.
            return MappingProposal(
                semantic_key=None,
                field_type="unknown",
                confidence=0.0,
                rationale="mapping call failed",
            )
