"""LLM access, behind one narrow interface.

Every agent needs exactly one thing from a model: *given a system prompt and a
user prompt, return an instance of this Pydantic type*. Constraining the
interface to that keeps free-form text out of the pipeline and lets the whole
system run on a deterministic stand-in when no credentials are present.

Three providers:

  anthropic  `client.messages.parse(..., output_format=Model)` against the
             Claude API. The default.
  bedrock    The same call surface via `AnthropicBedrockMantle`, for shops that
             need inference inside their own AWS account.
  fixture    No network. Deterministic rules that are good enough to drive the
             full pipeline, so `pytest` and `--provider fixture` work on a
             laptop with no key and no spend.

`fixture` is a test double, not a small model. It never guesses: when a rule
does not fire it returns low confidence and lets the caller decide.
"""

from __future__ import annotations

import os
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-opus-5"

# Non-streaming default from the Anthropic SDK guidance; every call here
# returns a small JSON object, so this is headroom rather than a target.
MAX_TOKENS = 16000


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    """The only thing an agent may ask of a model."""

    name: str

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        cache_system: bool = False,
        pdf: bytes | None = None,
    ) -> T: ...


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Claude API via the official SDK."""

    name = "anthropic"

    def __init__(self, model: str | None = None, effort: str = "high") -> None:
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - install-time
            raise LLMError(
                "provider 'anthropic' needs the SDK: pip install 'taxorchestra[llm]'"
            ) from exc

        self._anthropic = anthropic
        # Zero-arg constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
        # or an `ant auth login` profile, in that order.
        self._client = anthropic.Anthropic()
        self.model = model or os.getenv("TAXORCHESTRA_MODEL", DEFAULT_MODEL)
        self.effort = effort

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        cache_system: bool = False,
        pdf: bytes | None = None,
    ) -> T:
        if pdf is None:
            content: Any = user
        else:
            # The document block goes before the instruction text; base64 must
            # carry no newlines.
            import base64

            content = [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": base64.standard_b64encode(pdf).decode("ascii"),
                    },
                },
                {"type": "text", "text": user},
            ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_format": schema,
            # Adaptive thinking is the current API on Opus 5; `budget_tokens`
            # is rejected there. Effort is the depth dial.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if cache_system:
            # The field catalog prompt is a large, byte-stable prefix reused
            # across every field in a run — exactly what caching is for.
            kwargs["cache_control"] = {"type": "ephemeral"}

        try:
            response = self._client.messages.parse(**kwargs)
        except self._anthropic.BadRequestError as exc:
            raise LLMError(f"malformed request: {exc}") from exc
        except self._anthropic.AuthenticationError as exc:
            raise LLMError("authentication failed — check ANTHROPIC_API_KEY") from exc
        except self._anthropic.RateLimitError as exc:
            raise LLMError("rate limited; the SDK already retried") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError(f"network error reaching the Claude API: {exc}") from exc

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            category = getattr(detail, "category", None)
            raise LLMError(f"model declined the request (category={category})")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("model returned no parseable output")
        return parsed


# ---------------------------------------------------------------------------
# Amazon Bedrock
# ---------------------------------------------------------------------------


class BedrockClient(AnthropicClient):
    """Same call surface, inference inside the caller's AWS account."""

    name = "bedrock"

    def __init__(self, model: str | None = None, effort: str = "high") -> None:
        try:
            import anthropic
            from anthropic import AnthropicBedrockMantle
        except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover
            raise LLMError(
                "provider 'bedrock' needs: pip install 'taxorchestra[llm,aws]'"
            ) from exc

        self._anthropic = anthropic
        self._client = AnthropicBedrockMantle(
            aws_region=os.getenv("AWS_REGION", "us-east-1")
        )
        self.model = model or os.getenv("TAXORCHESTRA_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)
        self.effort = effort


# ---------------------------------------------------------------------------
# Deterministic stand-in
# ---------------------------------------------------------------------------

# Ordered most specific first: "wages" alone should not win over
# "federal income tax withheld" on a label containing both.
_LABEL_RULES: list[tuple[str, str]] = [
    # -- identity -----------------------------------------------------
    (r"if joint return, spouse.s first name", "spouse_first_name"),
    (r"your first name and middle initial", "taxpayer_first_name"),
    (r"spouse.s social security number", "spouse_ssn"),
    (r"your social security number", "taxpayer_ssn"),
    (r"home address", "address_street"),
    (r"apt\.? no", "address_apartment"),
    (r"city, town, or post office", "address_city"),
    (r"foreign province", "address_foreign_province"),
    (r"foreign country name", "address_foreign_country"),
    (r"^state$", "address_state"),
    (r"zip code", "address_zip"),
    (r"^\(1\) first name$", "dependent_first_name"),
    (r"^\(2\) last name$", "dependent_last_name"),
    (r"^last name$", "taxpayer_last_name"),
    # -- filing status (checkboxes) -----------------------------------
    (r"^single$", "filing_status_single"),
    (r"married filing jointly", "filing_status_married_jointly"),
    (r"married filing separately", "filing_status_married_separately"),
    (r"head of household", "filing_status_head_of_household"),
    (r"qualifying surviving spouse", "filing_status_qualifying_surviving_spouse"),
    # -- income -------------------------------------------------------
    (r"total amount from form\(s\) w-?2, box 1", "wages_w2_box1"),
    (r"household employee wages", "wages_household_employee"),
    (r"tip income not reported", "wages_unreported_tips"),
    (r"medicaid waiver payments", "wages_medicaid_waiver"),
    (r"dependent care benefits", "wages_dependent_care"),
    (r"employer-provided adoption", "wages_adoption_benefits"),
    (r"add lines 1a through 1h", "wages_total"),
    (r"tax-exempt interest", "interest_tax_exempt"),
    (r"taxable interest", "interest_taxable"),
    (r"qualified dividends", "dividends_qualified"),
    (r"ordinary dividends", "dividends_ordinary"),
    (r"this is your total income", "total_income"),
    (r"this is your adjusted gross income", "adjusted_gross_income"),
    # -- deductions ---------------------------------------------------
    (r"standard deduction or itemized", "deduction_standard_or_itemized"),
    (r"qualified business income", "deduction_qbi"),
    (r"add lines 12e, 13a, and 13b", "deduction_total"),
    (r"this is your taxable income", "taxable_income"),
    # -- tax, payments, settlement ------------------------------------
    (r"^16 tax", "tax_computed"),
    (r"this is your total tax", "tax_total"),
    (r"federal income tax withheld from.*w-?2", "withholding_w2"),
    (r"^form\(s\) w-?2$", "withholding_w2"),
    (r"^form\(s\) 1099$", "withholding_1099"),
    (r"add lines 25a through 25c", "withholding_total"),
    (r"these are your total payments", "payments_total"),
    (r"this is the amount you overpaid", "refund_overpaid"),
    (r"amount of line 34 you want refunded", "refund_requested"),
    (r"amount of line 34 you want applied", "refund_applied_next_year"),
    # Line 37's own sentence sits on a different row from its box; the text
    # actually on the box's baseline is the payment-instructions note.
    (r"for details on how to pay", "amount_owed"),
    (r"subtract line 33 from line 24", "amount_owed"),
    (r"estimated tax penalty", "estimated_tax_penalty"),
    # -- generic, last so anything 1040-specific above still wins ------
    # These are column headers and captions that recur across the whole form
    # family, which is what lets the fixture demonstrate a schedule at all. A
    # real provider names arbitrary labels and needs none of this.
    (r"^amount$", "amount"),
    (r"^list name of payer", "payer_name"),
    (r"^name of payer", "payer_name"),
    (r"^description", "description"),
    (r"^name", "name"),
    (r"^address", "address"),
    (r"^social security number", "ssn"),
    (r"^employer id", "ein"),
    (r"^date", "date"),
]

# The semantic key is a better type signal than the label: line 25a's printed
# label is just "Form(s) W-2", which contains no money word at all, but the key
# `withholding_w2` is unambiguous. Getting this wrong writes "0" onto a line the
# form expects to be left blank.
_MONEY_KEY_PREFIXES = (
    "wages_",
    "interest_",
    "dividends_",
    "withholding_",
    "payments_",
    "deduction_",
    "refund_",
    "tax_",
    "other_income",
    "total_income",
    "adjusted_gross_income",
    "taxable_income",
    # "amount" subsumes amount_owed, and covers the bare "Amount" column
    # header that recurs across the schedules.
    "amount",
    "estimated_tax_penalty",
)


class FixtureClient:
    """Deterministic stand-in so the pipeline runs with no credentials.

    Dispatches on the requested schema. Anything it cannot resolve comes back
    with confidence 0.0, which the mapping agent treats as unresolved — the
    same path a real model's low-confidence answer takes.
    """

    name = "fixture"

    def __init__(self, model: str | None = None, effort: str = "high") -> None:
        self.model = model or "fixture-deterministic"
        self.effort = effort
        self.calls = 0

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        cache_system: bool = False,
        pdf: bytes | None = None,
    ) -> T:
        self.calls += 1
        handler = getattr(self, f"_handle_{schema.__name__}", None)
        if handler is None:
            raise LLMError(
                f"fixture provider has no rule for {schema.__name__}; "
                "run with --provider anthropic for this path"
            )
        return handler(user)

    # -- per-schema rules -------------------------------------------------

    def _handle_MappingProposal(self, user: str) -> Any:  # noqa: N802
        from taxorchestra.agents.mapping import MappingProposal

        # Match the label line on its own. Several rules are anchored ("^state$",
        # "^last name$") to avoid colliding with longer labels that contain the
        # same words, and anchors only mean anything against the bare label.
        label = ""
        for line in user.splitlines():
            if line.lower().startswith("label:"):
                label = line.split(":", 1)[1].strip().lower()
                break

        for pattern, key in _LABEL_RULES:
            if re.search(pattern, label):
                return MappingProposal(
                    semantic_key=key,
                    field_type=self._infer_type(key, label),
                    confidence=0.9,
                    rationale=f"fixture rule matched /{pattern}/",
                )
        return MappingProposal(
            semantic_key=None,
            field_type="unknown",
            confidence=0.0,
            rationale="no fixture rule matched",
        )

    @staticmethod
    def _infer_type(key: str, text: str) -> str:
        if key == "ssn" or key.endswith("_ssn"):
            return "ssn"
        if key.startswith("filing_status_"):
            return "checkbox"
        if key.startswith(_MONEY_KEY_PREFIXES):
            return "money"
        return "text"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicClient,
    "bedrock": BedrockClient,
    "fixture": FixtureClient,
}


def build_client(provider: str | None = None, *, effort: str = "high") -> LLMClient:
    name = (provider or os.getenv("TAXORCHESTRA_LLM_PROVIDER", "fixture")).lower()
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise LLMError(
            f"unknown provider {name!r}; expected one of {', '.join(sorted(_PROVIDERS))}"
        ) from None
    return factory(effort=effort)
