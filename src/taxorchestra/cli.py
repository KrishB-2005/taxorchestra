"""Command line for TaxOrchestra."""

from __future__ import annotations

import json
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Annotated

# pypdf imports a deprecated cryptography symbol on load. Not our call to make
# and not actionable by the user, so it stays out of the CLI output. This has to
# run before anything pulls pypdf in, hence the E402 waivers below.
warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")

# Windows consoles default to cp1252, which cannot encode the em dashes this CLI
# prints; piping output then dies with UnicodeEncodeError before anything runs.
for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

# ruff: noqa: E402

import typer
from dotenv import load_dotenv

from taxorchestra.agents.extraction import ExtractionAgent
from taxorchestra.agents.mapping import FormMappingAgent
from taxorchestra.agents.orchestrator import Orchestrator
from taxorchestra.agents.validation import ValidationAgent
from taxorchestra.cache.store import build_cache
from taxorchestra.forms import acroform
from taxorchestra.knowledge.store import BM25Store
from taxorchestra.llm.client import build_client
from taxorchestra.models import Address, FilingStatus, Taxpayer
from taxorchestra.samples import SAMPLE_SSN, write_sample_documents

IRS_1040_URL = "https://www.irs.gov/pub/irs-pdf/f1040.pdf"
DEFAULT_TEMPLATE = "data/forms/f1040.pdf"

app = typer.Typer(
    add_completion=False,
    help="Multi-agent preparation of a real IRS Form 1040 PDF.",
)

load_dotenv()


def _build_orchestrator(provider: str | None, cache_backend: str | None) -> Orchestrator:
    llm = build_client(provider)
    return Orchestrator(
        extraction=ExtractionAgent(llm),
        mapping=FormMappingAgent(llm, build_cache(cache_backend)),
        validation=ValidationAgent(BM25Store()),
    )


@app.command("fetch-form")
def fetch_form(
    out: Annotated[Path, typer.Option(help="Where to save the blank form.")] = Path(
        DEFAULT_TEMPLATE
    ),
) -> None:
    """Download the blank Form 1040 from irs.gov.

    Not vendored into the repo: it is a government PDF that is reissued every
    tax year, and the cache is keyed on its digest so a new revision is picked
    up automatically.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"fetching {IRS_1040_URL}")
    with urllib.request.urlopen(IRS_1040_URL, timeout=60) as response:  # noqa: S310
        out.write_bytes(response.read())
    year = acroform.detect_form_year(out)
    typer.echo(f"saved {out} ({out.stat().st_size:,} bytes, tax year {year})")


@app.command()
def samples(
    out: Annotated[Path, typer.Option(help="Directory for the generated documents.")] = Path(
        "data/samples/generated"
    ),
) -> None:
    """Generate the synthetic W-2 and 1099-INT used by the demo."""
    for path in write_sample_documents(out):
        typer.echo(f"wrote {path}")


@app.command("map")
def map_fields(
    template: Annotated[Path, typer.Option(help="Blank form to map.")] = Path(DEFAULT_TEMPLATE),
    provider: Annotated[str | None, typer.Option(help="anthropic | bedrock | fixture")] = None,
    cache: Annotated[str | None, typer.Option(help="sqlite | dynamodb | memory")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Force re-resolution.")] = False,
    show: Annotated[int, typer.Option(help="How many resolved fields to print.")] = 25,
) -> None:
    """Resolve the form's cryptic field names and print the catalog."""
    agent = FormMappingAgent(build_client(provider), build_cache(cache))
    catalog = agent.build_catalog(str(template), use_cache=not no_cache)

    typer.echo(
        f"{catalog.form_id} (tax year {catalog.form_year})  "
        f"sha256={catalog.pdf_sha256[:12]}"
    )
    typer.echo(
        f"{catalog.resolved_count}/{len(catalog.mappings)} fields resolved  "
        f"cache_hit={agent.stats.cache_hit}  model_calls={agent.stats.llm_calls}"
    )
    typer.echo(f"label recovered by phase: {agent.stats.phase_counts}")
    typer.echo("")

    resolved = [m for m in catalog.mappings if m.resolved]
    for mapping in resolved[:show]:
        leaf = mapping.acro_name.split(".")[-1]
        typer.echo(
            f"  {leaf:12} line {str(mapping.line or '-'):5} "
            f"{mapping.semantic_key:36} {mapping.field_type.value}"
        )
    if len(resolved) > show:
        typer.echo(f"  … {len(resolved) - show} more")


@app.command("file")
def file_return(
    documents: Annotated[list[Path], typer.Argument(help="W-2 / 1099 PDFs to file from.")],
    first_name: Annotated[str, typer.Option()] = "Dana",
    last_name: Annotated[str, typer.Option()] = "Whitfield",
    ssn: Annotated[str, typer.Option()] = SAMPLE_SSN,
    street: Annotated[str, typer.Option()] = "418 Larkspur Lane",
    city: Annotated[str, typer.Option()] = "Silver Spring",
    state: Annotated[str, typer.Option()] = "MD",
    zip_code: Annotated[str, typer.Option("--zip")] = "20910",
    status: Annotated[FilingStatus, typer.Option(help="Filing status.")] = FilingStatus.SINGLE,
    template: Annotated[Path, typer.Option()] = Path(DEFAULT_TEMPLATE),
    out: Annotated[Path, typer.Option(help="Where to write the filled return.")] = Path(
        "out/f1040-filled.pdf"
    ),
    provider: Annotated[str | None, typer.Option(help="anthropic | bedrock | fixture")] = None,
    cache: Annotated[str | None, typer.Option(help="sqlite | dynamodb | memory")] = None,
    allow_invalid: Annotated[
        bool, typer.Option("--allow-invalid", help="Write even if validation fails.")
    ] = False,
    show_return: Annotated[bool, typer.Option("--show-return/--no-show-return")] = True,
) -> None:
    """Read the documents, compute the return, validate it, and fill the form."""
    taxpayer = Taxpayer(
        first_name=first_name,
        last_name=last_name,
        ssn=ssn,
        address=Address(street=street, city=city, state=state, zip_code=zip_code),
        filing_status=status,
    )

    orchestrator = _build_orchestrator(provider, cache)
    try:
        result = orchestrator.file_return(
            taxpayer=taxpayer,
            document_paths=list(documents),
            template_pdf=template,
            output_pdf=out,
            allow_invalid=allow_invalid,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if show_return:
        typer.echo(f"Form 1040 — tax year {result.tax_year}")
        for line in result.return_.lines:
            if line.value:
                typer.echo(
                    f"  line {line.line:4} {line.semantic_key:32} {line.value:>12,}"
                )
        typer.echo("")

    for issue in result.validation.issues:
        colour = (
            typer.colors.RED if issue.severity.value == "error" else typer.colors.YELLOW
        )
        typer.secho(f"  [{issue.severity.value}] {issue.message}", fg=colour)
        if issue.citation:
            typer.echo(f"           — {issue.citation}")

    typer.secho(
        f"wrote {result.output_pdf}  "
        f"({result.fields_written}/{result.fields_attempted} fields, "
        f"{result.elapsed_seconds}s, cache_hit={result.cache_hit})",
        fg=typer.colors.GREEN,
    )


@app.command()
def benchmark(
    template: Annotated[Path, typer.Option()] = Path(DEFAULT_TEMPLATE),
    provider: Annotated[str | None, typer.Option()] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write results as JSON.")] = None,
) -> None:
    """Measure fill accuracy and the cold/warm cache split."""
    from taxorchestra.benchmarks.accuracy import run_benchmark

    results = run_benchmark(template=str(template), provider=provider)
    typer.echo(json.dumps(results, indent=2))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(results, indent=2))
        typer.echo(f"\nwrote {json_out}")


@app.command()
def interview(
    template: Annotated[Path, typer.Option()] = Path(DEFAULT_TEMPLATE),
    out: Annotated[Path, typer.Option(help="Where to write the filled form.")] = Path(
        "out/f1040-filled.pdf"
    ),
    load: Annotated[
        Path | None, typer.Option(help="Reuse answers saved by a previous run.")
    ] = None,
    save: Annotated[
        Path | None, typer.Option(help="Save answers so you can rerun without retyping.")
    ] = None,
    provider: Annotated[str | None, typer.Option(help="anthropic | bedrock | fixture")] = None,
    cache: Annotated[str | None, typer.Option(help="sqlite | dynamodb | memory")] = None,
) -> None:
    """Fill the form by answering questions. Writes a PDF; files nothing."""
    from taxorchestra import interview as ui

    if not template.exists():
        typer.secho(
            f"No blank form at {template}. Run `taxorchestra fetch-form` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("")
    typer.secho(
        "  TaxOrchestra — Form 1040 worksheet",
        fg=typer.colors.BRIGHT_WHITE,
        bold=True,
    )
    ui.note("Fills the form from your answers. It does not file, submit, e-file")
    ui.note("or transmit anything — there is no such code in this project.")
    typer.secho(
        "   Not tax software. Check every figure before you rely on it.",
        fg=typer.colors.YELLOW,
    )

    prefill: dict = {}
    existing: list = []
    if load is not None:
        if not load.exists():
            typer.secho(f"No saved answers at {load}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        prefill, existing = ui.load_profile(load)
        ui.note(f"Loaded your previous answers from {load}.")

    taxpayer = ui.ask_taxpayer(prefill)
    orchestrator = _build_orchestrator(provider, cache)
    documents = ui.ask_documents(orchestrator.extraction, existing)

    if not documents:
        raise typer.Exit(code=1)

    ui.summarise(taxpayer, documents)
    typer.echo("")
    if not typer.confirm("  Is that right?", default=True):
        typer.echo("")
        typer.echo("  Nothing written. Rerun when you are ready.")
        raise typer.Exit(code=0)

    if save is not None:
        ui.save_profile(save, taxpayer, documents)
        ui.rule("Saved")
        ui.note(f"Answers written to {save}")
        typer.secho(
            "   That file holds your SSN and income figures. Keep it off shared"
            " drives and out of git.",
            fg=typer.colors.YELLOW,
        )

    try:
        result = orchestrator.file_documents(
            taxpayer=taxpayer,
            documents=documents,
            template_pdf=template,
            output_pdf=out,
        )
    except ValueError as exc:
        ui.rule("Stopped")
        typer.secho(f"  {exc}", fg=typer.colors.RED, err=True)
        typer.secho("  Nothing was written.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    ui.rule("Your return")
    typer.echo("")
    for line in result.return_.lines:
        if line.value:
            typer.echo(f"   line {line.line:4} {line.semantic_key:32} {line.value:>12,}")

    owed = result.return_.value("amount_owed")
    refund = result.return_.value("refund_overpaid")
    typer.echo("")
    if refund > 0:
        typer.secho(f"   Refund due to you: ${refund:,}", fg=typer.colors.GREEN, bold=True)
    elif owed > 0:
        typer.secho(f"   You owe: ${owed:,}", fg=typer.colors.YELLOW, bold=True)
    else:
        typer.secho("   Balance is zero.", fg=typer.colors.BRIGHT_WHITE)

    for issue in result.validation.issues:
        colour = typer.colors.RED if issue.severity.value == "error" else typer.colors.YELLOW
        typer.echo("")
        typer.secho(f"   [{issue.severity.value}] {issue.message}", fg=colour)
        if issue.citation:
            typer.echo(f"            — {issue.citation}")

    ui.rule("Done")
    typer.echo("")
    typer.secho(f"   Written to {result.output_pdf}", fg=typer.colors.GREEN)
    typer.echo(f"   {result.fields_written} fields filled on the 2025 Form 1040.")
    typer.echo("")
    typer.secho("   This is a draft worksheet, not a filed return.", fg=typer.colors.YELLOW)
    typer.echo("   Open it, check every figure against your own documents, and use it")
    typer.echo("   to cross-check whatever you actually file with. It covers wages,")
    typer.echo("   interest, dividends and contract income only — if you have anything")
    typer.echo("   else, the numbers above are incomplete.")
    typer.echo("")


@app.command("export-catalog")
def export_catalog(
    template: Annotated[Path, typer.Option()] = Path(DEFAULT_TEMPLATE),
    out: Annotated[Path, typer.Option(help="Where to write the JSON catalog.")] = Path(
        "out/catalog.json"
    ),
    provider: Annotated[str | None, typer.Option()] = None,
    cache: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Export the resolved field catalog as JSON.

    Field resolution is the expensive, interesting half of this project and it
    only depends on the blank form. Exporting the catalog lets other front ends
    — notably the browser demo — fill the same PDF without re-running any of the
    mapping, and without needing Python at all.
    """
    agent = FormMappingAgent(build_client(provider), build_cache(cache))
    catalog = agent.build_catalog(str(template))

    payload = {
        "form_id": catalog.form_id,
        "form_year": catalog.form_year,
        "pdf_sha256": catalog.pdf_sha256,
        "generated_by": "taxorchestra export-catalog",
        "fields": {
            m.semantic_key: {
                "acro_name": m.acro_name,
                "page": m.page,
                "line": m.line,
                "type": m.field_type.value,
                "on_state": None,
            }
            for m in catalog.mappings
            if m.resolved
        },
    }

    # Checkboxes are set with the name of an appearance stream, which differs
    # per field. A consumer that cannot open the PDF's widget tree needs it.
    on_states = {w.acro_name: w.on_states for w in acroform.load_widgets(str(template))}
    for entry in payload["fields"].values():
        states = on_states.get(entry["acro_name"], ())
        if entry["type"] == "checkbox" and states:
            entry["on_state"] = states[0]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.secho(
        f"wrote {out} — {len(payload['fields'])} resolved fields "
        f"for {catalog.form_id} {catalog.form_year}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
