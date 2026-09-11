# TaxOrchestra

Reads W-2s and 1099s and fills a **real IRS Form 1040 PDF** end to end — four
agents, one validated return, no manual data entry.

```
$ taxorchestra file data/samples/generated/*.pdf

Form 1040 — tax year 2025
  line 1a   wages_w2_box1                          92,450
  line 1z   wages_total                            92,450
  line 2b   interest_taxable                        1,284
  line 9    total_income                           93,734
  line 11   adjusted_gross_income                  93,734
  line 12   deduction_standard_or_itemized         15,750
  line 15   taxable_income                         77,984
  line 24   tax_total                              12,070
  line 25a  withholding_w2                         11,894
  line 37   amount_owed                               176

wrote out/f1040-filled.pdf  (22/22 fields, 1.63s, cache_hit=False)
```

> **This is not tax software.** It is a document-automation demonstration. It
> handles the ordinary-income case only — no capital gains worksheet, no AMT,
> no credits, no state return — and the tax constants in `taxdata.py` must be
> re-verified against the IRS revenue procedure for the year before anyone
> relies on a number it produces. Do not file a return with it.

---

## The actual problem

The IRS Form 1040 is a fillable PDF with **229 AcroForm fields**. Every one of
them is named like this:

```
topmostSubform[0].Page1[0].f1_47[0]
topmostSubform[0].Page1[0].f1_48[0]
```

There are no `/TU` tooltips. No alternate names. No schema, no mapping file,
nothing. The form carries zero machine-readable indication of what any field
means — a fact the test suite asserts, because the whole design follows from it:

```python
def test_the_form_has_no_usable_metadata(self):
    fields = PdfReader(TEMPLATE).get_fields()
    assert len(fields) > 200
    assert not any(meta.get("/TU") for meta in fields.values())
```

So before you can write a wage figure into the right box, you have to work out
which box that is.

## How the field mapping works

The one thing the PDF does tell you is **geometry**. Each field has a rectangle,
and the label the IRS printed next to it is a text run at a known position. Join
them and the meaning falls out:

```
f1_47[0]  ←  ['Income', '1', 'a', 'Total amount from Form(s) W-2, box 1']
```

Five phases, cheapest first:

| Phase | What it does | Cost |
|---|---|---|
| 1. Geometric | Text on the widget's baseline, to its left | free |
| 2. Structural | Carries line numbers and section headings down the page, so a continuation row labelled `b` becomes `1b` under *Income* | free |
| 3. Caption | For the identity block, where the label is printed *above* the box and there is nothing to the left | free |
| 4. Table | Groups what is left into columns and takes the header printed above each. The schedules are mostly tables of identical boxes with no per-row label | free |
| 5. Semantic | Hands the recovered label to Claude, gets back a canonical `snake_case` key and a field type | one call per **distinct label** |

On the 2025 form the free phases recover a label for 162 of 199 fields without
a single model call; the semantic phase names them.

Two things that look like details and are not:

**Two-column rows.** The 1040 puts `2a Tax-exempt interest ___  b Taxable
interest ___` on one line. Take "everything to the left" and box 2b claims 2a's
label as well, and you write tax-exempt interest into the taxable line. Each
widget's label search is clipped at its left-hand neighbour on the row.

**Confidence floor.** A field the model is less than 55% sure about is left
*unresolved* rather than guessed. An empty box on a tax form is a visible
omission; a wrong box is a silent error.

## It is not a 1040 tool

Nothing in the mapping is specific to the 1040 — the phases run against any
fillable IRS PDF. Measured across eleven real forms pulled from irs.gov:

| Form | Fields | Label recovered |
|---|---|---|
| 1040 | 199 | 81% |
| Schedule B | 72 | **100%** |
| Schedule C | 105 | 90% |
| Schedule SE | 27 | **100%** |
| Schedule SR | 27 | 93% |
| Form 2441 | 72 | 79% |
| W-9 | 23 | 78% |
| Schedule 1 / 2 / 3 | 173 | 62-66% |
| Form 8812 | 38 | 58% |
| **Total** | **736** | **80%** |

Schedule B was 31% before the table phase: it is a two-column ledger of payer
names and amounts, rows every 12pt, no per-row label anywhere — the label is a
column header printed once at the top. Grouping leftover widgets into columns
and reading the header above each took it to 100%, and gives every cell a row
index so it stays addressable:

```bash
taxorchestra fetch-form --form f1040sb
taxorchestra map --template data/forms/f1040sb.pdf
taxorchestra fill-form data/forms/f1040sb.pdf --values amounts.json
```

```json
{ "amount_row1": 1284, "amount_row2": 512, "amount_row3": 96 }
```

**What generalises is placement, not correctness.** Putting a value in the right
box on an arbitrary form is solved. Knowing *which* value belongs there is tax
logic, written per form, and only the 1040 ordinary-income case is implemented
(`taxorchestra file`). `fill-form` is deliberately the honest surface for
everything else: you supply the numbers, it places them.

One caveat on the bundled `fixture` provider: its naming rules are mostly
1040-specific, so semantic resolution on other forms is low without a real
model. Label recovery — the part that is actually hard — is provider-independent,
which is why the table above measures that.

## Caching

Resolving all 229 fields is the only stage that costs money. The result is a
pure function of the blank PDF, so it is cached against the form's SHA-256 —
which means a new tax year ships a new PDF, gets a new digest, and re-resolves
automatically. There is no failure mode where last year's field numbering
quietly fills this year's form.

Measured, `--provider fixture`, 199 widgets:

| | time | model calls |
|---|---|---|
| Cold (resolve + compute + fill) | 1.63 s | 145 |
| Warm (cached catalog) | 1.25 s | 0 |
| Cached catalog read alone | **~2 ms** | 0 |

The warm end-to-end time is dominated by parsing and rewriting a 220 KB PDF,
not by the mapping. Backends: `sqlite` (default), `dynamodb`, `memory`.

## The four agents

**1. Extraction** — source documents to typed records. Tries the PDF text layer
first with anchored patterns (free, deterministic, handles the common case);
falls back to handing Claude the PDF itself for scans and unusual layouts. The
cheapest correct answer wins, and `confidence` records which path produced the
record.

**2. FormMapping** — the five phases above.

**3. Validation** — re-derives every total from its components. This is
deliberately *not* a model call. The failure mode it exists to catch is a
language model inventing a number, and you do not catch invented numbers by
asking another language model whether they look plausible:

```python
def test_catches_a_tampered_total(self):
    result = compute_return(taxpayer, [SAMPLE_W2])
    for line in result.lines:
        if line.semantic_key == "taxable_income":
            line.value = Decimal("1")          # as if a model had made it up

    report = ValidationAgent(BM25Store()).validate(result, [SAMPLE_W2])
    assert not report.ok
    assert all(i.citation for i in report.errors)
```

Retrieval's job here is *grounding*, not judgement — BM25 over a curated corpus
of 1040 rules attaches a citation to each finding, so a reviewer gets "line 15
disagrees with line 11 − 14, per Form 1040 line 15" rather than "the validator
flagged line 15". BM25 rather than embeddings because the corpus is a few dozen
short rules: no model to load, no index to build, and the same query returns the
same rules every time, which is what makes the output testable.

**4. Orchestrator** — extract → compute → validate → map → render → fill.
Validation runs *before* anything is written; a return that fails an identity is
reported, not filed.

## Verification

Accuracy means: reopen the written PDF and check every value is stored in the
field it was meant to go in. Counting successful write calls would be measuring
our own optimism — a field can be written and still not land.

```
accuracy: 22/22 = 100.0%   mismatches=[]
```

```bash
taxorchestra benchmark --json out/benchmark.json
```

The full suite is 40 tests, all on the deterministic `fixture` provider — no API
key, no spend:

```bash
pytest          # 40 passed
```

One of them caught a real bug worth keeping: `Decimal.quantize` defaults to
banker's rounding, which sends $1,192.50 down to $1,192. The IRS instruction is
to round 50 cents **up**. `taxdata.to_dollars` uses `ROUND_HALF_UP`.

## Install and run

```bash
git clone https://github.com/KrishB-2005/taxorchestra
cd taxorchestra
pip install -e ".[llm,api,dev]"

taxorchestra fetch-form        # pulls the blank 1040 from irs.gov
taxorchestra samples           # synthetic W-2 + 1099-INT
taxorchestra file data/samples/generated/*.pdf
```

Runs with **no credentials** on the `fixture` provider — a deterministic
stand-in that resolves labels by rule, so the pipeline, the tests and the
benchmark all work on a laptop with no key. For the real thing:

```bash
cp .env.example .env           # set TAXORCHESTRA_LLM_PROVIDER=anthropic
taxorchestra file data/samples/generated/*.pdf --provider anthropic
```

The blank form is not vendored — it is a government PDF reissued every tax year,
and `fetch-form` gets the current one.

<details>
<summary>Windows PowerShell</summary>

The commands above are POSIX shell. In Windows PowerShell 5.1 the `taxorchestra`
commands and the `*.pdf` glob work unchanged, but two things differ:

```powershell
Copy-Item .env.example .env    # not `cp`
cd C:\path\to\taxorchestra; taxorchestra file data\samples\generated\*.pdf
```

Chain with `;` — `&&` is a parser error in 5.1, not just unsupported. PowerShell
7 accepts `&&` normally.

</details>

### Filling it in by answering questions

`interview` asks for what it needs instead of taking flags, and writes a draft
PDF at the end. It reads each W-2 and 1099 from a file, or takes the box numbers
typed in by hand if you have a paper copy or a scan.

```bash
taxorchestra interview
```

```
-- About you -------------------------------------------------
   This stays on your machine. Nothing is uploaded or transmitted.

  First name and middle initial: Krish
  Last name: Bhatt
  Social security number (###-##-####): 900-11-2222
  ...

  What would you like to add?
    1. W-2 - wages from an employer
    2. 1099-INT - bank or savings interest
    3. 1099-DIV - dividends from investments
    4. 1099-NEC - contract or freelance work
    5. Nothing more to add

   Refund due to you: $1,451
```

`--save answers.json` records what you typed so a rerun with `--load
answers.json` only asks about what changed. **That file holds your SSN and your
income figures** — `.gitignore` already covers `*answers*.json`, `*profile*.json`
and `.taxorchestra/`, and it should stay off shared drives.

**It does not file anything.** There is no e-file code, no network submission,
no IRS transmission path anywhere in this project — the only outputs are a PDF
and, optionally, that answers file. Treat the PDF as a worksheet to check your
real return against, not as a return.

### Commands

| | |
|---|---|
| `fetch-form` | download any blank IRS form (`--form f1040sb`) |
| `samples` | generate synthetic source documents |
| `map` | resolve the form's field names and print the catalog |
| `file` | read documents, compute, validate, fill (1040 only) |
| `fill-form` | place a JSON of values onto any mapped form |
| `interview` | fill the form by answering questions |
| `benchmark` | fill accuracy and the cold/warm cache split |

## Layout

```
src/taxorchestra/
  agents/        extraction · mapping · validation · orchestrator
  forms/         AcroForm geometry, label recovery, fill
  knowledge/     IRS rule corpus + BM25 retrieval
  llm/           anthropic | bedrock | fixture, behind one parse() call
  cache/         sqlite | dynamodb | memory
  returns.py     documents → 1040 line values (pure)
  taxdata.py     brackets, standard deduction, IRS rounding
  samples.py     synthetic W-2 / 1099 generation, incl. a small PDF writer
```

Everything crossing an agent boundary is a Pydantic model with `extra="forbid"`.
An LLM that invents a field is telling you the prompt drifted; that should fail
loudly at the boundary rather than vanish and surface three stages later as a
wrong refund.

## Notes

- Synthetic data only. SSNs use the 900-range the SSA never issues; EINs are
  invalid by construction.
- `samples.py` includes a ~60-line PDF writer rather than pulling in a rendering
  library that only the fixtures need.
- Swappable infrastructure throughout: Claude API ↔ Bedrock, SQLite ↔ DynamoDB,
  BM25 ↔ Elasticsearch. Nothing upstream changes when a deployment outgrows the
  local default.

MIT.
