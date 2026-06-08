"""
Drug Indication Researcher
===========================
Researches drug indications from INNOVATOR / INVESTOR sources:
  - Company investor presentations & pipeline pages
  - FDA / EMA approval labels
  - Press releases & earnings calls
  - Company R&D day presentations

Uses Gemini 2.5 Flash with Google Search grounding (no BigQuery).

Output: one indication per row in standardized format (same columns as
label.py and drug_literature_fetcher.py).

Usage:
    python drug_indication_researcher.py semaglutide
    python drug_indication_researcher.py semaglutide --company "Novo Nordisk"
"""

import sys
import os
import json
import re
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from indication_standardizer import process_indications

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)


# ══════════════════════════════════════════════════════════════════════════════
#  RESEARCH QUERIES — each targets a different innovator source type
# ══════════════════════════════════════════════════════════════════════════════

def _build_queries(molecule: str, company: str | None) -> list[dict]:
    """Return a list of research tasks, each with a prompt for Gemini + Search."""
    company_clause = f" by {company}" if company else ""
    queries = [
        {
            "source_type": "FDA / EMA Approved Label",
            "prompt": (
                f"Search for the FDA-approved label and EMA SmPC for {molecule}.\n"
                f"List every approved therapeutic indication for {molecule} from regulatory labels.\n"
                f"Include the brand name(s) and approval year if available.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "source": "...", "source_url": "...", "detail": "..."}}\n'
                f"]}}\n"
                f"Rules:\n"
                f"- Each indication is a separate entry\n"
                f"- source = the label / document name\n"
                f"- source_url = URL if found, else empty string\n"
                f"- detail = brief note (approval year, patient population)\n"
                f"- No explanation, only JSON"
            ),
        },
        {
            "source_type": "Innovator Pipeline / Investor Presentation",
            "prompt": (
                f"Search for the latest investor presentations, pipeline updates, "
                f"and R&D day slides{company_clause} about {molecule}.\n"
                f"List all disease indications or therapeutic areas being studied or "
                f"promoted for {molecule} in these investor/company materials.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "source": "...", "source_url": "...", "detail": "..."}}\n'
                f"]}}\n"
                f"Rules:\n"
                f"- Each indication is a separate entry\n"
                f"- source = presentation or document title\n"
                f"- source_url = URL if found, else empty string\n"
                f"- detail = phase, trial name, or key claim from the presentation\n"
                f"- No explanation, only JSON"
            ),
        },
        {
            "source_type": "Press Release / Earnings Call",
            "prompt": (
                f"Search for recent press releases, earnings call transcripts, "
                f"and news announcements{company_clause} about {molecule}.\n"
                f"List all disease indications mentioned in the context of clinical "
                f"development, regulatory filings, or commercial strategy for {molecule}.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "source": "...", "source_url": "...", "detail": "..."}}\n'
                f"]}}\n"
                f"Rules:\n"
                f"- Each indication is a separate entry\n"
                f"- source = article / press release title\n"
                f"- source_url = URL if found, else empty string\n"
                f"- detail = key context (phase, filing, approval)\n"
                f"- No explanation, only JSON"
            ),
        },
        {
            "source_type": "Company Pipeline Page",
            "prompt": (
                f"Search for the official company pipeline page{company_clause} that lists {molecule}.\n"
                f"List every indication for {molecule} shown on the company's pipeline, "
                f"including the development phase for each.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "source": "...", "source_url": "...", "detail": "..."}}\n'
                f"]}}\n"
                f"Rules:\n"
                f"- Each indication is a separate entry\n"
                f"- source = company pipeline page name\n"
                f"- source_url = URL of the pipeline page if found, else empty string\n"
                f"- detail = development phase (e.g. Phase 3, Filed, Approved)\n"
                f"- No explanation, only JSON"
            ),
        },
    ]
    return queries


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI + GOOGLE SEARCH GROUNDING
# ══════════════════════════════════════════════════════════════════════════════

def research_indications(molecule: str, company: str | None) -> list[dict]:
    """
    Run multiple Gemini + Google Search queries to gather indications
    from innovator / investor sources.

    Returns flat rows in common format (one indication per row).
    """
    queries = _build_queries(molecule, company)
    all_rows = []

    for q in queries:
        source_type = q["source_type"]
        prompt      = q["prompt"]

        print(f"\n  🔍 {source_type}…")

        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    system_instruction=(
                        "You are a pharmaceutical research analyst. "
                        "Search the web and return ONLY valid JSON. "
                        "No markdown fences, no explanation."
                    ),
                ),
            )

            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

            data = json.loads(text)
            entries = data.get("entries", [])

            if not entries:
                print(f"     ⚠️  No entries found")
                continue

            print(f"     ✅ {len(entries)} entries")

            # ── Standardize + classify + unnest ──────────────────────────
            raw_indications = [e.get("indication", "") for e in entries]
            processed = process_indications(molecule, raw_indications)

            # Build a lookup from raw → entry details for source info
            entry_lookup = {}
            for e in entries:
                from indication_standardizer import standardize_indication
                std = standardize_indication(e.get("indication", ""))
                if std not in entry_lookup:
                    entry_lookup[std] = e

            for ind in processed:
                entry = entry_lookup.get(ind["indication"], {})
                all_rows.append({
                    "molecule_name":   molecule.title(),
                    "company_name":    company or "",
                    "indication":      ind["indication"],
                    "indication_type": ind["indication_type"],
                    "trial_title":     entry.get("detail", ""),
                    "trial_id":        "",
                    "phase":           "",
                    "source_url":      entry.get("source_url", ""),
                    "data_source":     f"Innovator: {source_type}",
                })

        except json.JSONDecodeError:
            print(f"     ⚠️  Could not parse JSON from Gemini response")
            # Try to extract what we can with a fallback
            _fallback_extract(molecule, company, source_type, text, all_rows)
        except Exception as e:
            print(f"     ❌ Error: {e}")

    # ── Deduplicate across source types (keep first occurrence) ───────────
    seen = set()
    deduped = []
    for row in all_rows:
        key = row["indication"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    print(f"\n  ✅ Total unique indications: {len(deduped)} (from {len(all_rows)} raw)\n")
    return deduped


def _fallback_extract(molecule, company, source_type, raw_text, all_rows):
    """If JSON parse fails, try a second Gemini call to extract indications from the text."""
    try:
        fallback_prompt = (
            f"Extract all disease/medical indications from this text about {molecule}.\n"
            f"Return ONLY a JSON array of strings, e.g. [\"Obesity\", \"T2DM\"].\n\n"
            f"Text:\n{raw_text[:3000]}"
        )
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=fallback_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                system_instruction="Return ONLY a JSON array of strings.",
            ),
        )
        text = resp.text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        parsed = json.loads(text)
        if isinstance(parsed, list):
            processed = process_indications(molecule, [str(x) for x in parsed])
            for ind in processed:
                all_rows.append({
                    "molecule_name":   molecule.title(),
                    "company_name":    company or "",
                    "indication":      ind["indication"],
                    "indication_type": ind["indication_type"],
                    "trial_title":     "",
                    "trial_id":        "",
                    "phase":           "",
                    "source_url":      "",
                    "data_source":     f"Innovator: {source_type}",
                })
            print(f"     🔄 Fallback extracted {len(processed)} indications")
    except Exception:
        print(f"     ⚠️  Fallback extraction also failed")


# ══════════════════════════════════════════════════════════════════════════════
#  COMMON FORMAT HEADERS (same as label.py / drug_literature_fetcher.py)
# ══════════════════════════════════════════════════════════════════════════════

HEADERS    = ["molecule_name", "company_name", "indication", "indication_type",
              "trial_title", "trial_id", "phase", "source_url", "data_source"]
COL_WIDTHS = [18, 22, 28, 16, 50, 18, 10, 40, 16]


def _thin_border():
    return Border(bottom=Side(style="thin", color="DDDDDD"),
                  right=Side(style="thin", color="DDDDDD"))


def write_excel(rows, molecule_name):
    print("🔹 Writing Excel...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Innovator Research"

    thin     = _thin_border()
    hfill    = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")
    ncols    = len(HEADERS)

    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value     = f"Drug Indication Research (Innovator)  —  {molecule_name.title()}"
    c.font      = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    for col, h in enumerate(HEADERS, 1):
        c           = ws.cell(row=2, column=col, value=h)
        c.font      = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill      = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border    = thin
    ws.row_dimensions[2].height = 22

    for idx, row in enumerate(rows, start=3):
        fill = alt_fill if idx % 2 == 1 else None
        for col, key in enumerate(HEADERS, 1):
            c           = ws.cell(row=idx, column=col, value=row.get(key, ""))
            c.font      = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            c.border    = thin
            if fill:
                c.fill = fill
        ws.row_dimensions[idx].height = 18

    ws.auto_filter.ref = f"A2:{get_column_letter(ncols)}{2 + len(rows)}"
    ws.freeze_panes = "A3"
    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    file_name = f"{molecule_name.lower().replace(' ', '_')}_indication_research.xlsx"
    wb.save(file_name)
    print(f"✅ Excel saved → {file_name}\n")
    return file_name


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Research drug indications from innovator/investor sources (Gemini + Google Search)"
    )
    parser.add_argument("molecule_name", help="Drug/molecule name, e.g. semaglutide")
    parser.add_argument("--company", default=None,
                        help="Innovator company name, e.g. 'Novo Nordisk' (optional, improves search)")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    molecule = args.molecule_name
    company  = args.company

    if not GEMINI_API_KEY:
        sys.exit("❌  GEMINI_API_KEY not set in .env")

    print(f"\n🚀 Drug Indication Researcher")
    print(f"   Molecule : {molecule.title()}")
    if company:
        print(f"   Company  : {company}")
    print()

    print("[1/2] Researching innovator sources…")
    rows = research_indications(molecule, company)

    if not rows:
        print("❌ No indications found from innovator sources")
        return

    print("[2/2] Generating Excel…")
    output = write_excel(rows, molecule)

    print("🎉 DONE")
    print(f"📄 File: {output}")
    print(f"📊 Indications found: {len(rows)}")


if __name__ == "__main__":
    main()
