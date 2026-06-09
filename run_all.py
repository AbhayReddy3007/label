"""
Unified Drug Research Pipeline
===============================
Runs all two modules and writes every result into a single Excel file,
one sheet per module, plus a combined sheet.

Modules:
  [1/2] Clinical Efficacy    — BigQuery clinical trial data enriched via Gemini
  [2/2] Drug Indication Research — Innovator sources (FDA labels, investor
         presentations, press releases, pipeline pages, SEC filings) via
         Gemini + Google Search

All sheets use the SAME standardized column format:
    molecule_name | company_name | indication | indication_type |
    trial_title   | trial_id     | phase      | source_url      | data_source

Indications are standardized and classified as Primary or Secondary.
Each row has exactly ONE indication (multi-indication trials are unnested).

Usage:
    python run_all.py --molecule semaglutide
    python run_all.py --molecule semaglutide --company "Novo Nordisk"
"""

import sys
import os
import re
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

from indication_standardizer import (
    standardize_indication,
    classify_indication,
    get_therapy_area,
    process_indications,
)

# ── shared env ────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "").strip()
GBQ_PROJECT     = os.getenv("GBQ_PROJECT")
GBQ_DATASET     = os.getenv("GBQ_DATASET")
GBQ_TABLE       = os.getenv("GBQ_TABLE", "clinical_efficacy")
GBQ_SERVICE_KEY = os.getenv(
    "GBQ_SERVICE_KEY",
    r"C:\Users\p90022569\Downloads\cognito-prod-394707-d38a0283cb16 (2).json",
)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ══════════════════════════════════════════════════════════════════════════════
#  SECONDARY INDICATION CRITERIA PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SECONDARY_INDICATION_CRITERIA = """
You must extract ONLY secondary indications. Do NOT include the primary indication.

A secondary indication qualifies ONLY if ALL of the following are true:

- The indication represents a true expansion — i.e., it is not part of the primary indication (for clinical assets) or currently approved label (for commercial assets)
- The indication is described at a clear disease-level definition, avoiding vague, overlapping, or synonymous representations
- The source must describe observed or measured outcomes in that specific indication (e.g., trial results, endpoint readouts, biomarker response), not just planned evaluation or exploratory intent

The following must NOT be considered secondary indications:
* Indications mentioned only as hypothesis, targets, or exploratory possibilities
* Pipeline indications without any data or outcomes
* Mechanism based assumptions without clinical or empirical validation
* Early discovery or preclinical signals without human data
* Any indication lacking traceable, verifiable evidence of results

If no indication meets these criteria, return an empty list [].
"""


# ══════════════════════════════════════════════════════════════════════════════
#  COMMON FORMAT — all sheets share these columns
# ══════════════════════════════════════════════════════════════════════════════

HEADERS    = ["molecule_name", "company_name", "indication", "indication_type",
              "therapy_area", "trial_title", "trial_id", "phase", "source_url", "data_source"]
COL_WIDTHS = [18, 22, 28, 16, 18, 50, 18, 10, 40, 16]


# ══════════════════════════════════════════════════════════════════════════════
#  BIGQUERY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bq_client():
    from google.cloud import bigquery
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        GBQ_SERVICE_KEY,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=GBQ_PROJECT, credentials=credentials)


def fetch_bq_rows(molecule_name: str) -> list[dict]:
    print("  🔹 Connecting to BigQuery…")
    from google.cloud import bigquery
    client = _bq_client()
    query = f"""
        SELECT molecule_name, company_name, source_url, phase, trial_id
        FROM `{GBQ_PROJECT}.{GBQ_DATASET}.{GBQ_TABLE}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name)]
    )
    results = client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    print(f"  ✅ Retrieved {len(rows)} rows")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checkpoint(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI ENRICHMENT (clinical trials)
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_enrich_trials(rows: list[dict], molecule_name: str,
                          checkpoint_prefix: str, data_source_label: str) -> list[dict]:
    from google import genai
    from google.genai import types

    client  = genai.Client(api_key=GEMINI_API_KEY)
    total   = len(rows)
    cp_file = f"checkpoint_{checkpoint_prefix}_{molecule_name.lower().replace(' ', '_')}.json"
    cp      = load_checkpoint(cp_file)

    print(f"  📁 Checkpoint: {len(cp)} completed rows\n")

    flat = []
    for i, row in enumerate(rows, 1):
        trial_id = row.get("trial_id")
        print(f"  [{i}/{total}] {trial_id}")

        if trial_id in cp:
            data = cp[trial_id]
            conditions  = data["conditions"]
            trial_title = data["trial_title"]
            print("     ⏭ skipped (checkpoint)")
        else:
            prompt = f"""
You are a clinical trial data assistant.

Trial details:
Molecule: {row.get('molecule_name')}
Company:  {row.get('company_name')}
Trial ID: {trial_id}
Phase:    {row.get('phase')}
Source URL: {row.get('source_url')}

{SECONDARY_INDICATION_CRITERIA}

Return ONLY valid JSON:
{{
  "conditions": ["secondary_indication_1", "secondary_indication_2"],
  "trial_title": "<trial title>"
}}

Rules:
- Return ONLY secondary indications in the conditions list — do NOT include the primary indication
- If no secondary indications qualify, return: {{"conditions": [], "trial_title": "<trial title>"}}
- No explanations, only JSON
"""
            try:
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        system_instruction="Return ONLY valid JSON output",
                    ),
                )
                text = resp.text.strip()
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                data = json.loads(text)
                conditions  = data.get("conditions", [])
                trial_title = data.get("trial_title", "N/A")
                if isinstance(conditions, list):
                    conditions = list(set(c.strip() for c in conditions))
                else:
                    conditions = [str(conditions)]
                print(f"     ✅ {', '.join(conditions)}")
                cp[trial_id] = {"conditions": conditions, "trial_title": trial_title}
                save_checkpoint(cp_file, cp)
            except Exception as e:
                conditions  = ["Error"]
                trial_title = str(e)
                print(f"     ❌ {e}")

        # ── Standardize + classify + unnest ──────────────────────────────
        processed = process_indications(molecule_name, conditions)
        if not processed:
            processed = [{"indication": "No indication found", "indication_type": "", "therapy_area": ""}]

        for ind in processed:
            flat.append({
                "molecule_name":   row.get("molecule_name"),
                "company_name":    row.get("company_name"),
                "indication":      ind["indication"],
                "indication_type": ind["indication_type"],
                "therapy_area":    ind.get("therapy_area", ""),
                "trial_title":     trial_title,
                "trial_id":        trial_id,
                "phase":           row.get("phase"),
                "source_url":      row.get("source_url"),
                "data_source":     data_source_label,
            })

    print(f"\n  ✅ Unnested → {len(flat)} rows (from {total} trials)\n")
    return flat


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE A — label.py  (Clinical Efficacy)
# ══════════════════════════════════════════════════════════════════════════════

def run_label(molecule: str) -> list[dict]:
    print("\n━━━ [1/2] Clinical Efficacy (label.py) ━━━")
    rows = fetch_bq_rows(molecule)
    if not rows:
        print("  ❌ No data found in BigQuery")
        return []
    return _gemini_enrich_trials(rows, molecule, "label", "Clinical Trials")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE B — drug_indication_researcher.py
#
#  Searches innovator/investor/SEC sources: FDA labels, investor presentations,
#  press releases, pipeline pages, SEC filings (10-K/10-Q/8-K/20-F/6-K).
#  Each row = one indication × one source document.
# ══════════════════════════════════════════════════════════════════════════════

def _identify_company(molecule: str) -> str:
    from google import genai as _genai
    from google.genai import types as _types
    client = _genai.Client(api_key=GEMINI_API_KEY)
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                f"Who is the innovator pharmaceutical company that developed {molecule}? "
                f"Return ONLY JSON: {{\"company\": \"<name>\", \"brand_names\": [\"name1\"]}}"
            ),
            config=_types.GenerateContentConfig(
                temperature=0,
                tools=[_types.Tool(google_search=_types.GoogleSearch())],
                system_instruction="Return ONLY valid JSON. No markdown.",
            ),
        )
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()).strip()
        data = json.loads(text)
        company = data.get("company", "")
        print(f"  🏢 Auto-detected innovator: {company}")
        return company
    except Exception:
        return ""


def _build_innovator_queries(molecule: str, company: str) -> list[dict]:
    return [
        {
            "source_type": "Regulatory Label",
            "prompt": (
                f"Search for official FDA prescribing information and EMA SmPC for {molecule} (by {company}).\n"
                f"For EACH approved indication, return a separate entry.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{\"entries\": [\n'
                f'  {{\"indication\": \"...\", \"brand_name\": \"...\", \"source_document\": \"<exact label title>\", '
                f'\"phase\": \"Approved\", \"source_url\": \"...\", \"detail\": \"<approval year, population>\"}}\n'
                f"]}}\n"
                f"source_document must be the REAL document title. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Investor Presentation",
            "prompt": (
                f"Search for {company}'s latest investor presentations, Capital Markets Day slides, "
                f"R&D day presentations about {molecule}.\n"
                f"For EACH indication mentioned, return a separate entry.\n\n"
                f"{SECONDARY_INDICATION_CRITERIA}\n"
                f"Return ONLY valid JSON:\n"
                f'{{\"entries\": [\n'
                f'  {{\"indication\": \"...\", \"brand_name\": \"...\", '
                f'\"source_document\": \"<real presentation title with year>\", '
                f'\"phase\": \"<Phase 1/2/3/Filed/Approved/Launched>\", '
                f'\"source_url\": \"...\", \"detail\": \"<specific claim from the presentation>\"}}\n'
                f"]}}\n"
                f"Only include indications with observed outcomes data. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Press Release",
            "prompt": (
                f"Search for press releases and earnings call statements from {company} about {molecule}.\n"
                f"For EACH indication mentioned, return a separate entry.\n\n"
                f"{SECONDARY_INDICATION_CRITERIA}\n"
                f"Return ONLY valid JSON:\n"
                f'{{\"entries\": [\n'
                f'  {{\"indication\": \"...\", \"brand_name\": \"...\", '
                f'\"source_document\": \"<real press release headline or earnings call date>\", '
                f'\"phase\": \"<Phase 1/2/3/Filed/Approved>\", '
                f'\"source_url\": \"...\", \"detail\": \"<key announcement>\"}}\n'
                f"]}}\n"
                f"Only include indications backed by reported outcomes. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Pipeline Page",
            "prompt": (
                f"Search for {company}'s official pipeline page listing {molecule}.\n"
                f"For EACH indication on the pipeline, return a separate entry.\n\n"
                f"{SECONDARY_INDICATION_CRITERIA}\n"
                f"Return ONLY valid JSON:\n"
                f'{{\"entries\": [\n'
                f'  {{\"indication\": \"...\", \"brand_name\": \"...\", '
                f'\"source_document\": \"<e.g. {company} Pipeline — Q1 2025>\", '
                f'\"phase\": \"<Preclinical/Phase 1/2/3/Filed/Approved>\", '
                f'\"source_url\": \"...\", \"detail\": \"<status note>\"}}\n'
                f"]}}\n"
                f"Exclude Preclinical entries (no human data). phase MUST be filled. No explanation, only JSON."
            ),
        },
        {
            "source_type": "SEC Filing",
            "prompt": (
                f"Search the SEC EDGAR database and the web for SEC filings by {company} "
                f"(10-K annual reports, 10-Q quarterly reports, 8-K current reports, "
                f"20-F annual reports for foreign private issuers, 6-K reports) "
                f"that disclose clinical or regulatory information about {molecule}.\n\n"
                f"For EACH indication described with actual clinical data or regulatory outcomes "
                f"in these filings, return a separate entry.\n\n"
                f"{SECONDARY_INDICATION_CRITERIA}\n"
                f"Return ONLY valid JSON:\n"
                f'{{\"entries\": [\n'
                f'  {{\"indication\": \"...\", \"brand_name\": \"...\", '
                f'\"source_document\": \"<exact filing type and period, e.g. {company} 10-K FY2024>\", '
                f'\"phase\": \"<Phase 1/2/3/Filed/Approved/Launched>\", '
                f'\"source_url\": \"<SEC EDGAR URL>\", '
                f'\"detail\": \"<specific disclosed data: trial outcome, milestone, MD&A section>\"}}\n'
                f"]}}\n"
                f"Only include indications with actual results reported in the filing — "
                f"not forward-looking statements or boilerplate risk factors. No explanation, only JSON."
            ),
        },
    ]


def _innovator_research(molecule: str, company: str) -> list[dict]:
    from google import genai as _genai
    from google.genai import types as _types

    client   = _genai.Client(api_key=GEMINI_API_KEY)
    queries  = _build_innovator_queries(molecule, company)
    all_rows = []

    for q in queries:
        source_type = q["source_type"]
        print(f"\n  🔍 {source_type}…")
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=q["prompt"],
                config=_types.GenerateContentConfig(
                    temperature=0,
                    tools=[_types.Tool(google_search=_types.GoogleSearch())],
                    system_instruction=(
                        "You are a pharmaceutical analyst researching what the "
                        "innovator company publicly says about this drug, including SEC filings. "
                        "Search the web and SEC EDGAR. Return ONLY valid JSON. "
                        "Apply strict secondary indication criteria: only include indications "
                        "with documented, verifiable clinical outcomes."
                    ),
                ),
            )
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()).strip()
            entries = json.loads(text).get("entries", [])
            if not entries:
                print(f"     ⚠️  No entries")
                continue
            print(f"     ✅ {len(entries)} entries")

            for e in entries:
                raw = e.get("indication", "").strip()
                if not raw:
                    continue
                std = standardize_indication(raw)
                if std.lower() in ("error", "n/a", "none", "no indication found"):
                    continue
                all_rows.append({
                    "molecule_name":   molecule.title(),
                    "company_name":    company,
                    "indication":      std,
                    "indication_type": classify_indication(molecule, std),
                    "therapy_area":    get_therapy_area(std),
                    "trial_title":     e.get("source_document", ""),
                    "trial_id":        e.get("brand_name", ""),
                    "phase":           e.get("phase", ""),
                    "source_url":      e.get("source_url", ""),
                    "data_source":     f"Innovator: {source_type}",
                })
        except json.JSONDecodeError:
            print(f"     ⚠️  JSON parse failed")
        except Exception as ex:
            print(f"     ❌ {ex}")

    # dedup only exact same indication + same source doc
    seen, unique = set(), []
    for row in all_rows:
        key = (row["indication"], row["trial_title"], row["data_source"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    print(f"\n  📊 Innovator rows: {len(unique)} (deduped from {len(all_rows)})\n")
    return unique


def run_indication_researcher(molecule: str, company: str | None = None) -> list[dict]:
    print("\n━━━ [2/2] Drug Indication Research (Innovator + SEC Sources) ━━━")
    if not company:
        print("  🔎 Identifying innovator company…")
        company = _identify_company(molecule) or ""
    return _innovator_research(molecule, company)


# ══════════════════════════════════════════════════════════════════════════════
#  EXCEL WRITER — common format for all sheets
# ══════════════════════════════════════════════════════════════════════════════

def _thin_border():
    return Border(bottom=Side(style="thin", color="DDDDDD"),
                  right=Side(style="thin", color="DDDDDD"))


def _write_standard_sheet(ws, rows: list[dict], molecule: str, sheet_title: str):
    thin     = _thin_border()
    hfill    = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")
    ncols    = len(HEADERS)

    # Title row
    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c           = ws["A1"]
    c.value     = f"{sheet_title}  —  {molecule.title()}"
    c.font      = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    # Subtitle row
    ws.merge_cells(f"A2:{get_column_letter(ncols)}2")
    c           = ws["A2"]
    c.value     = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   Rows: {len(rows)}"
    c.font      = Font(name="Arial", italic=True, size=9, color="888888")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 16

    # Header row
    for col, h in enumerate(HEADERS, 1):
        c           = ws.cell(row=3, column=col, value=h)
        c.font      = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill      = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border    = thin
    ws.row_dimensions[3].height = 22

    # Data rows
    for idx, row in enumerate(rows, start=4):
        fill = alt_fill if idx % 2 == 0 else None
        for col, key in enumerate(HEADERS, 1):
            c           = ws.cell(row=idx, column=col, value=row.get(key, ""))
            c.font      = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            c.border    = thin
            if fill:
                c.fill = fill
        ws.row_dimensions[idx].height = 18

    ws.auto_filter.ref = f"A3:{get_column_letter(ncols)}{3 + len(rows)}"
    ws.freeze_panes = "A4"
    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified drug research pipeline → single Excel")
    parser.add_argument("--molecule", required=True, help="Drug/molecule name, e.g. semaglutide")
    parser.add_argument("--company",  default=None,
                        help="Innovator company name, e.g. 'Novo Nordisk' (improves innovator search)")
    args = parser.parse_args()

    molecule = args.molecule.strip()
    slug     = molecule.lower().replace(" ", "_")
    out_file = f"{slug}_research.xlsx"

    print(f"\n{'━'*62}")
    print(f"  Molecule : {molecule.title()}")
    print(f"  Output   : {out_file}")
    print(f"{'━'*62}")

    if not GEMINI_API_KEY:
        sys.exit("❌  GEMINI_API_KEY not set in .env")

    # ── run all modules ───────────────────────────────────────────────────────
    label_rows      = run_label(molecule)
    indication_rows = run_indication_researcher(molecule, args.company)

    # ── assemble workbook ─────────────────────────────────────────────────────
    print(f"\n📊 Writing {out_file}…")
    wb = openpyxl.Workbook()

    # Sheet 1 — Clinical Efficacy
    ws1       = wb.active
    ws1.title = "Clinical Efficacy"
    _write_standard_sheet(ws1, label_rows, molecule, "Clinical Efficacy")

    # Sheet 2 — Drug Indication Research (Innovator + SEC)
    ws2       = wb.create_sheet("Drug Indication Research")
    _write_standard_sheet(ws2, indication_rows, molecule, "Drug Indication Research")

    # Sheet 3 — Combined (both merged)
    combined  = label_rows + indication_rows
    ws3       = wb.create_sheet("All Combined")
    _write_standard_sheet(ws3, combined, molecule, "All Sources Combined")

    wb.save(out_file)

    print(f"\n✅  Done!")
    print(f"📄  File   : {out_file}")
    print(f"📊  Sheets : Clinical Efficacy ({len(label_rows)}) | "
          f"Drug Indication Research ({len(indication_rows)}) | "
          f"All Combined ({len(combined)})")


if __name__ == "__main__":
    main()
