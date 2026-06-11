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
    molecule_name | company_name | indication | rationale | indication_type |
    therapy_area  | trial_title  | trial_id   | phase     | source_url      | data_source

Indications are standardized and classified as Primary or Secondary.
Each row has exactly ONE indication (multi-indication trials are unnested).

Usage:
    # Single molecule
    python run_all.py --molecule semaglutide
    python run_all.py --molecule semaglutide --company "Novo Nordisk"

    # All molecules in BigQuery (one Excel file per molecule + a master file)
    python run_all.py --all
    python run_all.py --all --skip-existing   # skip molecules already processed
"""

import sys
import os
import re
import json
import time
import argparse
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── Parallel config ───────────────────────────────────────────────────────────
MAX_WORKERS_TRIALS    = 10   # parallel Gemini calls for clinical trial enrichment
MAX_WORKERS_INNOVATOR = 5    # parallel Gemini calls for innovator source queries

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# ══════════════════════════════════════════════════════════════════════════════
#  COMMON FORMAT — all sheets share these columns
# ══════════════════════════════════════════════════════════════════════════════

HEADERS    = ["molecule_name", "company_name", "indication", "rationale",
              "indication_type", "therapy_area", "trial_title", "trial_id",
              "phase", "source_url", "data_source"]
COL_WIDTHS = [18, 22, 28, 40, 16, 18, 50, 18, 10, 40, 16]


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


def fetch_all_molecules() -> list[str]:
    """Return the list of distinct molecule names in the BigQuery table."""
    print("  🔹 Fetching all distinct molecules from BigQuery…")
    client = _bq_client()
    query = f"""
        SELECT DISTINCT molecule_name
        FROM `{GBQ_PROJECT}.{GBQ_DATASET}.{GBQ_TABLE}`
        ORDER BY molecule_name
    """
    results = client.query(query).result()
    molecules = [row["molecule_name"] for row in results if row["molecule_name"]]
    print(f"  ✅ Found {len(molecules)} distinct molecules: {molecules}\n")
    return molecules


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
#  GEMINI ENRICHMENT (clinical trials) — PARALLELIZED
# ══════════════════════════════════════════════════════════════════════════════

def _enrich_single_trial(row, molecule_name, client, gen_types, checkpoint, cp_file, cp_lock):
    """Enrich a single trial. Returns (trial_id, conditions, trial_title, row)."""
    trial_id = row.get("trial_id")

    with cp_lock:
        if trial_id in checkpoint:
            data = checkpoint[trial_id]
            return (trial_id, data["conditions"], data["trial_title"], row)

    prompt = f"""
You are a clinical trial data assistant.

Trial details:
Molecule: {row.get('molecule_name')}
Company:  {row.get('company_name')}
Trial ID: {trial_id}
Phase:    {row.get('phase')}
Source URL: {row.get('source_url')}

Your task: Extract ALL disease indications being studied in this clinical trial.
Include BOTH the primary indication AND any secondary/exploratory indications that have documented outcomes.

Look at the trial title, trial ID, source URL, and any available information to identify every
disease or condition this trial is investigating.

Common patterns to look for:
- The trial title often contains the indication (e.g., "cardiovascular outcomes" → CV Risk Reduction)
- "in subjects with type 2 diabetes" → T2DM
- Trial acronyms like PIONEER, SUSTAIN, STEP, SELECT often indicate specific conditions
- Outcome studies (e.g., cardiovascular, renal) count as indications

Return ONLY valid JSON:
{{
  "conditions": [
    {{"indication": "<disease or condition>", "rationale": "<why this indication was identified — cite the specific evidence from trial title, ID, or known trial data>"}},
    {{"indication": "<disease or condition>", "rationale": "<why>"}}
  ],
  "trial_title": "<trial title>"
}}

Rules:
- Include ALL indications the trial is evaluating — both primary and secondary
- Always extract at least the primary indication from the trial title/details
- Each indication must have a rationale explaining what evidence led to its identification
- If the trial title mentions a condition (e.g., "type 2 diabetes", "cardiovascular outcomes"), that IS an indication — extract it
- No explanations outside the JSON
"""
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=gen_types.GenerateContentConfig(
                temperature=0,
                system_instruction="Return ONLY valid JSON output",
            ),
        )
        text = resp.text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        data = json.loads(text)
        conditions  = data.get("conditions", [])
        trial_title = data.get("trial_title", "N/A")

        # Normalize to list of dicts
        if isinstance(conditions, list):
            normalized = []
            for c in conditions:
                if isinstance(c, dict):
                    normalized.append({
                        "indication": (c.get("indication") or "").strip(),
                        "rationale": (c.get("rationale") or "").strip(),
                    })
                elif isinstance(c, str) and c.strip():
                    normalized.append({"indication": c.strip(), "rationale": ""})
            conditions = normalized
        else:
            conditions = [{"indication": str(conditions), "rationale": ""}]

        # Deduplicate by indication
        seen, deduped = set(), []
        for c in conditions:
            key = c["indication"].lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c)
        conditions = deduped

        print(f"     ✅ {trial_id}: {', '.join(c['indication'] for c in conditions)}")
        with cp_lock:
            checkpoint[trial_id] = {"conditions": conditions, "trial_title": trial_title}
            save_checkpoint(cp_file, checkpoint)

    except Exception as e:
        conditions  = []
        trial_title = str(e)
        print(f"     ❌ {trial_id}: {e}")

    return (trial_id, conditions, trial_title, row)


def _gemini_enrich_trials(rows: list[dict], molecule_name: str,
                          checkpoint_prefix: str, data_source_label: str) -> list[dict]:
    from google import genai
    from google.genai import types as gen_types

    client  = genai.Client(api_key=GEMINI_API_KEY)
    total   = len(rows)
    cp_file = f"checkpoint_{checkpoint_prefix}_{molecule_name.lower().replace(' ', '_')}.json"
    cp      = load_checkpoint(cp_file)
    cp_lock = threading.Lock()

    print(f"  📁 Checkpoint: {len(cp)} completed rows")
    print(f"  🚀 Processing {total} trials with {MAX_WORKERS_TRIALS} parallel workers…\n")

    # ── Parallel enrichment ──────────────────────────────────────────────
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_TRIALS) as executor:
        futures = {
            executor.submit(
                _enrich_single_trial, row, molecule_name, client, gen_types, cp, cp_file, cp_lock
            ): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                row = rows[idx]
                print(f"     ❌ Unexpected: {row.get('trial_id')}: {e}")
                results.append((row.get("trial_id"), [], str(e), row))

    # ── Sort by original order ───────────────────────────────────────────
    trial_order = {row.get("trial_id"): i for i, row in enumerate(rows)}
    results.sort(key=lambda r: trial_order.get(r[0], 0))

    # ── Standardize + classify + unnest ──────────────────────────────────
    flat = []
    for trial_id, conditions, trial_title, row in results:
        rationale_map = {}
        raw_indications = []
        for c in conditions:
            if isinstance(c, dict):
                ind = c.get("indication", "")
                rat = c.get("rationale", "")
                if ind:
                    raw_indications.append(ind)
                    rationale_map[ind.lower()] = rat
            elif isinstance(c, str) and c.strip():
                raw_indications.append(c.strip())

        processed = process_indications(molecule_name, raw_indications)
        if not processed:
            processed = [{"indication": "No indication found", "indication_type": "", "therapy_area": ""}]

        for ind in processed:
            # Resolve rationale: try standardized name, then try matching raw→std
            rationale = ""
            ind_name = ind["indication"]
            if ind_name.lower() in rationale_map:
                rationale = rationale_map[ind_name.lower()]
            else:
                for raw_key, rat in rationale_map.items():
                    if standardize_indication(raw_key) == ind_name:
                        rationale = rat
                        break

            flat.append({
                "molecule_name":   row.get("molecule_name"),
                "company_name":    row.get("company_name"),
                "indication":      ind["indication"],
                "rationale":       rationale,
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
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", "source_document": "<exact label title>", '
                f'"phase": "Approved", "source_url": "...", '
                f'"detail": "<approval year, population, dose>", '
                f'"rationale": "<why this indication was identified — cite the specific label section or approval>"}}\n'
                f"]}}\n"
                f"source_document must be the REAL document title. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Investor Presentation",
            "prompt": (
                f"Search for {company}'s latest investor presentations, Capital Markets Day slides, "
                f"R&D day presentations about {molecule}.\n"
                f"For EACH indication mentioned with documented outcomes, return a separate entry.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", '
                f'"source_document": "<real presentation title with year>", '
                f'"phase": "<Phase 1/2/3/Filed/Approved/Launched>", '
                f'"source_url": "...", '
                f'"detail": "<specific claim from the presentation>", '
                f'"rationale": "<why this indication was identified — cite the specific evidence>"}}\n'
                f"]}}\n"
                f"Only include indications with observed outcomes data. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Press Release",
            "prompt": (
                f"Search for press releases and earnings call statements from {company} about {molecule}.\n"
                f"For EACH indication mentioned with documented outcomes, return a separate entry.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", '
                f'"source_document": "<real press release headline or earnings call date>", '
                f'"phase": "<Phase 1/2/3/Filed/Approved>", '
                f'"source_url": "...", '
                f'"detail": "<key announcement>", '
                f'"rationale": "<why this indication was identified — cite the specific news or data>"}}\n'
                f"]}}\n"
                f"Only include indications backed by reported outcomes. No explanation, only JSON."
            ),
        },
        {
            "source_type": "Pipeline Page",
            "prompt": (
                f"Search for {company}'s official pipeline page listing {molecule}.\n"
                f"For EACH indication on the pipeline, return a separate entry.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", '
                f'"source_document": "<e.g. {company} Pipeline — Q1 2025>", '
                f'"phase": "<Preclinical/Phase 1/2/3/Filed/Approved>", '
                f'"source_url": "...", '
                f'"detail": "<status note>", '
                f'"rationale": "<why this indication was identified — cite the pipeline entry>"}}\n'
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
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", '
                f'"source_document": "<exact filing type and period, e.g. {company} 10-K FY2024>", '
                f'"phase": "<Phase 1/2/3/Filed/Approved/Launched>", '
                f'"source_url": "<SEC EDGAR URL>", '
                f'"detail": "<specific disclosed data: trial outcome, milestone, MD&A section>", '
                f'"rationale": "<why this indication was identified — cite the filing section and data>"}}\n'
                f"]}}\n"
                f"Only include indications with actual results reported in the filing — "
                f"not forward-looking statements or boilerplate risk factors. No explanation, only JSON."
            ),
        },
    ]


def _research_single_source(q, molecule, company, client, gen_types):
    """Query a single innovator source. Returns (source_type, rows)."""
    source_type = q["source_type"]
    rows = []
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=q["prompt"],
            config=gen_types.GenerateContentConfig(
                temperature=0,
                tools=[gen_types.Tool(google_search=gen_types.GoogleSearch())],
                system_instruction=(
                    "You are a pharmaceutical analyst researching what the "
                    "innovator company publicly says about this drug, including SEC filings. "
                    "Search the web and SEC EDGAR. Return ONLY valid JSON. "
                    "Include indications with documented, verifiable clinical outcomes."
                ),
            ),
        )
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip()).strip()
        entries = json.loads(text).get("entries", [])
        if not entries:
            print(f"     ⚠️  {source_type}: No entries")
            return (source_type, [])
        print(f"     ✅ {source_type}: {len(entries)} entries")

        for e in entries:
            raw = e.get("indication", "").strip()
            if not raw:
                continue
            std = standardize_indication(raw)
            if std.lower() in ("error", "n/a", "none", "no indication found"):
                continue
            rationale = (e.get("rationale") or e.get("detail") or "").strip()
            rows.append({
                "molecule_name":   molecule.title(),
                "company_name":    company,
                "indication":      std,
                "rationale":       rationale,
                "indication_type": classify_indication(molecule, std),
                "therapy_area":    get_therapy_area(std),
                "trial_title":     e.get("source_document", ""),
                "trial_id":        e.get("brand_name", ""),
                "phase":           e.get("phase", ""),
                "source_url":      e.get("source_url", ""),
                "data_source":     f"Innovator: {source_type}",
            })
    except json.JSONDecodeError:
        print(f"     ⚠️  {source_type}: JSON parse failed")
    except Exception as ex:
        print(f"     ❌ {source_type}: {ex}")

    return (source_type, rows)


def _innovator_research(molecule: str, company: str) -> list[dict]:
    from google import genai as _genai
    from google.genai import types as _types

    client   = _genai.Client(api_key=GEMINI_API_KEY)
    queries  = _build_innovator_queries(molecule, company)
    all_rows = []

    print(f"  🚀 Querying {len(queries)} innovator sources in parallel…\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_INNOVATOR) as executor:
        futures = {
            executor.submit(
                _research_single_source, q, molecule, company, client, _types
            ): q["source_type"]
            for q in queries
        }
        for future in as_completed(futures):
            source_type = futures[future]
            try:
                _, rows = future.result()
                all_rows.extend(rows)
            except Exception as ex:
                print(f"     ❌ {source_type}: {ex}")

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
#  PER-MOLECULE EXCEL WRITER
# ══════════════════════════════════════════════════════════════════════════════

def write_molecule_excel(molecule: str, label_rows: list[dict],
                         indication_rows: list[dict], out_file: str):
    """Write the 3-sheet Excel for a single molecule."""
    wb = openpyxl.Workbook()

    ws1       = wb.active
    ws1.title = "Clinical Efficacy"
    _write_standard_sheet(ws1, label_rows, molecule, "Clinical Efficacy")

    ws2       = wb.create_sheet("Drug Indication Research")
    _write_standard_sheet(ws2, indication_rows, molecule, "Drug Indication Research")

    combined  = label_rows + indication_rows
    ws3       = wb.create_sheet("All Combined")
    _write_standard_sheet(ws3, combined, molecule, "All Sources Combined")

    wb.save(out_file)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER EXCEL WRITER  (one sheet per molecule + grand summary)
# ══════════════════════════════════════════════════════════════════════════════

def write_master_excel(all_results: dict[str, list[dict]], out_file: str):
    """
    all_results: {molecule_name: [combined rows]}
    Writes one sheet per molecule + a grand 'All Molecules' sheet.
    """
    print(f"\n📊 Writing master file: {out_file}…")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)   # remove default empty sheet

    grand_rows = []

    for molecule, rows in all_results.items():
        # Truncate sheet name to Excel's 31-char limit
        sheet_name = molecule.title()[:31]
        ws = wb.create_sheet(sheet_name)
        _write_standard_sheet(ws, rows, molecule, "All Sources")
        grand_rows.extend(rows)

    ws_all = wb.create_sheet("All Molecules")
    _write_standard_sheet(ws_all, grand_rows, "All Molecules", "Grand Summary")

    wb.save(out_file)
    print(f"✅  Master file saved: {out_file}  ({len(grand_rows)} total rows across {len(all_results)} molecules)")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-MOLECULE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_molecule(molecule: str, company: str | None = None,
                     output_dir: str = ".") -> tuple[list[dict], list[dict]]:
    """Run both modules for one molecule. Returns (label_rows, indication_rows)."""
    label_rows      = run_label(molecule)
    indication_rows = run_indication_researcher(molecule, company)
    return label_rows, indication_rows


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified drug research pipeline → Excel")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--molecule", help="Single drug/molecule name, e.g. semaglutide")
    group.add_argument("--all", action="store_true",
                       help="Run pipeline for ALL distinct molecules in BigQuery")
    parser.add_argument("--company", default=None,
                        help="Innovator company (only used with --molecule)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to write output files (default: current dir)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="(--all mode) skip molecules whose Excel file already exists")
    parser.add_argument("--master-file", default=None,
                        help="(--all mode) filename for the master combined Excel "
                             "(default: all_molecules_research_<timestamp>.xlsx)")
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        sys.exit("❌  GEMINI_API_KEY not set in .env")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── SINGLE-MOLECULE MODE ──────────────────────────────────────────────
    if args.molecule:
        molecule = args.molecule.strip()
        slug     = molecule.lower().replace(" ", "_")
        out_file = output_dir / f"{slug}_research.xlsx"

        print(f"\n{'━'*62}")
        print(f"  Molecule : {molecule.title()}")
        print(f"  Output   : {out_file}")
        print(f"{'━'*62}")

        label_rows, indication_rows = process_molecule(molecule, args.company)
        combined = write_molecule_excel(molecule, label_rows, indication_rows, str(out_file))

        print(f"\n✅  Done!")
        print(f"📄  File   : {out_file}")
        print(f"📊  Sheets : Clinical Efficacy ({len(label_rows)}) | "
              f"Drug Indication Research ({len(indication_rows)}) | "
              f"All Combined ({len(combined)})")
        return

    # ── ALL-MOLECULES MODE ────────────────────────────────────────────────
    print(f"\n{'━'*62}")
    print(f"  Mode     : ALL MOLECULES")
    print(f"  Output   : {output_dir}")
    print(f"{'━'*62}\n")

    molecules = fetch_all_molecules()
    if not molecules:
        sys.exit("❌  No molecules found in BigQuery.")

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_file  = args.master_file or f"all_molecules_research_{timestamp}.xlsx"
    master_path  = output_dir / master_file

    all_results: dict[str, list[dict]] = {}
    failed:      list[str]             = []

    total = len(molecules)
    for i, molecule in enumerate(molecules, 1):
        slug     = molecule.lower().replace(" ", "_")
        out_file = output_dir / f"{slug}_research.xlsx"

        print(f"\n{'═'*62}")
        print(f"  [{i}/{total}] {molecule.title()}")
        print(f"{'═'*62}")

        # Skip if already done
        if args.skip_existing and out_file.exists():
            print(f"  ⏭️  Skipping — file already exists: {out_file}")
            # Still try to load existing combined rows for the master file
            try:
                wb_existing = openpyxl.load_workbook(str(out_file), read_only=True, data_only=True)
                ws_comb = wb_existing["All Combined"]
                rows_existing = []
                headers_row = None
                for row_cells in ws_comb.iter_rows(min_row=3, values_only=True):
                    if headers_row is None:
                        headers_row = list(row_cells)
                        continue
                    if any(c for c in row_cells):
                        rows_existing.append(dict(zip(HEADERS, row_cells)))
                all_results[molecule] = rows_existing
                wb_existing.close()
            except Exception as e:
                print(f"  ⚠️  Could not read existing file for master: {e}")
                all_results[molecule] = []
            continue

        try:
            label_rows, indication_rows = process_molecule(molecule)
            combined = write_molecule_excel(molecule, label_rows, indication_rows, str(out_file))
            all_results[molecule] = combined

            print(f"\n  ✅ {molecule.title()} done → {out_file}")
            print(f"     Clinical Efficacy: {len(label_rows)} | "
                  f"Indication Research: {len(indication_rows)} | "
                  f"Combined: {len(combined)}")

        except Exception as e:
            print(f"\n  ❌ FAILED: {molecule} — {e}")
            failed.append(molecule)
            all_results[molecule] = []

    # Write master file
    write_master_excel(all_results, str(master_path))

    # Summary
    print(f"\n{'━'*62}")
    print(f"  Pipeline complete!")
    print(f"  Molecules processed : {total}")
    print(f"  Succeeded           : {total - len(failed)}")
    print(f"  Failed              : {len(failed)}")
    if failed:
        print(f"  Failed molecules    : {', '.join(failed)}")
    print(f"  Master file         : {master_path}")
    print(f"  Per-molecule files  : {output_dir}/<molecule>_research.xlsx")
    print(f"{'━'*62}\n")


if __name__ == "__main__":
    main()
