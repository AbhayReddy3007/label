"""
Clinical Efficacy — label.py
Fetches clinical trial data from BigQuery, enriches with Gemini,
standardizes indications, classifies primary/secondary.

Output: one indication per row in standardized format.

Usage:
    python label.py semaglutide
"""

import sys
import os
import json
import re
import argparse
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account
from google import genai
from google.genai import types
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from indication_standardizer import standardize_indication, classify_indication, process_indications

load_dotenv(override=True)

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GBQ_PROJECT     = os.getenv("GBQ_PROJECT")
GBQ_DATASET     = os.getenv("GBQ_DATASET")
GBQ_TABLE       = os.getenv("GBQ_TABLE", "clinical_efficacy")
GBQ_SERVICE_KEY = os.getenv(
    "GBQ_SERVICE_KEY",
    r"C:\Users\p90022569\Downloads\cognito-prod-394707-d38a0283cb16 (2).json",
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ══════════════════════════════════════════════════════════════════════════════
#  SECONDARY INDICATION CRITERIA
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

def _bq_client():
    credentials = service_account.Credentials.from_service_account_file(
        GBQ_SERVICE_KEY,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=GBQ_PROJECT, credentials=credentials)

def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def fetch_rows(molecule_name):
    print("🔹 Connecting to BigQuery...")
    client = _bq_client()
    query = f"""
        SELECT molecule_name, company_name, source_url, phase, trial_id
        FROM `{GBQ_PROJECT}.{GBQ_DATASET}.{GBQ_TABLE}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name)]
    )
    print("🔹 Running query...")
    results = client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    print(f"✅ Retrieved {len(rows)} rows\n")
    return rows

def enrich_with_gemini(rows, molecule_name):
    enriched = []
    total = len(rows)
    checkpoint_file = f"checkpoint_{molecule_name.lower().replace(' ', '_')}.json"
    checkpoint = load_checkpoint(checkpoint_file)
    print(f"📁 Loaded checkpoint → {len(checkpoint)} completed rows\n")

    for i, row in enumerate(rows, start=1):
        trial_id = row.get("trial_id")
        print(f"[{i}/{total}] Trial ID: {trial_id}")

        if trial_id in checkpoint:
            cp = checkpoint[trial_id]
            conditions = cp["conditions"]
            trial_title = cp["trial_title"]
            print("   ⏭️ Skipped (checkpoint)")
        else:
            prompt = f"""
You are a clinical trial data assistant.

Trial details:
Molecule: {row.get('molecule_name')}
Company: {row.get('company_name')}
Trial ID: {trial_id}
Phase: {row.get('phase')}
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
                import threading

                result_holder = {}
                error_holder  = {}

                def _call_gemini():
                    try:
                        resp = gemini_client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                temperature=0,
                                system_instruction="Return ONLY valid JSON output"
                            ),
                        )
                        result_holder["response"] = resp
                    except Exception as ex:
                        error_holder["error"] = ex

                thread = threading.Thread(target=_call_gemini, daemon=True)
                thread.start()
                thread.join(timeout=60)

                if thread.is_alive():
                    print(f"   ⏱️  Timeout (>60s) — skipping trial {trial_id}")
                    conditions  = []
                    trial_title = "Skipped (timeout)"
                elif "error" in error_holder:
                    raise error_holder["error"]
                else:
                    response    = result_holder["response"]
                    text        = response.text.strip()
                    text        = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                    data        = json.loads(text)
                    conditions  = data.get("conditions", [])
                    trial_title = data.get("trial_title", "N/A")
                    if isinstance(conditions, list):
                        conditions = list(set((c or "").strip() for c in conditions if c))
                    else:
                        conditions = [str(conditions)]
                    print(f"   ✅ {', '.join(conditions) if conditions else 'no secondary indications'}")
                    checkpoint[trial_id] = {"conditions": conditions, "trial_title": trial_title}
                    save_checkpoint(checkpoint_file, checkpoint)

            except Exception as e:
                conditions  = []
                trial_title = str(e)
                print(f"   ❌ Error: {e}")

        # ── Standardize + classify + unnest (one row per indication) ─────
        processed = process_indications(molecule_name, conditions)
        if not processed:
            processed = [{"indication": "No indication found", "indication_type": ""}]

        for ind in processed:
            enriched.append({
                "molecule_name":   row.get("molecule_name"),
                "company_name":    row.get("company_name"),
                "indication":      ind["indication"],
                "indication_type": ind["indication_type"],
                "therapy_area":    ind["therapy_area"],
                "trial_title":     trial_title if 'trial_title' in dir() else cp.get("trial_title", ""),
                "trial_id":        trial_id,
                "phase":           row.get("phase"),
                "source_url":      row.get("source_url"),
                "data_source":     "Clinical Trials",
            })

    print(f"\n✅ Completed. {len(enriched)} rows (unnested, standardized)\n")
    return enriched


# ── COMMON FORMAT HEADERS ────────────────────────────────────────────────────
HEADERS    = ["molecule_name", "company_name", "indication", "indication_type",
              "therapy_area", "trial_title", "trial_id", "phase", "source_url", "data_source"]
COL_WIDTHS = [18, 22, 28, 16, 18, 50, 18, 10, 40, 16]


def _thin_border():
    return Border(bottom=Side(style="thin", color="DDDDDD"),
                  right=Side(style="thin", color="DDDDDD"))


def write_excel(rows, molecule_name):
    print("🔹 Writing Excel...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Clinical Efficacy"

    thin     = _thin_border()
    hfill    = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")
    ncols    = len(HEADERS)

    # Title row
    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value     = f"Clinical Efficacy  —  {molecule_name.title()}"
    c.font      = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    # Header row
    for col, h in enumerate(HEADERS, 1):
        c           = ws.cell(row=2, column=col, value=h)
        c.font      = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill      = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border    = thin
    ws.row_dimensions[2].height = 22

    # Data rows
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

    file_name = f"{molecule_name.lower().replace(' ', '_')}_clinical_efficacy.xlsx"
    wb.save(file_name)
    print(f"✅ Excel saved → {file_name}\n")
    return file_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("molecule_name")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])
    molecule = args.molecule_name

    print("\n🚀 Pipeline Started\n")
    print("[1/3] Fetching data...")
    rows = fetch_rows(molecule)
    if not rows:
        print("❌ No data found")
        return
    print("[2/3] Enriching data...")
    enriched = enrich_with_gemini(rows, molecule)
    print("[3/3] Generating Excel...")
    output = write_excel(enriched, molecule)
    print("🎉 DONE")
    print(f"📄 File: {output}")
    print(f"📊 Rows processed: {len(enriched)}")

if __name__ == "__main__":
    main()
