"""
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

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv(override=True)

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GBQ_PROJECT     = os.getenv("GBQ_PROJECT")
GBQ_DATASET     = os.getenv("GBQ_DATASET")
GBQ_TABLE       = os.getenv("GBQ_TABLE", "clinical_efficacy")

GBQ_SERVICE_KEY = r"C:\Users\p90022569\Downloads\cognito-prod-394707-d38a0283cb16 (2).json"

# ----------------------------
# CLIENTS
# ----------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def _bq_client():
    credentials = service_account.Credentials.from_service_account_file(
        GBQ_SERVICE_KEY,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(project=GBQ_PROJECT, credentials=credentials)

# ----------------------------
# CHECKPOINT FUNCTIONS
# ----------------------------
def load_checkpoint(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# FETCH DATA
# ----------------------------
def fetch_rows(molecule_name):
    print("🔹 Connecting to BigQuery...")
    client = _bq_client()

    query = f"""
        SELECT molecule_name, company_name, source_url, phase, trial_id
        FROM `{GBQ_PROJECT}.{GBQ_DATASET}.{GBQ_TABLE}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name)
        ]
    )

    print("🔹 Running query...")
    results = client.query(query, job_config=job_config).result()

    rows = [dict(row) for row in results]
    print(f"✅ Retrieved {len(rows)} rows\n")

    return rows

# ----------------------------
# GEMINI ENRICHMENT
# ----------------------------
def enrich_with_gemini(rows, molecule_name):
    enriched = []
    total = len(rows)

    checkpoint_file = f"checkpoint_{molecule_name.lower().replace(' ', '_')}.json"
    checkpoint = load_checkpoint(checkpoint_file)

    print(f"📁 Loaded checkpoint → {len(checkpoint)} completed rows\n")

    for i, row in enumerate(rows, start=1):
        trial_id = row.get("trial_id")

        print(f"[{i}/{total}] Trial ID: {trial_id}")

        # ✅ skip already processed
        if trial_id in checkpoint:
            cp = checkpoint[trial_id]
            row["condition"] = ", ".join(cp["conditions"])
            row["trial_title"] = cp["trial_title"]

            print("   ⏭️ Skipped (checkpoint)")
            enriched.append(row)
            continue

        prompt = f"""
Extract clinical trial details:

Molecule: {row.get('molecule_name')}
Company: {row.get('company_name')}
Trial ID: {trial_id}
Phase: {row.get('phase')}
Source URL: {row.get('source_url')}

Return ONLY JSON:

{{
  "conditions": ["condition1", "condition2"],
  "trial_title": "<trial title>"
}}

Rules:
- Always return conditions as a list
- No explanation
"""

        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    system_instruction="Return valid JSON only"
                ),
            )

            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()

            data = json.loads(text)

            conditions = data.get("conditions", [])
            trial_title = data.get("trial_title", "N/A")

            # clean conditions
            if isinstance(conditions, list):
                conditions = list(set([c.strip() for c in conditions]))
            else:
                conditions = [str(conditions)]

            row["condition"] = ", ".join(conditions)
            row["trial_title"] = trial_title

            print(f"   ✅ {row['condition']}")

            # save checkpoint
            checkpoint[trial_id] = {
                "conditions": conditions,
                "trial_title": trial_title
            }
            save_checkpoint(checkpoint_file, checkpoint)

        except Exception as e:
            row["condition"] = "Error"
            row["trial_title"] = str(e)

            print(f"   ❌ Error: {e}")

        enriched.append(row)

    print(f"\n✅ Completed. Checkpoint saved → {checkpoint_file}\n")
    return enriched

# ----------------------------
# WRITE EXCEL ✅ UPDATED
# ----------------------------
def write_excel(rows, molecule_name):
    print("🔹 Writing Excel...")

    wb = openpyxl.Workbook()
    ws = wb.active

    headers = [
        "molecule_name",
        "company_name",   # ✅ added
        "condition",
        "trial_title",
        "trial_id",
        "phase"
    ]

    # header
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h)

    # data
    for r, row in enumerate(rows, start=2):
        ws.cell(r, 1, row.get("molecule_name"))
        ws.cell(r, 2, row.get("company_name"))   # ✅ added
        ws.cell(r, 3, row.get("condition"))
        ws.cell(r, 4, row.get("trial_title"))
        ws.cell(r, 5, row.get("trial_id"))
        ws.cell(r, 6, row.get("phase"))

        if r % 10 == 0:
            print(f"   ✍️ Written {r-1} rows...")

    file_name = f"{molecule_name.lower().replace(' ', '_')}_clinical_efficacy.xlsx"
    wb.save(file_name)

    print(f"✅ Excel saved → {file_name}\n")
    return file_name

# ----------------------------
# MAIN
# ----------------------------
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

# ✅ REQUIRED
if __name__ == "__main__":
    main()
