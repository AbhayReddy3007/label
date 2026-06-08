"""
Unified Drug Research Pipeline
===============================
Runs all three modules and writes every result into a single Excel file,
one sheet per module, plus a combined sheet.

All sheets use the SAME standardized column format:
    molecule_name | company_name | indication | indication_type |
    trial_title   | trial_id     | phase      | source_url      | data_source

Indications are standardized (e.g. "Type 2 diabetes mellitus" → "T2DM",
"overweight and obesity" → "Obesity") and classified as Primary or Secondary.

Each row has exactly ONE indication (multi-indication trials are unnested).

Usage:
    python run_all.py --molecule semaglutide
    python run_all.py --molecule semaglutide --sources pubmed,openalex
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
#  COMMON FORMAT — all sheets share these columns
# ══════════════════════════════════════════════════════════════════════════════

HEADERS    = ["molecule_name", "company_name", "indication", "indication_type",
              "trial_title", "trial_id", "phase", "source_url", "data_source"]
COL_WIDTHS = [18, 22, 28, 16, 50, 18, 10, 40, 16]


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

Extract from the given trial:
Molecule: {row.get('molecule_name')}
Company:  {row.get('company_name')}
Trial ID: {trial_id}
Phase:    {row.get('phase')}
Source URL: {row.get('source_url')}

Return ONLY valid JSON:
{{
  "conditions": ["condition1", "condition2"],
  "trial_title": "<trial title>"
}}

Rules:
- ALWAYS return conditions as a list
- If only one condition → list with one item
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
            processed = [{"indication": "No indication found", "indication_type": ""}]

        for ind in processed:
            flat.append({
                "molecule_name":   row.get("molecule_name"),
                "company_name":    row.get("company_name"),
                "indication":      ind["indication"],
                "indication_type": ind["indication_type"],
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
    print("\n━━━ [1/3] Clinical Efficacy (label.py) ━━━")
    rows = fetch_bq_rows(molecule)
    if not rows:
        print("  ❌ No data found in BigQuery")
        return []
    return _gemini_enrich_trials(rows, molecule, "label", "Clinical Trials")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE B — drug_indication_researcher.py
# ══════════════════════════════════════════════════════════════════════════════

def run_indication_researcher(molecule: str) -> list[dict]:
    print("\n━━━ [2/3] Drug Indication Research ━━━")
    rows = fetch_bq_rows(molecule)
    if not rows:
        print("  ❌ No data found in BigQuery")
        return []
    return _gemini_enrich_trials(rows, molecule, "indication", "Innovator Research")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE C — drug_literature_fetcher.py
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX = 50

SOURCE_COLORS = {
    "PubMed":           "2E6DA4",
    "Europe PMC":       "217A4E",
    "Semantic Scholar": "8B4513",
    "OpenAlex":         "6A0DAD",
    "CORE":             "B8860B",
}


def fetch_pubmed(drug: str, max_results: int = DEFAULT_MAX) -> list[dict]:
    SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    BASE   = "https://pubmed.ncbi.nlm.nih.gov"
    try:
        r = requests.get(SEARCH, params={
            "db": "pubmed", "term": f'"{drug}"[Title/Abstract]',
            "retmax": max_results, "retmode": "json",
        }, timeout=15)
        r.raise_for_status()
        pmids = r.json()["esearchresult"]["idlist"]
        if not pmids:
            return []
        articles = []
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]
            resp  = requests.get(FETCH, params={
                "db": "pubmed", "id": ",".join(batch),
                "retmode": "xml", "rettype": "abstract",
            }, timeout=30)
            resp.raise_for_status()
            articles.extend(_parse_pubmed_xml(resp.text, BASE))
            time.sleep(0.35)
        return articles
    except Exception as e:
        print(f"  ⚠️  PubMed error: {e}")
        return []


def _parse_pubmed_xml(xml_text: str, base_url: str) -> list[dict]:
    articles = []
    for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml_text, re.DOTALL):
        pmid     = _rx(r"<PMID[^>]*>(.*?)</PMID>", block)
        title    = _strip_tags(_rx(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL))
        journal  = _strip_tags(_rx(r"<Title>(.*?)</Title>", block, re.DOTALL))
        abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL)
        abstract = _strip_tags(" ".join(abstract_parts))
        if not abstract or not pmid:
            continue
        articles.append({
            "source": "PubMed", "title": title, "journal": journal,
            "abstract": abstract, "link": f"{base_url}/{pmid}/",
            "dedup_key": pmid,
        })
    return articles


def fetch_europepmc(drug: str, max_results: int = DEFAULT_MAX) -> list[dict]:
    BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    try:
        articles, page, page_size = [], 1, min(max_results, 100)
        while len(articles) < max_results:
            r = requests.get(BASE, params={
                "query": f'ABSTRACT:"{drug}" HAS_ABSTRACT:Y',
                "resultType": "core", "format": "json",
                "pageSize": page_size, "page": page,
            }, timeout=20)
            r.raise_for_status()
            results = r.json().get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                abstract = item.get("abstractText", "").strip()
                if not abstract:
                    continue
                pmid = item.get("pmid") or item.get("id", "")
                articles.append({
                    "source": "Europe PMC",
                    "title":   item.get("title", "N/A"),
                    "journal": item.get("journalTitle", "N/A"),
                    "abstract": abstract,
                    "link": f"https://europepmc.org/article/{item.get('source','MED')}/{item.get('id','')}",
                    "dedup_key": pmid or item.get("id", ""),
                })
            if len(results) < page_size:
                break
            page += 1
            time.sleep(0.3)
        return articles[:max_results]
    except Exception as e:
        print(f"  ⚠️  Europe PMC error: {e}")
        return []


def fetch_semantic_scholar(drug: str, max_results: int = DEFAULT_MAX) -> list[dict]:
    BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
    try:
        articles, offset, limit = [], 0, min(max_results, 100)
        while len(articles) < max_results:
            r = requests.get(BASE, params={
                "query": drug, "limit": limit, "offset": offset,
                "fields": "title,abstract,journal,externalIds,url",
            }, timeout=20)
            if r.status_code == 429:
                time.sleep(5)
                continue
            r.raise_for_status()
            papers = r.json().get("data", [])
            if not papers:
                break
            for p in papers:
                abstract = (p.get("abstract") or "").strip()
                if not abstract:
                    continue
                pmid  = (p.get("externalIds") or {}).get("PubMed", "")
                ss_id = p.get("paperId", "")
                articles.append({
                    "source": "Semantic Scholar",
                    "title":   p.get("title", "N/A"),
                    "journal": (p.get("journal") or {}).get("name", "N/A"),
                    "abstract": abstract,
                    "link":   p.get("url") or f"https://www.semanticscholar.org/paper/{ss_id}",
                    "dedup_key": pmid or ss_id,
                })
            if len(papers) < limit:
                break
            offset += limit
            time.sleep(0.5)
        return articles[:max_results]
    except Exception as e:
        print(f"  ⚠️  Semantic Scholar error: {e}")
        return []


def fetch_openalex(drug: str, max_results: int = DEFAULT_MAX) -> list[dict]:
    BASE = "https://api.openalex.org/works"
    try:
        articles, page, per_page = [], 1, min(max_results, 50)
        while len(articles) < max_results:
            r = requests.get(BASE, params={
                "search": drug, "filter": "has_abstract:true",
                "per-page": per_page, "page": page,
                "select": "id,title,abstract_inverted_index,primary_location,doi",
            }, headers={"User-Agent": "DrugLitFetcher/2.0 (research tool)"}, timeout=20)
            r.raise_for_status()
            works = r.json().get("results", [])
            if not works:
                break
            for w in works:
                abstract = _reconstruct_openalex_abstract(w.get("abstract_inverted_index", {}))
                if not abstract:
                    continue
                loc    = w.get("primary_location") or {}
                source = loc.get("source") or {}
                doi    = w.get("doi") or ""
                link   = f"https://doi.org/{doi.replace('https://doi.org/', '')}" if doi else w.get("id", "")
                articles.append({
                    "source": "OpenAlex",
                    "title":   w.get("title", "N/A"),
                    "journal": source.get("display_name", "N/A"),
                    "abstract": abstract,
                    "link":    link,
                    "dedup_key": doi or w.get("id", ""),
                })
            if len(works) < per_page:
                break
            page += 1
            time.sleep(0.3)
        return articles[:max_results]
    except Exception as e:
        print(f"  ⚠️  OpenAlex error: {e}")
        return []


def _reconstruct_openalex_abstract(inverted_index: dict) -> str:
    if not inverted_index:
        return ""
    pos_word = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            pos_word[pos] = word
    return " ".join(pos_word[p] for p in sorted(pos_word))


def fetch_core(drug: str, max_results: int = DEFAULT_MAX) -> list[dict]:
    BASE = "https://api.core.ac.uk/v3/search/works"
    try:
        r = requests.post(BASE, json={
            "q": drug, "limit": min(max_results, 100),
            "fields": ["title", "abstract", "journals", "sourceFulltextUrls", "doi"],
        }, timeout=20)
        r.raise_for_status()
        articles = []
        for item in r.json().get("results", []):
            abstract = (item.get("abstract") or "").strip()
            if not abstract:
                continue
            doi  = item.get("doi") or ""
            urls = item.get("sourceFulltextUrls") or []
            link = f"https://doi.org/{doi}" if doi else (urls[0] if urls else "https://core.ac.uk")
            journals     = item.get("journals") or []
            journal_name = journals[0].get("title", "N/A") if journals else "N/A"
            articles.append({
                "source": "CORE",
                "title":   item.get("title", "N/A"),
                "journal": journal_name,
                "abstract": abstract,
                "link":    link,
                "dedup_key": doi or str(item.get("id", "")),
            })
        return articles[:max_results]
    except Exception as e:
        print(f"  ⚠️  CORE error: {e}")
        return []


def deduplicate(articles: list[dict]) -> list[dict]:
    seen_keys, seen_titles, unique = set(), [], []
    for a in articles:
        key = (a.get("dedup_key") or "").strip().lower()
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        title_norm = re.sub(r"\W+", " ", (a.get("title") or "")).lower().strip()
        if any(_jaccard(title_norm, t) > 0.85 for t in seen_titles):
            continue
        seen_titles.append(title_norm)
        unique.append(a)
    return unique


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def extract_indications_lit(drug: str, title: str, abstract: str) -> list[str]:
    prompt = (
        f"You are a biomedical expert.\n\n"
        f"Drug: {drug}\nPaper title: {title}\nAbstract:\n{abstract}\n\n"
        f"Task: List ONLY the diseases, medical conditions, or clinical indications "
        f"this abstract mentions in the context of treating, preventing, or studying with {drug}.\n\n"
        f"Rules:\n"
        f"- Output a JSON array of short strings, e.g. [\"Type 2 Diabetes\",\"Obesity\"]\n"
        f"- Each entry: 1-5 words, title case\n"
        f"- Only include conditions where {drug} is a treatment/therapy/study subject\n"
        f"- If none found, return []\n"
        f"- Output ONLY valid JSON — no markdown, no explanation"
    )
    try:
        r = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512},
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = (
            r.json().get("candidates", [{}])[0]
             .get("content", {}).get("parts", [{}])[0]
             .get("text", "[]").strip()
        )
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return []


SOURCES = {
    "pubmed":          ("PubMed",           fetch_pubmed),
    "europepmc":       ("Europe PMC",       fetch_europepmc),
    "semanticscholar": ("Semantic Scholar", fetch_semantic_scholar),
    "openalex":        ("OpenAlex",         fetch_openalex),
    "core":            ("CORE",             fetch_core),
}


def run_literature(drug: str, sources_arg: str = "all") -> list[dict]:
    print("\n━━━ [3/3] Drug Literature Fetcher ━━━")

    if sources_arg.lower() == "all":
        active = list(SOURCES.keys())
    else:
        active = [s.strip().lower() for s in sources_arg.split(",") if s.strip().lower() in SOURCES]

    all_articles, source_stats = [], {}
    for key in active:
        label, fn = SOURCES[key]
        print(f"  📡 {label}…")
        fetched = fn(drug)
        for a in fetched:
            a["source"] = label
        source_stats[label] = len(fetched)
        all_articles.extend(fetched)
        print(f"     → {len(fetched)} articles")

    print(f"\n  🔗 Total raw: {len(all_articles)}")
    all_articles = deduplicate(all_articles)
    print(f"  ✂️  After dedup: {len(all_articles)}\n")

    print("  🤖 Extracting indications with Gemini…\n")
    flat = []
    for i, article in enumerate(all_articles, 1):
        print(f"    [{i:>3}/{len(all_articles)}] [{article['source']}] {article['title'][:60]}")
        raw_inds = extract_indications_lit(drug, article["title"], article["abstract"])
        time.sleep(0.2)

        processed = process_indications(drug, raw_inds)
        if not processed:
            processed = [{"indication": "No indication found", "indication_type": ""}]

        for ind in processed:
            flat.append({
                "molecule_name":   drug.title(),
                "company_name":    article["source"],
                "indication":      ind["indication"],
                "indication_type": ind["indication_type"],
                "trial_title":     article["title"],
                "trial_id":        "",
                "phase":           "",
                "source_url":      article["link"],
                "data_source":     "Literature",
            })

    print(f"\n  ✅ Literature rows: {len(flat)}\n")
    return flat


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


# ── utility ───────────────────────────────────────────────────────────────────

def _rx(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else ""

def _strip_tags(text):
    return re.sub(r"<[^>]+>", "", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified drug research pipeline → single Excel")
    parser.add_argument("--molecule", required=True, help="Drug/molecule name, e.g. semaglutide")
    parser.add_argument("--sources",  default="all",
                        help=f"Literature sources (default: all). Options: {', '.join(SOURCES)}")
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
    indication_rows = run_indication_researcher(molecule)
    lit_rows        = run_literature(molecule, args.sources)

    # ── assemble workbook ─────────────────────────────────────────────────────
    print(f"\n📊 Writing {out_file}…")
    wb = openpyxl.Workbook()

    # Sheet 1 — Clinical Efficacy
    ws1       = wb.active
    ws1.title = "Clinical Efficacy"
    _write_standard_sheet(ws1, label_rows, molecule, "Clinical Efficacy")

    # Sheet 2 — Drug Indication Research
    ws2       = wb.create_sheet("Drug Indication Research")
    _write_standard_sheet(ws2, indication_rows, molecule, "Drug Indication Research")

    # Sheet 3 — Literature
    ws3       = wb.create_sheet("Literature")
    _write_standard_sheet(ws3, lit_rows, molecule, "Drug Literature")

    # Sheet 4 — Combined (all three merged)
    combined  = label_rows + indication_rows + lit_rows
    ws4       = wb.create_sheet("All Combined")
    _write_standard_sheet(ws4, combined, molecule, "All Sources Combined")

    wb.save(out_file)

    print(f"\n✅  Done!")
    print(f"📄  File   : {out_file}")
    print(f"📊  Sheets : Clinical Efficacy ({len(label_rows)}) | "
          f"Drug Indication Research ({len(indication_rows)}) | "
          f"Literature ({len(lit_rows)}) | "
          f"All Combined ({len(combined)})")


if __name__ == "__main__":
    main()
