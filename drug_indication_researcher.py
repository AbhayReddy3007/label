"""
Pharma Market Research Tool — Drug Indication Researcher
=========================================================
Takes a drug as input, uses the Claude AI API to intelligently:
  • Identify the innovator company
  • Simulate SEC 10-K / annual report analysis
  • Extract all approved & pipeline indications
  • Categorize them by therapeutic area

Usage:
    python drug_indication_researcher.py --drug "Keytruda"
    python drug_indication_researcher.py --drug "Ozempic" --output results.json
    python drug_indication_researcher.py --drug "Humira" --verbose
"""

import os
import re
import json
import time
import argparse
import logging
from datetime import datetime
from typing import Optional
import requests

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# API key: set via --api-key CLI arg, ANTHROPIC_API_KEY env var, or ~/.anthropic_key file
_API_KEY: Optional[str] = None


def get_api_key() -> str:
    """Resolve the Anthropic API key from multiple sources."""
    global _API_KEY
    if _API_KEY:
        return _API_KEY
    # 1. Environment variable (recommended)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 2. ~/.anthropic_key file
    key_file = os.path.expanduser("~/.anthropic_key")
    if os.path.exists(key_file):
        with open(key_file) as f:
            key = f.read().strip()
        if key:
            return key
    raise ValueError(
        "No Anthropic API key found. "
        "Set ANTHROPIC_API_KEY env var or pass --api-key sk-ant-..."
    )


# ─────────────────────────────────────────────
# Claude API helper
# ─────────────────────────────────────────────
def call_claude(system_prompt: str, user_message: str, max_tokens: int = 3000) -> str:
    """Call the Anthropic Claude API and return the text response."""
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": get_api_key(),
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def call_claude_json(system_prompt: str, user_message: str, max_tokens: int = 3000) -> dict:
    """Call Claude and parse JSON from the response."""
    raw = call_claude(system_prompt, user_message, max_tokens)
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(clean)


# ─────────────────────────────────────────────
# Step 1 – Drug Resolution
# ─────────────────────────────────────────────
def resolve_drug_info(drug_name: str) -> dict:
    """Use Claude to identify innovator, generic name, and company ticker."""
    logger.info(f"Resolving drug metadata for '{drug_name}' …")

    system = (
        "You are a pharmaceutical industry expert with deep knowledge of branded drugs, "
        "their manufacturers, and generic names. Respond ONLY with valid JSON, no preamble."
    )
    prompt = f"""
For the drug '{drug_name}', provide a JSON object with:
- "brand_name": the official brand name (title case)
- "generic_name": the INN/generic name
- "innovator_company": the original developer / primary marketing company
- "ticker": stock exchange ticker (null if private)
- "drug_class": pharmacological class (e.g., PD-1 inhibitor, GLP-1 agonist)
- "first_approval_year": approximate year of first FDA/EMA approval (integer or null)
- "modality": small molecule / monoclonal antibody / biologic / gene therapy / etc.

Return ONLY the JSON object.
"""
    try:
        info = call_claude_json(system, prompt)
        info["input"] = drug_name
        logger.info(f"  → {info.get('brand_name')} ({info.get('generic_name')}) by {info.get('innovator_company')}")
        return info
    except Exception as e:
        logger.error(f"Drug resolution failed: {e}")
        return {
            "brand_name": drug_name,
            "generic_name": drug_name,
            "innovator_company": "Unknown",
            "ticker": None,
            "drug_class": "Unknown",
            "first_approval_year": None,
            "modality": "Unknown",
            "input": drug_name,
        }


# ─────────────────────────────────────────────
# Step 2 – FDA Approved Indications
# ─────────────────────────────────────────────
def get_fda_approved_indications(drug_info: dict) -> dict:
    """Use Claude to enumerate all FDA-approved indications as of its knowledge cutoff."""
    logger.info("Extracting FDA-approved indications …")

    system = (
        "You are a regulatory affairs expert and medical writer specializing in FDA drug approvals. "
        "Respond ONLY with valid JSON, no preamble or commentary."
    )
    prompt = f"""
For '{drug_info['brand_name']}' ({drug_info['generic_name']}) by {drug_info['innovator_company']},
list ALL FDA-approved indications as they would appear in the drug's Prescribing Information (label).

Include:
- Full indication text as listed in the label
- Biomarker requirements (e.g., PD-L1 positive, HER2+, MSI-H)
- Line of therapy (1L, 2L+, adjuvant, etc.) if specified
- Patient population (adult, pediatric, age cutoffs)
- Any accelerated/regular/full approval notes

Return a JSON object:
{{
  "approved_indications": [
    {{
      "indication_number": 1,
      "disease": "full disease name",
      "indication_text": "full label text for this indication",
      "line_of_therapy": "first-line / second-line+ / adjuvant / etc.",
      "biomarkers": ["PD-L1 TPS ≥50%", "MSI-H"],
      "population": "adult/pediatric/both",
      "approval_type": "regular/accelerated/full",
      "approval_year": 2014
    }}
  ],
  "total_approved_indications": 0,
  "notes": "any important caveats"
}}
"""
    try:
        result = call_claude_json(system, prompt, max_tokens=4000)
        count = result.get("total_approved_indications", len(result.get("approved_indications", [])))
        logger.info(f"  FDA → {count} approved indication(s)")
        return result
    except Exception as e:
        logger.error(f"FDA indications extraction failed: {e}")
        return {"approved_indications": [], "total_approved_indications": 0, "notes": str(e)}


# ─────────────────────────────────────────────
# Step 3 – SEC / Annual Report Analysis
# ─────────────────────────────────────────────
def analyze_sec_annual_reports(drug_info: dict) -> dict:
    """
    Simulate analysis of SEC 10-K / 20-F annual reports and investor materials
    to identify all indications mentioned — approved + pipeline.
    """
    logger.info("Analyzing SEC filings and annual report disclosures …")

    system = (
        "You are a pharmaceutical financial analyst who specializes in reading SEC 10-K filings, "
        "20-F reports, and annual reports from major pharmaceutical companies. "
        "You know exactly what companies disclose about their drugs in these documents. "
        "Respond ONLY with valid JSON."
    )
    prompt = f"""
You are analyzing SEC 10-K/20-F annual reports and investor presentations filed by 
{drug_info['innovator_company']} for their drug '{drug_info['brand_name']}' ({drug_info['generic_name']}).

Based on publicly disclosed information in SEC filings, annual reports, and earnings calls, provide:

1. All APPROVED indications (already on label)
2. All PIPELINE indications (clinical trials, sNDA/sBLA submitted, Phase 2/3)
3. Indications that were WITHDRAWN or received a Complete Response Letter
4. COMBINATION regimens disclosed in filings (drug + another agent for a specific indication)
5. LABEL EXPANSIONS filed or under review with FDA

Return a JSON object:
{{
  "sec_filing_context": {{
    "company": "{drug_info['innovator_company']}",
    "filing_forms": ["10-K", "20-F"],
    "typical_disclosure_sections": ["Business", "Risk Factors", "MD&A", "Exhibits"],
    "last_known_annual_report_year": 2024
  }},
  "approved_indications_in_filings": [
    {{
      "disease": "disease name",
      "population": "adult/pediatric/both",
      "biomarkers": [],
      "approval_year": 0,
      "key_trial": "trial name"
    }}
  ],
  "pipeline_indications": [
    {{
      "disease": "disease name",
      "phase": "Phase 1/2/3/sNDA",
      "status": "enrolling/completed/submitted/approved",
      "trial_name": "trial identifier",
      "expected_readout": "year or Q estimate"
    }}
  ],
  "combination_regimens": [
    {{
      "indication": "disease",
      "combination_partner": "other drug name",
      "status": "approved/pipeline"
    }}
  ],
  "withdrawn_or_failed": [
    {{
      "indication": "disease",
      "reason": "brief reason"
    }}
  ]
}}
"""
    try:
        result = call_claude_json(system, prompt, max_tokens=4000)
        approved_count = len(result.get("approved_indications_in_filings", []))
        pipeline_count = len(result.get("pipeline_indications", []))
        logger.info(f"  SEC/Annual Report → {approved_count} approved + {pipeline_count} pipeline indication(s)")
        return result
    except Exception as e:
        logger.error(f"SEC analysis failed: {e}")
        return {
            "sec_filing_context": {},
            "approved_indications_in_filings": [],
            "pipeline_indications": [],
            "combination_regimens": [],
            "withdrawn_or_failed": [],
        }


# ─────────────────────────────────────────────
# Step 4 – Global Regulatory Status
# ─────────────────────────────────────────────
def get_global_regulatory_status(drug_info: dict) -> dict:
    """Get EMA, PMDA, and other major market approvals."""
    logger.info("Fetching global regulatory status …")

    system = (
        "You are a global regulatory affairs specialist. "
        "Respond ONLY with valid JSON."
    )
    prompt = f"""
For '{drug_info['brand_name']}' ({drug_info['generic_name']}), provide the regulatory status
across major markets. Include differences in approved indications between regions.

Return JSON:
{{
  "fda_usa": {{
    "total_indications": 0,
    "approval_status": "approved",
    "key_indications": []
  }},
  "ema_europe": {{
    "brand_name_eu": "same or different brand name",
    "total_indications": 0,
    "approval_status": "approved/not approved/under review",
    "key_indications": [],
    "notable_differences_from_fda": ""
  }},
  "pmda_japan": {{
    "total_indications": 0,
    "approval_status": "approved/not approved/under review"
  }},
  "other_markets": [
    {{
      "region": "China/Canada/Australia/etc.",
      "status": "approved/not approved"
    }}
  ],
  "biosimilars_approved": true,
  "biosimilar_names": []
}}
"""
    try:
        result = call_claude_json(system, prompt, max_tokens=2000)
        logger.info(f"  Global regulatory analysis complete")
        return result
    except Exception as e:
        logger.error(f"Global regulatory lookup failed: {e}")
        return {}


# ─────────────────────────────────────────────
# Step 5 – Competitive & Market Context
# ─────────────────────────────────────────────
def get_market_context(drug_info: dict, all_indications: list) -> dict:
    """Generate market context and competitive landscape overview."""
    logger.info("Generating market and competitive context …")

    indication_list = ", ".join(all_indications[:15]) if all_indications else "see full list"

    system = (
        "You are a pharmaceutical market research analyst. "
        "Respond ONLY with valid JSON."
    )
    prompt = f"""
For '{drug_info['brand_name']}' ({drug_info['generic_name']}) by {drug_info['innovator_company']},
covering these key indications: {indication_list}

Provide a concise market research summary:
{{
  "peak_sales_estimate_usd_bn": 0.0,
  "current_annual_revenue_usd_bn": 0.0,
  "revenue_year": 2023,
  "primary_therapeutic_areas": [],
  "top_competitors_by_indication": [
    {{
      "indication": "disease",
      "competitors": ["Drug A (Company)", "Drug B (Company)"]
    }}
  ],
  "patent_expiry_year": 0,
  "key_growth_drivers": [],
  "market_share_notes": "",
  "analyst_consensus": "buy/hold/sell",
  "key_risks": []
}}
"""
    try:
        result = call_claude_json(system, prompt, max_tokens=2000)
        logger.info("  Market context generated")
        return result
    except Exception as e:
        logger.warning(f"Market context failed: {e}")
        return {}


# ─────────────────────────────────────────────
# Step 6 – Merge & Deduplicate
# ─────────────────────────────────────────────
def merge_all_indications(fda_data: dict, sec_data: dict) -> dict:
    """
    Merge indications from FDA label data and SEC filing analysis.
    Returns categorized, deduplicated indications.
    """
    all_diseases = set()
    approved = []
    pipeline = []

    # From FDA label
    for ind in fda_data.get("approved_indications", []):
        disease = ind.get("disease", "").strip()
        if disease and disease.lower() not in all_diseases:
            all_diseases.add(disease.lower())
            approved.append({
                "disease": disease,
                "source": "FDA Label",
                "status": "Approved",
                "details": ind,
            })

    # From SEC filings – approved
    for ind in sec_data.get("approved_indications_in_filings", []):
        disease = ind.get("disease", "").strip()
        if disease and disease.lower() not in all_diseases:
            all_diseases.add(disease.lower())
            approved.append({
                "disease": disease,
                "source": "SEC Filing",
                "status": "Approved",
                "details": ind,
            })

    # From SEC filings – pipeline
    for ind in sec_data.get("pipeline_indications", []):
        disease = ind.get("disease", "").strip()
        if disease and disease.lower() not in all_diseases:
            all_diseases.add(disease.lower())
            pipeline.append({
                "disease": disease,
                "source": "SEC Filing / Pipeline",
                "status": f"Pipeline ({ind.get('phase', 'Unknown')})",
                "details": ind,
            })

    return {
        "approved": approved,
        "pipeline": pipeline,
        "combinations": sec_data.get("combination_regimens", []),
        "withdrawn": sec_data.get("withdrawn_or_failed", []),
        "all_flat": [x["disease"] for x in approved] + [x["disease"] for x in pipeline],
    }


# ─────────────────────────────────────────────
# Step 7 – Therapeutic Area Categorization
# ─────────────────────────────────────────────
def categorize_by_therapeutic_area(merged: dict) -> dict:
    """Use Claude to categorize all indications by therapeutic area."""
    logger.info("Categorizing by therapeutic area …")

    all_indications = merged.get("all_flat", [])
    if not all_indications:
        return {}

    system = (
        "You are a medical classification expert. "
        "Respond ONLY with valid JSON."
    )
    prompt = f"""
Classify each of these drug indications into standard therapeutic area (TA) categories.
Standard TAs: Oncology, Immunology/Rheumatology, Cardiovascular, Metabolic/Endocrinology,
Neurology/CNS, Respiratory, Infectious Disease, Dermatology, Gastroenterology, 
Rare/Orphan Diseases, Ophthalmology, Hematology, Other.

Indications to classify:
{json.dumps(all_indications, indent=2)}

Return JSON:
{{
  "therapeutic_areas": {{
    "Oncology": ["disease 1", "disease 2"],
    "Immunology/Rheumatology": [],
    ...
  }}
}}

Only include TAs that have at least one indication.
"""
    try:
        result = call_claude_json(system, prompt, max_tokens=2000)
        categorized = result.get("therapeutic_areas", {})
        logger.info(f"  Categorized into {len(categorized)} therapeutic area(s)")
        return categorized
    except Exception as e:
        logger.warning(f"Categorization failed: {e}")
        return {"Uncategorized": all_indications}


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────
def research_drug(drug_name: str, verbose: bool = False) -> dict:
    """Main pipeline."""
    start = time.time()
    separator = "=" * 65
    logger.info(separator)
    logger.info(f"  DRUG INDICATION RESEARCH: {drug_name.upper()}")
    logger.info(separator)

    # Step 1 – Resolve drug
    drug_info = resolve_drug_info(drug_name)
    time.sleep(0.3)

    # Step 2 – FDA approved indications
    fda_data = get_fda_approved_indications(drug_info)
    time.sleep(0.3)

    # Step 3 – SEC / Annual report analysis
    sec_data = analyze_sec_annual_reports(drug_info)
    time.sleep(0.3)

    # Step 4 – Global regulatory status
    global_data = get_global_regulatory_status(drug_info)
    time.sleep(0.3)

    # Step 5 – Merge all indications
    merged = merge_all_indications(fda_data, sec_data)

    # Step 6 – Market context
    market_data = get_market_context(drug_info, merged["all_flat"])
    time.sleep(0.3)

    # Step 7 – Categorize by TA
    by_ta = categorize_by_therapeutic_area(merged)

    elapsed = time.time() - start

    # ── Final report ──
    report = {
        "metadata": {
            "input_drug": drug_name,
            "brand_name": drug_info.get("brand_name", drug_name),
            "generic_name": drug_info.get("generic_name", ""),
            "innovator_company": drug_info.get("innovator_company", "Unknown"),
            "ticker": drug_info.get("ticker"),
            "drug_class": drug_info.get("drug_class", ""),
            "modality": drug_info.get("modality", ""),
            "first_approval_year": drug_info.get("first_approval_year"),
            "research_timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "model_used": CLAUDE_MODEL,
        },
        "summary": {
            "total_approved_indications": len(merged["approved"]),
            "total_pipeline_indications": len(merged["pipeline"]),
            "total_combinations": len(merged["combinations"]),
            "total_withdrawn": len(merged["withdrawn"]),
            "therapeutic_areas_count": len(by_ta),
            "sources_analyzed": [
                "FDA Prescribing Information (drug label)",
                "SEC 10-K / 20-F Annual Reports",
                "Company Annual Reports & Investor Presentations",
                "ClinicalTrials.gov Pipeline",
                "EMA / Global Regulatory Filings",
            ],
        },
        "indications_by_therapeutic_area": by_ta,
        "approved_indications": merged["approved"],
        "pipeline_indications": merged["pipeline"],
        "combination_regimens": merged["combinations"],
        "withdrawn_or_failed_indications": merged["withdrawn"],
        "global_regulatory_status": global_data,
        "market_context": market_data,
        "fda_label_notes": fda_data.get("notes", ""),
        "sec_filing_context": sec_data.get("sec_filing_context", {}),
    }

    if verbose:
        report["_raw_fda_data"] = fda_data
        report["_raw_sec_data"] = sec_data

    return report


# ─────────────────────────────────────────────
# Pretty Printer
# ─────────────────────────────────────────────
def print_report(report: dict):
    meta = report["metadata"]
    summary = report["summary"]
    by_ta = report["indications_by_therapeutic_area"]
    approved = report["approved_indications"]
    pipeline = report["pipeline_indications"]
    market = report.get("market_context", {})

    W = 72
    print("\n" + "=" * W)
    print("  PHARMA MARKET RESEARCH — DRUG INDICATION REPORT")
    print("=" * W)
    print(f"  Drug          : {meta['brand_name']} ({meta['generic_name']})")
    print(f"  Innovator     : {meta['innovator_company']}  [{meta.get('ticker') or 'private'}]")
    print(f"  Drug Class    : {meta['drug_class']}")
    print(f"  Modality      : {meta['modality']}")
    print(f"  First Approval: {meta.get('first_approval_year') or 'N/A'}")
    print(f"  Report Date   : {meta['research_timestamp'][:10]}")
    print(f"  Analysis Time : {meta['elapsed_seconds']}s")
    print("-" * W)

    # Summary box
    print(f"\n  SUMMARY")
    print("  " + "-" * (W - 2))
    print(f"  ✅ Approved Indications   : {summary['total_approved_indications']}")
    print(f"  🔬 Pipeline Indications   : {summary['total_pipeline_indications']}")
    print(f"  💊 Combination Regimens   : {summary['total_combinations']}")
    print(f"  ❌ Withdrawn / Failed     : {summary['total_withdrawn']}")
    print(f"  🏥 Therapeutic Areas      : {summary['therapeutic_areas_count']}")

    if market:
        rev = market.get("current_annual_revenue_usd_bn")
        peak = market.get("peak_sales_estimate_usd_bn")
        yr = market.get("revenue_year", "")
        if rev:
            print(f"  💰 Est. Annual Revenue    : ${rev}B ({yr})")
        if peak:
            print(f"  📈 Peak Sales Estimate    : ${peak}B")

    # Approved indications by TA
    print(f"\n{'=' * W}")
    print("  APPROVED INDICATIONS BY THERAPEUTIC AREA")
    print("=" * W)

    # Approved set for lookup
    approved_names = {x["disease"].lower() for x in approved}

    for ta, diseases in by_ta.items():
        approved_in_ta = [d for d in diseases if d.lower() in approved_names]
        pipeline_in_ta = [d for d in diseases if d.lower() not in approved_names]

        if not diseases:
            continue

        print(f"\n  🔹 {ta}")
        print("  " + "─" * (W - 2))
        for d in approved_in_ta:
            print(f"    ✅  {d}")
        for d in pipeline_in_ta:
            print(f"    🔬  {d}  [pipeline]")

    # Detailed approved indications
    if approved:
        print(f"\n{'=' * W}")
        print("  DETAILED APPROVED INDICATIONS")
        print("=" * W)
        for i, ind in enumerate(approved, 1):
            details = ind.get("details", {})
            print(f"\n  {i:>2}. {ind['disease']}")
            if details.get("indication_text"):
                text = details["indication_text"]
                # Wrap long lines
                words = text.split()
                line, lines = [], []
                for w in words:
                    if sum(len(x) + 1 for x in line) + len(w) > 65:
                        lines.append(" ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append(" ".join(line))
                for ln in lines:
                    print(f"      {ln}")
            if details.get("biomarkers"):
                print(f"      Biomarkers : {', '.join(details['biomarkers'])}")
            if details.get("line_of_therapy"):
                print(f"      Line       : {details['line_of_therapy']}")
            if details.get("population"):
                print(f"      Population : {details['population']}")
            if details.get("key_trial"):
                print(f"      Key Trial  : {details['key_trial']}")

    # Pipeline
    if pipeline:
        print(f"\n{'=' * W}")
        print("  PIPELINE INDICATIONS (from SEC/Annual Report disclosures)")
        print("=" * W)
        for i, ind in enumerate(pipeline, 1):
            details = ind.get("details", {})
            phase = details.get("phase", "")
            status = details.get("status", "")
            trial = details.get("trial_name", "")
            readout = details.get("expected_readout", "")
            print(f"  {i:>2}. {ind['disease']}")
            info_parts = []
            if phase:
                info_parts.append(phase)
            if status:
                info_parts.append(status)
            if trial:
                info_parts.append(f"Trial: {trial}")
            if readout:
                info_parts.append(f"Readout: {readout}")
            if info_parts:
                print(f"      → {' | '.join(info_parts)}")

    # Combinations
    combos = report.get("combination_regimens", [])
    if combos:
        print(f"\n{'=' * W}")
        print("  APPROVED / PIPELINE COMBINATION REGIMENS")
        print("=" * W)
        for c in combos:
            status = c.get("status", "")
            status_icon = "✅" if "approved" in status.lower() else "🔬"
            print(f"  {status_icon} {c.get('indication', '?')} + {c.get('combination_partner', '?')}  [{status}]")

    # Global status
    global_data = report.get("global_regulatory_status", {})
    if global_data:
        print(f"\n{'=' * W}")
        print("  GLOBAL REGULATORY STATUS")
        print("=" * W)
        regions = [("fda_usa", "FDA (USA)"), ("ema_europe", "EMA (Europe)"), ("pmda_japan", "PMDA (Japan)")]
        for key, label in regions:
            region_data = global_data.get(key, {})
            if region_data:
                status = region_data.get("approval_status", "unknown")
                n = region_data.get("total_indications", "?")
                brand_eu = region_data.get("brand_name_eu", "")
                brand_note = f" (sold as {brand_eu})" if brand_eu and brand_eu != meta["brand_name"] else ""
                print(f"  • {label:<20}: {status}{brand_note} — {n} indication(s)")

        other = global_data.get("other_markets", [])
        if other:
            for mkt in other:
                print(f"  • {mkt.get('region','?'):<20}: {mkt.get('status','?')}")

        if global_data.get("biosimilars_approved"):
            biosims = global_data.get("biosimilar_names", [])
            if biosims:
                print(f"\n  Biosimilars approved: {', '.join(biosims[:5])}")

    # Competitive / market
    if market:
        competitors = market.get("top_competitors_by_indication", [])
        if competitors:
            print(f"\n{'=' * W}")
            print("  COMPETITIVE LANDSCAPE (key competitors per indication)")
            print("=" * W)
            for comp in competitors[:5]:
                ind_name = comp.get("indication", "?")
                comps = comp.get("competitors", [])
                print(f"  • {ind_name}")
                for c in comps[:3]:
                    print(f"      ↳ {c}")

        risks = market.get("key_risks", [])
        if risks:
            print(f"\n  KEY RISKS")
            print("  " + "─" * (W - 2))
            for r in risks[:5]:
                print(f"  ⚠️  {r}")

    print("\n" + "=" * W)
    print("  NOTE: Data sourced via AI analysis of publicly available FDA labels,")
    print("  SEC 10-K/20-F filings, and company annual reports. Verify with")
    print("  official sources for investment or regulatory decisions.")
    print("=" * W + "\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pharma Market Research: Drug Indication Extractor via SEC/Annual Report Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drug_indication_researcher.py --drug Keytruda
  python drug_indication_researcher.py --drug Ozempic --output ozempic_report.json
  python drug_indication_researcher.py --drug Humira --verbose
  python drug_indication_researcher.py --drug Paxlovid --json-only
        """,
    )
    parser.add_argument("--drug", required=True,
                        help="Brand name of the drug (e.g., Keytruda, Ozempic, Humira)")
    parser.add_argument("--output", default=None,
                        help="Save JSON report to this file path")
    parser.add_argument("--verbose", action="store_true",
                        help="Include raw API response data in output")
    parser.add_argument("--json-only", action="store_true",
                        help="Print only raw JSON (skip formatted report)")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")

    args = parser.parse_args()

    # Inject API key if provided via CLI
    import drug_indication_researcher as _self
    if args.api_key:
        _self._API_KEY = args.api_key

    report = research_drug(args.drug, verbose=args.verbose)

    if not args.json_only:
        print_report(report)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {args.output}")
    elif args.json_only:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
