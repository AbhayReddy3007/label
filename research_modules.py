"""Module A (Clinical Efficacy) and Module B (Drug Indication Research).

These are the two data-gathering modules `label_expansion.py` orchestrates:

  Module A — run_clinical_efficacy():
      Clinical-trial rows fetched from BigQuery, enriched with Gemini
      (+ Google Search) to extract every indication each trial studies.
      Per-trial extraction is checkpointed to GCS so a re-run resumes
      instead of re-processing finished trials.

  Module B — run_indication_research():
      What the innovator/originator company itself says about the drug,
      researched directly with Gemini + Google Search across five source
      types: regulatory labels, investor presentations, press releases,
      pipeline pages, and SEC filings.

Both modules share the same Gemini plumbing, Primary/Secondary + therapy
area classification, and Ep/Et scoring (all defined in this file), and
return a flat list of row dicts in the common format that excel_writer.py
expects.

gcp_utils.py is shared, unmodified infrastructure (it also serves the
medical_potential pipeline) and only exposes generic `get_bq_client()` /
`get_gcs_client()` factories — it has no label-expansion-specific query or
checkpoint helpers. This module calls those two factories directly and
owns its own query + checkpoint logic, rather than adding pipeline-specific
functions to the shared file.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.cloud import bigquery
from google.genai import types as genai_types

import medical_potential.config as config
import gcp_utils
from indication_standardizer import standardize_indication

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  SECONDARY INDICATION CRITERIA (used by the classification prompt below)
# ══════════════════════════════════════════════════════════════════════════
SECONDARY_INDICATION_CRITERIA = """
A secondary indication qualifies ONLY if ALL of the following are true:

- The indication represents a true expansion — i.e., it is not part of the
  primary indication (for clinical assets) or currently approved label
  (for commercial assets)
- The indication is described at a clear disease-level definition, avoiding
  vague, overlapping, or synonymous representations
- The source must describe observed or measured outcomes in that specific
  indication (e.g., trial results, endpoint readouts, biomarker response),
  not just planned evaluation or exploratory intent

The following must NOT be considered secondary indications:
* Indications mentioned only as hypothesis, targets, or exploratory possibilities
* Pipeline indications without any data or outcomes
* Mechanism based assumptions without clinical or empirical validation
* Early discovery or preclinical signals without human data
* Any indication lacking traceable, verifiable evidence of results
"""


# ══════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT + PLUMBING
# ══════════════════════════════════════════════════════════════════════════
_gemini_client: genai.Client | None = None
_gemini_client_lock = threading.Lock()


def _get_gemini_client() -> genai.Client:
    """Return a lazily-created, cached Gemini client (thread-safe)."""
    global _gemini_client
    if _gemini_client is None:
        with _gemini_client_lock:
            if _gemini_client is None:
                if not config.GEMINI_API_KEY:
                    raise RuntimeError(
                        "GEMINI_API_KEY is not set. Add it to your .env file or environment."
                    )
                _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _safe_response_text(resp) -> str:
    """Safely extract text from a Gemini response.

    With Google Search grounding, the simple `resp.text` property can be
    None or raise. Walk the full candidates → parts tree and concatenate
    every text fragment found.
    """
    try:
        if resp.text is not None:
            return resp.text.strip()
    except Exception:
        pass

    texts = []
    try:
        for candidate in (resp.candidates or []):
            try:
                parts = candidate.content.parts
            except Exception:
                continue
            for part in (parts or []):
                try:
                    t = getattr(part, "text", None)
                    if t:
                        texts.append(t.strip())
                except Exception:
                    continue
    except Exception:
        pass

    return "\n".join(texts) if texts else ""


def _is_transient_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(k in err for k in (
        "503", "429", "unavailable", "overloaded", "resource exhausted",
        "rate limit", "deadline exceeded", "connection", "timeout", "502", "500",
    ))


def _gemini_generate(
    prompt: str,
    *,
    system_instruction: str = "",
    use_search: bool = True,
    model: str | None = None,
) -> str:
    """Call Gemini with optional Google Search grounding.

    Retries on transient errors (503, 429, connection, etc.) with
    exponential backoff. If `use_search` is True and the response comes
    back empty, retries once more without grounding.
    """
    client = _get_gemini_client()
    model = model or config.GEMINI_FLASH_PREVIEW_MODEL
    max_retries = config.GEMINI_MAX_RETRIES
    base_delay = config.GEMINI_RETRY_BASE_DELAY_SECONDS

    configs = []
    if use_search:
        configs.append(genai_types.GenerateContentConfig(
            temperature=0,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            system_instruction=system_instruction or "Return ONLY valid JSON.",
        ))
    configs.append(genai_types.GenerateContentConfig(
        temperature=0,
        system_instruction=system_instruction or "Return ONLY valid JSON.",
    ))

    for i, cfg in enumerate(configs):
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                text = _safe_response_text(resp)
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                if text:
                    return text
                break  # empty response — fall through to next config
            except Exception as e:
                last_err = e
                if _is_transient_error(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Gemini call failed (%s) — retrying in %ss (%d/%d)",
                                   e, delay, attempt + 1, max_retries)
                    time.sleep(delay)
                elif i == 0 and len(configs) > 1:
                    logger.info("Error with Search grounding (%s) — retrying without grounding", e)
                    break
                else:
                    raise
        else:
            if i == 0 and len(configs) > 1:
                logger.info("Retries exhausted with Search grounding — retrying without grounding")
                continue
            if last_err:
                raise last_err

        if i == 0 and len(configs) > 1:
            logger.info("Empty response with Search grounding — retrying without grounding")

    return ""


def _extract_json(text: str) -> dict | list:
    """Extract and parse the first JSON object or array from *text*,
    even if surrounded by prose, markdown fences, or whitespace."""
    if not text:
        raise ValueError("Cannot extract JSON from empty text")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\s*```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = stripped.find(open_ch)
        if start == -1:
            continue
        depth, end = 0, start
        in_str = False
        while end < len(stripped):
            ch = stripped[end]
            if ch == '"' and (end == 0 or stripped[end - 1] != '\\'):
                in_str = not in_str
            elif not in_str:
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(stripped[start:end + 1])
                        except json.JSONDecodeError:
                            break
            end += 1

    raise ValueError(f"No valid JSON found in response (first 200 chars): {text[:200]}")


# ══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION — Primary / Secondary + therapy area (LLM + Search)
# ══════════════════════════════════════════════════════════════════════════
def classify_indications_with_llm(molecule_name: str, unique_indications: list[str]) -> dict:
    """Use Gemini + Google Search to research the drug and classify each
    indication as Primary / Secondary, and assign therapy areas.

    Returns dict: indication_name(lower) → {indication_type, therapy_area, rationale}
    """
    if not unique_indications:
        return {}

    indications_json = json.dumps(unique_indications, indent=2)

    prompt = f"""
You are a pharmaceutical analyst. Your task is to research the drug
"{molecule_name}" and classify each of the following indications.

Drug / Molecule: {molecule_name}

Indications to classify:
{indications_json}

═══ STEP 1 — Research the drug ═══
Search the web to find:
  • What "{molecule_name}" is primarily approved / developed for
  • The FDA / EMA approved labels for this drug
  • The originator company's pipeline and clinical development programs

═══ STEP 2 — Classify each indication ═══
For EACH indication in the list above:

  indication_type:
    "Primary"   — this IS one of the drug's main approved or originally
                   intended indications (what it was designed and first
                   developed / approved to treat).
    "Secondary" — this is a label expansion or additional indication
                   beyond the primary use.
                   {SECONDARY_INDICATION_CRITERIA}

  therapy_area:
    Look at the INDICATION itself (the disease / condition) and determine
    which medical specialty treats it.
    Choose from: Metabolic, Cardiovascular, Oncology, Neuroscience,
    Immunology, Respiratory, Nephrology, Hepatology, Ophthalmology,
    Musculoskeletal, Gastroenterology, Infectious Disease, Dermatology,
    Hematology, Endocrinology, Rare Disease, or another appropriate
    broad therapeutic area.

    Examples:
      T2DM / Type 2 Diabetes / Obesity → Metabolic
      Heart Failure / MACE / CV Risk Reduction → Cardiovascular
      CKD / Diabetic Kidney Disease → Nephrology
      NASH / MASH → Hepatology
      NSCLC / Breast Cancer → Oncology
      Alzheimer's Disease → Neuroscience
      Rheumatoid Arthritis / Psoriasis → Immunology
      COPD / Asthma → Respiratory
      OSA (Obstructive Sleep Apnea) → Respiratory

Return ONLY valid JSON — no markdown fences, no explanation:
{{
  "classifications": [
    {{
      "indication": "<exact indication name from input list>",
      "indication_type": "Primary" or "Secondary",
      "therapy_area": "<therapy area>",
      "rationale": "<explain WHY this indication is Primary or Secondary for this drug — cite the specific evidence you found, e.g. FDA approval, original label, known label expansion, clinical data>"
    }}
  ]
}}
"""

    logger.info("Classifying %d unique indication(s) for %s (LLM + Search)",
                len(unique_indications), molecule_name)
    try:
        text = _gemini_generate(
            prompt,
            system_instruction=(
                "You are a pharmaceutical analyst. "
                "Search the web to find what this drug is approved for. "
                "Return ONLY valid JSON — no markdown, no explanation."
            ),
            use_search=True,
        )
        if not text:
            raise ValueError("Empty response from Gemini")
        data = _extract_json(text)
        classifications = data.get("classifications", [])

        result = {}
        for c in classifications:
            name = c.get("indication", "").strip()
            if name:
                result[name.lower()] = {
                    "indication_type": c.get("indication_type", "Secondary"),
                    "therapy_area": c.get("therapy_area", "Other"),
                    "rationale": c.get("rationale", ""),
                }
                logger.debug("%s: %s | %s", name, c.get("indication_type"), c.get("therapy_area"))

        # Retry missing indications individually
        for ind in unique_indications:
            if ind.lower() not in result:
                logger.info("Missing classification for '%s' — retrying individually", ind)
                result[ind.lower()] = _classify_single_indication(molecule_name, ind)

        return result

    except Exception as e:
        logger.warning("Bulk classification failed (%s) — classifying each indication individually", e)
        result = {}
        for ind in unique_indications:
            result[ind.lower()] = _classify_single_indication(molecule_name, ind)
        return result


def _classify_single_indication(molecule_name: str, indication: str) -> dict:
    """Classify a single indication via LLM + Google Search."""
    prompt = f"""
You are a pharmaceutical analyst. Research the drug "{molecule_name}".

Drug: {molecule_name}
Indication: {indication}

1. Is this indication "Primary" or "Secondary" for this drug?
   - "Primary": one of the drug's main approved / originally intended indications
   - "Secondary": a label expansion beyond the primary use

2. What therapy area does this INDICATION belong to?
   Choose from: Metabolic, Cardiovascular, Oncology, Neuroscience, Immunology,
   Respiratory, Nephrology, Hepatology, Ophthalmology, Musculoskeletal,
   Gastroenterology, Infectious Disease, Dermatology, Hematology,
   Endocrinology, Rare Disease, or another appropriate area.

3. Explain your reasoning.

Return ONLY valid JSON:
{{"indication_type": "Primary" or "Secondary", "therapy_area": "<area>", "rationale": "<why>"}}
"""
    try:
        text = _gemini_generate(
            prompt,
            system_instruction="Return ONLY valid JSON.",
            use_search=True,
        )
        data = _extract_json(text)
        cls = {
            "indication_type": data.get("indication_type", "Secondary"),
            "therapy_area": data.get("therapy_area", "Other"),
            "rationale": data.get("rationale", ""),
        }
        logger.debug("%s: %s | %s", indication, cls["indication_type"], cls["therapy_area"])
        return cls
    except Exception as ex:
        logger.warning("Individual classification failed for '%s': %s", indication, ex)
        return {"indication_type": "Secondary", "therapy_area": "Other",
                "rationale": f"Classification failed: {ex}"}


# ══════════════════════════════════════════════════════════════════════════
#  Ep SCORE — row-wise, based on trial phase
# ══════════════════════════════════════════════════════════════════════════
def compute_ep(phase) -> str:
    """Phase 4 or 3 → 5, Phase 2 → 4, Phase 1 → 3."""
    phase_str = str(phase).strip().lower() if phase else ""
    nums = re.findall(r"\d+", phase_str)
    if not nums:
        if any(kw in phase_str for kw in ("approved", "launched", "marketed")):
            return "5"
        return ""
    max_phase = max(int(n) for n in nums)
    if max_phase >= 3:
        return "5"
    elif max_phase == 2:
        return "4"
    elif max_phase == 1:
        return "3"
    return ""


# ══════════════════════════════════════════════════════════════════════════
#  Et SCORE — drug-level
# ══════════════════════════════════════════════════════════════════════════
def compute_et(enriched_rows: list[dict], molecule_name: str) -> str:
    """
    Multiple therapy areas                          → 5
    1 therapy area, 2+ secondary indications         → 4
    1 therapy area, 1 secondary (broad)              → 3
    1 therapy area, 1 secondary (very niche)         → 2
    1 therapy area, primary indication(s) only       → 1
    """
    drug_rows = [r for r in enriched_rows
                 if (r.get("molecule_name") or "").lower() == molecule_name.lower()]
    if not drug_rows:
        return ""

    therapy_areas = set(
        r["therapy_area"] for r in drug_rows
        if r.get("therapy_area") and (r["therapy_area"] or "").lower() not in ("other", "")
    )
    secondary_indications = list(set(
        r["indication"] for r in drug_rows
        if (r.get("indication_type") or "").lower() == "secondary"
    ))
    primary_indications = list(set(
        r["indication"] for r in drug_rows
        if (r.get("indication_type") or "").lower() == "primary"
    ))

    if len(therapy_areas) > 1:
        return "5"
    elif len(secondary_indications) >= 2:
        return "4"
    elif len(secondary_indications) == 1:
        is_niche = _check_niche(molecule_name, primary_indications, secondary_indications[0])
        return "2" if is_niche else "3"
    else:
        return "1"


def _check_niche(molecule: str, primary_list: list[str], secondary_indication: str) -> bool:
    """Use Gemini + Search to determine if a secondary indication is very niche."""
    primary_str = ", ".join(primary_list) if primary_list else "unknown"
    prompt = f"""
You are a pharmaceutical analyst.

Drug: {molecule}
Primary indication(s): {primary_str}
Secondary (label expansion) indication: {secondary_indication}

Is this label expansion "very niche"?
A label expansion is "very niche" if it targets a narrow, specialized patient
sub-population, a rare disease, an orphan indication, or a highly specific
clinical context that affects relatively few patients compared to the primary
indication.

Search the web to determine the patient population size for this secondary
indication compared to the primary indication.

Return ONLY valid JSON:
{{"is_niche": true}} or {{"is_niche": false}}
No explanation.
"""
    try:
        text = _gemini_generate(
            prompt,
            system_instruction="Return ONLY valid JSON.",
            use_search=True,
        )
        if not text:
            return False
        data = _extract_json(text)
        return bool(data.get("is_niche", False))
    except Exception as e:
        logger.warning("Niche check failed (%s) — defaulting to non-niche", e)
        return False


# ══════════════════════════════════════════════════════════════════════════
#  MODULE A — CLINICAL EFFICACY (BigQuery clinical trials + Gemini)
# ══════════════════════════════════════════════════════════════════════════
def _fetch_clinical_trials(molecule_name: str) -> list[dict]:
    """Fetch every clinical-trial row for *molecule_name* from CLINICAL_TRIALS_TABLE.

    Uses gcp_utils.get_bq_client() directly: gcp_utils.py is shared,
    unmodified infrastructure with no label-expansion-specific query
    helpers, so this module owns its own query.
    """
    table_id = f"{config.PROJECT_ID}.{config.BQ_DATASET_ID}.{config.CLINICAL_TRIALS_TABLE}"
    query = f"""
        SELECT molecule_name, company_name, source_url, phase, trial_id
        FROM `{table_id}`
        WHERE LOWER(molecule_name) = LOWER(@molecule)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name)]
    )

    logger.info("Querying %s for molecule '%s'", table_id, molecule_name)
    client = gcp_utils.get_bq_client()
    results = client.query(query, job_config=job_config).result()
    rows = [dict(row) for row in results]
    logger.info("Retrieved %d trial row(s) for '%s'", len(rows), molecule_name)
    return rows


def _checkpoint_blob_path(molecule_name: str) -> str:
    slug = molecule_name.strip().lower().replace(" ", "_")
    return f"{config.GCS_LE_CHECKPOINTS_PATH.rstrip('/')}/checkpoint_{slug}.json"


def _load_checkpoint(molecule_name: str) -> dict:
    """Load the trial-extraction checkpoint for *molecule_name* from GCS.

    Uses gcp_utils.get_gcs_client() directly, same reasoning as
    _fetch_clinical_trials above. Returns {} if no checkpoint exists yet
    or if it can't be read, so the pipeline always has a safe starting point.
    """
    blob_path = _checkpoint_blob_path(molecule_name)
    try:
        bucket = gcp_utils.get_gcs_client().bucket(config.GCS_BUCKET)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            logger.info("No existing checkpoint at gs://%s/%s — starting fresh", config.GCS_BUCKET, blob_path)
            return {}
        data = json.loads(blob.download_as_text())
        logger.info("Loaded checkpoint (%d entries) from gs://%s/%s", len(data), config.GCS_BUCKET, blob_path)
        return data
    except Exception:
        logger.exception("Failed to load checkpoint from gs://%s/%s — starting fresh",
                          config.GCS_BUCKET, blob_path)
        return {}


def _save_checkpoint(molecule_name: str, data: dict) -> None:
    """Persist the trial-extraction checkpoint for *molecule_name* to GCS.

    Failures are logged and swallowed — checkpointing is a resume
    optimisation, not something that should crash the pipeline.
    """
    blob_path = _checkpoint_blob_path(molecule_name)
    try:
        bucket = gcp_utils.get_gcs_client().bucket(config.GCS_BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(
            json.dumps(data, indent=2, default=str),
            content_type="application/json",
        )
        logger.debug("Checkpoint saved → gs://%s/%s (%d entries)", config.GCS_BUCKET, blob_path, len(data))
    except Exception:
        logger.exception("Failed to save checkpoint to gs://%s/%s", config.GCS_BUCKET, blob_path)


def _enrich_single_trial(row: dict, molecule_name: str, checkpoint: dict, cp_lock: threading.Lock):
    """Extract indications from a single trial. Classification happens later."""
    trial_id = row.get("trial_id")

    with cp_lock:
        if trial_id in checkpoint:
            cp = checkpoint[trial_id]
            return trial_id, cp["conditions"], cp["trial_title"], cp.get("phase", ""), row

    prompt = f"""
You are a clinical trial data assistant.

Trial details:
Molecule: {row.get('molecule_name')}
Company: {row.get('company_name')}
Trial ID: {trial_id}
Phase: {row.get('phase')}
Source URL: {row.get('source_url')}

═══ STEP 1 — Look up the trial ═══
Search for this trial using the Trial ID "{trial_id}" on ClinicalTrials.gov
or other clinical trial registries (e.g. EudraCT, WHO ICTRP).
Find the EXACT official trial title as registered.

If the source URL is provided, also check that URL for the trial title.

═══ STEP 2 — Extract indications ═══
From the trial record, extract ALL disease indications being studied.
Include BOTH the primary indication AND any secondary/exploratory indications
that have documented outcomes.

Look at:
- The official trial title (most reliable source of the indication)
- The trial's "Conditions" or "Diseases" field on the registry
- Primary and secondary outcome measures
- The trial description / brief summary

Common patterns:
- "in subjects with type 2 diabetes" → T2DM
- "cardiovascular outcomes" → CV Risk Reduction
- Trial acronyms like PIONEER, SUSTAIN, STEP, SELECT often indicate specific programs
- Outcome studies (e.g., cardiovascular, renal) count as indications

Return ONLY valid JSON:
{{
  "conditions": [
    {{"indication": "<disease or condition>", "rationale": "<why — cite the trial record field>"}},
    {{"indication": "<disease or condition>", "rationale": "<why>"}}
  ],
  "trial_title": "<EXACT official trial title as registered on the clinical trial registry>",
  "phase": "<Phase from the registry, e.g. Phase 1, Phase 2, Phase 3, Phase 4, Phase 2/3>"
}}

Rules:
- trial_title must be the EXACT title from the registry, not a summary or guess
- phase must match what the registry lists (e.g. Phase 3, Phase 2/Phase 3)
- Include ALL indications the trial is evaluating
- Always extract at least the primary indication
- No explanations outside the JSON
"""
    try:
        text = _gemini_generate(
            prompt,
            system_instruction=(
                "You are a clinical trial data assistant. "
                "Search for the trial on ClinicalTrials.gov or other registries "
                "to get the exact title. Return ONLY valid JSON."
            ),
            use_search=True,
        )
        if not text:
            raise ValueError("Empty response from Gemini")
        data = _extract_json(text)
        conditions = data.get("conditions", [])
        trial_title = data.get("trial_title", "N/A")
        extracted_phase = (data.get("phase") or "").strip()

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

        seen, deduped = set(), []
        for c in conditions:
            key = (c["indication"] or "").lower()
            if key and key not in seen:
                seen.add(key)
                deduped.append(c)
        conditions = deduped

        logger.info("%s: %s", trial_id, ", ".join(c["indication"] for c in conditions) or "no indications")

        with cp_lock:
            checkpoint[trial_id] = {
                "conditions": conditions, "trial_title": trial_title, "phase": extracted_phase,
            }
            _save_checkpoint(molecule_name, checkpoint)

    except Exception as e:
        conditions = []
        trial_title = str(e)
        extracted_phase = ""
        logger.error("%s: %s", trial_id, e)

    return trial_id, conditions, trial_title, extracted_phase, row


def _gemini_enrich_trials(rows: list[dict], molecule_name: str) -> list[dict]:
    """Extract indications from every trial (parallel + GCS-checkpointed),
    classify them, and score Ep/Et."""
    total = len(rows)
    checkpoint = _load_checkpoint(molecule_name)
    cp_lock = threading.Lock()

    logger.info("Checkpoint: %d trial(s) already completed", len(checkpoint))
    logger.info("Processing %d trial(s) with %d parallel worker(s)", total, config.MAX_WORKERS_TRIALS)

    # ── STEP 1: Parallel extraction ──────────────────────────────────────
    results = []
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_TRIALS) as executor:
        futures = {
            executor.submit(_enrich_single_trial, row, molecule_name, checkpoint, cp_lock): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                row = rows[idx]
                logger.error("Unexpected error for trial %s: %s", row.get("trial_id"), e)
                results.append((row.get("trial_id"), [], str(e), "", row))

    # Restore original order
    trial_order = {row.get("trial_id"): i for i, row in enumerate(rows)}
    results.sort(key=lambda r: trial_order.get(r[0], 0))

    # ── Unnest into flat rows (no classification yet) ────────────────────
    flat_rows = []
    for trial_id, conditions, trial_title, extracted_phase, row in results:
        phase = row.get("phase") or extracted_phase or ""

        if not conditions:
            flat_rows.append({
                "molecule_name": row.get("molecule_name"),
                "company_name": row.get("company_name"),
                "indication": "No indication found",
                "rationale": "",
                "trial_title": trial_title,
                "trial_id": trial_id,
                "phase": phase,
                "source_url": row.get("source_url"),
                "data_source": "Clinical Trials",
            })
            continue

        seen = set()
        for c in conditions:
            raw = c.get("indication", "")
            if not raw:
                continue
            std = standardize_indication(raw)
            if not std or std.lower() in ("error", "n/a", "no indication found", "none"):
                continue
            if std.lower() in seen:
                continue
            seen.add(std.lower())

            flat_rows.append({
                "molecule_name": row.get("molecule_name"),
                "company_name": row.get("company_name"),
                "indication": std,
                "rationale": c.get("rationale", ""),
                "trial_title": trial_title,
                "trial_id": trial_id,
                "phase": phase,
                "source_url": row.get("source_url"),
                "data_source": "Clinical Trials",
            })

    # ── STEP 2: Classify all unique indications in ONE LLM+Search call ───
    unique_indications = sorted(set(
        r["indication"] for r in flat_rows if r["indication"] != "No indication found"
    ))
    classification_map = classify_indications_with_llm(molecule_name, unique_indications)

    for row in flat_rows:
        cls = classification_map.get((row["indication"] or "").lower(), {})
        row["indication_type"] = cls.get("indication_type", "")
        row["therapy_area"] = cls.get("therapy_area", "")
        row["rationale"] = cls.get("rationale", "")

    # Override: Primary + non-Metabolic therapy area → Secondary
    for row in flat_rows:
        if ((row.get("indication_type") or "").lower() == "primary"
                and (row.get("therapy_area") or "").lower() != "metabolic"):
            row["indication_type"] = "Secondary"

    # ── STEP 3: Ep (row-wise, only for Primary/Secondary) ─────────────
    for row in flat_rows:
        ind_type = (row.get("indication_type") or "").lower()
        row["Ep"] = compute_ep(row.get("phase")) if ind_type in ("primary", "secondary") else ""

    # ── STEP 4: Et (drug-level, only Primary/Secondary rows) ──────────
    logger.info("Computing Et (drug-level expansion score)")
    scored_rows = [r for r in flat_rows if (r.get("indication_type") or "").lower() in ("primary", "secondary")]
    et_value = compute_et(scored_rows, molecule_name)
    for row in flat_rows:
        row["Et"] = et_value
    logger.info("Et = %s", et_value)

    logger.info("Clinical Efficacy: %d row(s) from %d trial(s)", len(flat_rows), total)
    return flat_rows


def run_clinical_efficacy(molecule_name: str) -> list[dict]:
    """Module A: fetch clinical-trial rows from BigQuery and enrich them."""
    logger.info("── Module A: Clinical Efficacy ──")
    rows = _fetch_clinical_trials(molecule_name)
    if not rows:
        logger.warning("No clinical-trial data found in BigQuery for '%s'", molecule_name)
        return []
    return _gemini_enrich_trials(rows, molecule_name)


# ══════════════════════════════════════════════════════════════════════════
#  MODULE B — DRUG INDICATION RESEARCH (innovator / investor / SEC sources)
# ══════════════════════════════════════════════════════════════════════════
def _identify_company(molecule: str) -> str:
    """Auto-detect the innovator/originator company via Gemini + Search."""
    try:
        text = _gemini_generate(
            (
                f"Who is the innovator pharmaceutical company that developed {molecule}? "
                f'Return ONLY JSON: {{"company": "<name>", "brand_names": ["name1"]}}'
            ),
            system_instruction="Return ONLY valid JSON. No markdown.",
            use_search=True,
        )
        if not text:
            return ""
        data = _extract_json(text)
        company = data.get("company", "")
        logger.info("Auto-detected innovator company: %s", company)
        return company
    except Exception as e:
        logger.warning("Could not auto-detect innovator company: %s", e)
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
                f'"source_url": "...", '
                f'"detail": "<approval year, population, dose>", '
                f'"rationale": "<why this indication was identified>"}}\n'
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
                f'"source_url": "...", '
                f'"detail": "<specific claim from the presentation>", '
                f'"rationale": "<why this indication was identified>"}}\n'
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
                f'"source_url": "...", '
                f'"detail": "<key announcement>", '
                f'"rationale": "<why this indication was identified>"}}\n'
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
                f'"source_url": "...", '
                f'"detail": "<status note>", '
                f'"rationale": "<why this indication was identified>"}}\n'
                f"]}}\n"
                f"Exclude Preclinical entries. No explanation, only JSON."
            ),
        },
        {
            "source_type": "SEC Filing",
            "prompt": (
                f"Search SEC EDGAR for filings by {company} "
                f"(10-K, 10-Q, 8-K, 20-F, 6-K) about {molecule}.\n\n"
                f"For EACH indication with actual clinical data or regulatory outcomes, "
                f"return a separate entry.\n\n"
                f"Return ONLY valid JSON:\n"
                f'{{"entries": [\n'
                f'  {{"indication": "...", "brand_name": "...", '
                f'"source_document": "<exact filing type and period>", '
                f'"source_url": "<SEC EDGAR URL>", '
                f'"detail": "<specific disclosed data>", '
                f'"rationale": "<why this indication was identified>"}}\n'
                f"]}}\n"
                f"Only indications with actual results — not forward-looking statements. "
                f"No explanation, only JSON."
            ),
        },
    ]


def _research_single_source(q: dict, molecule: str, company: str) -> tuple[str, list[dict]]:
    """Query a single innovator source. Returns (source_type, rows)."""
    source_type = q["source_type"]
    rows = []
    try:
        text = _gemini_generate(
            q["prompt"],
            system_instruction=(
                "You are a pharmaceutical analyst researching what the "
                "innovator company publicly says about this drug. "
                "Search the web and SEC EDGAR. Return ONLY valid JSON."
            ),
            use_search=True,
        )
        if not text:
            logger.warning("%s: empty response", source_type)
            return source_type, []
        entries = _extract_json(text).get("entries", [])
        if not entries:
            logger.info("%s: no entries", source_type)
            return source_type, []
        logger.info("%s: %d entries", source_type, len(entries))

        for e in entries:
            raw = e.get("indication", "").strip()
            if not raw:
                continue
            std = standardize_indication(raw)
            if not std or std.lower() in ("error", "n/a", "none", "no indication found"):
                continue
            rationale = (e.get("rationale") or e.get("detail") or "").strip()

            rows.append({
                "molecule_name": molecule.title(),
                "company_name": company,
                "indication": std,
                "rationale": rationale,
                "indication_type": "",  # filled later by classification
                "therapy_area": "",     # filled later by classification
                "trial_title": e.get("source_document", ""),
                "trial_id": e.get("brand_name", ""),
                "phase": "",
                "source_url": e.get("source_url", ""),
                "data_source": f"Innovator: {source_type}",
            })
    except json.JSONDecodeError:
        logger.warning("%s: JSON parse failed", source_type)
    except Exception as ex:
        logger.error("%s: %s", source_type, ex)

    return source_type, rows


def _innovator_research(molecule: str, company: str) -> list[dict]:
    queries = _build_innovator_queries(molecule, company)
    all_rows = []

    logger.info("Querying %d innovator source(s) in parallel", len(queries))

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_INNOVATOR) as executor:
        futures = {
            executor.submit(_research_single_source, q, molecule, company): q["source_type"]
            for q in queries
        }
        for future in as_completed(futures):
            source_type = futures[future]
            try:
                _, rows = future.result()
                all_rows.extend(rows)
            except Exception as ex:
                logger.error("%s: %s", source_type, ex)

    # Deduplicate
    seen, unique = set(), []
    for row in all_rows:
        key = (row["indication"], row["trial_title"], row["data_source"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    # ── Classify via LLM + Search ──────────────────────────────────────
    unique_indications = sorted(set(r["indication"] for r in unique))
    classification_map = classify_indications_with_llm(molecule, unique_indications)
    for row in unique:
        cls = classification_map.get((row["indication"] or "").lower(), {})
        row["indication_type"] = cls.get("indication_type", "")
        row["therapy_area"] = cls.get("therapy_area", "")
        row["rationale"] = cls.get("rationale", "")

    # Override: Primary + non-Metabolic therapy area → Secondary
    for row in unique:
        if ((row.get("indication_type") or "").lower() == "primary"
                and (row.get("therapy_area") or "").lower() != "metabolic"):
            row["indication_type"] = "Secondary"

    # ── Ep (row-wise, only for Primary/Secondary) ──────────────────────
    for row in unique:
        ind_type = (row.get("indication_type") or "").lower()
        row["Ep"] = compute_ep(row.get("phase")) if ind_type in ("primary", "secondary") else ""

    # ── Et (drug-level, only Primary/Secondary rows) ───────────────────
    scored_rows = [r for r in unique if (r.get("indication_type") or "").lower() in ("primary", "secondary")]
    if scored_rows:
        logger.info("Computing Et for innovator rows")
        et_value = compute_et(scored_rows, molecule)
        for row in unique:
            row["Et"] = et_value
        logger.info("Et = %s", et_value)

    logger.info("Indication Research: %d row(s) (deduped from %d)", len(unique), len(all_rows))
    return unique


def run_indication_research(molecule: str, company: str | None = None) -> list[dict]:
    """Module B: research what the innovator company says about the drug."""
    logger.info("── Module B: Drug Indication Research ──")
    if not company:
        logger.info("Identifying innovator company")
        company = _identify_company(molecule) or ""
    return _innovator_research(molecule, company)
