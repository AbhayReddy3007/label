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

      Additionally:
        - Searches online for approved secondary indications with their
          trial phases.
        - Fetches the drug's Mechanism of Action (MOA) from BigQuery.
        - Uses Gemini+Search to find the OpenTargets Ensembl code for the MOA.
        - Queries the OpenTargets GraphQL API to find associated diseases
          and their association scores.

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

import requests
from google import genai
from google.cloud import bigquery
from google.genai import types as genai_types

import medical_potential.config as config
import medical_potential.gcp_utils as gcp_utils
from medical_potential.label_expansion.indication_standardizer import standardize_indications

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
def _classify_indication_batch(molecule_name: str, batch: list[str]) -> dict:
    """Classify a batch of indications in a single Gemini + Search call.

    Returns dict: indication_name(lower) → {indication_type, therapy_area, rationale}
    """
    indications_json = json.dumps(batch, indent=2)

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
        for ind in batch:
            if ind.lower() not in result:
                logger.info("Missing classification for '%s' — retrying individually", ind)
                result[ind.lower()] = _classify_single_indication(molecule_name, ind)

        return result

    except Exception as e:
        logger.warning("Batch classification failed (%s) — classifying each individually", e)
        result = {}
        for ind in batch:
            result[ind.lower()] = _classify_single_indication(molecule_name, ind)
        return result


def classify_indications_with_llm(molecule_name: str, unique_indications: list[str]) -> dict:
    """Use Gemini + Google Search to research the drug and classify each
    indication as Primary / Secondary, and assign therapy areas.

    Processes indications in batches of config.INDICATION_BATCH_SIZE.

    Returns dict: indication_name(lower) → {indication_type, therapy_area, rationale}
    """
    if not unique_indications:
        return {}

    batch_size = max(1, config.INDICATION_BATCH_SIZE)
    batches = [unique_indications[i:i + batch_size] for i in range(0, len(unique_indications), batch_size)]

    logger.info(
        "Classifying %d unique indication(s) for %s in %d batch(es) of up to %d",
        len(unique_indications), molecule_name, len(batches), batch_size,
    )

    result = {}
    for i, batch in enumerate(batches, 1):
        logger.info("Classification batch %d/%d (%d indications)", i, len(batches), len(batch))
        batch_result = _classify_indication_batch(molecule_name, batch)
        result.update(batch_result)

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
        SELECT molecule_name, company_name, source_url, phase, trial_id, trial_size
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


def _build_batch_trial_prompt(batch: list[dict]) -> str:
    """Build the Gemini prompt for a batch of trials."""
    trials_block = ""
    for i, row in enumerate(batch, 1):
        trials_block += f"""
Trial {i}:
  Trial ID : {row.get('trial_id')}
  Molecule : {row.get('molecule_name')}
  Company  : {row.get('company_name')}
  Phase    : {row.get('phase')}
  Source URL: {row.get('source_url')}
"""
    return f"""
You are a clinical trial data assistant.

Below are {len(batch)} clinical trial(s). For EACH trial:

═══ STEP 1 — Look up the trial ═══
Search for the trial using its Trial ID on ClinicalTrials.gov or other
clinical trial registries (EudraCT, WHO ICTRP). Find the EXACT official
title as registered. Also check the Source URL if provided.

═══ STEP 2 — Extract indications ═══
Extract ALL disease indications being studied in that trial — both the
primary indication and any secondary/exploratory indications with
documented outcomes. Look at:
- The official trial title (most reliable)
- The trial's "Conditions" or "Diseases" field on the registry
- Primary and secondary outcome measures
- The trial description / brief summary

Common patterns:
- "in subjects with type 2 diabetes" → T2DM
- "cardiovascular outcomes" → CV Risk Reduction
- Trial acronyms like PIONEER, SUSTAIN, STEP, SELECT often indicate programs
- Outcome studies (cardiovascular, renal) count as indications

Trials:
{trials_block}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "trials": [
    {{
      "trial_id": "<Trial ID from input>",
      "trial_title": "<EXACT official title from the registry>",
      "phase": "<Phase from the registry, e.g. Phase 3, Phase 2/3>",
      "conditions": [
        {{"indication": "<disease or condition>", "rationale": "<cite the registry field>"}},
        ...
      ]
    }},
    ...
  ]
}}

Rules:
- Return one entry per trial, in the same order as the input.
- trial_title must be the EXACT registry title, not a summary or guess.
- phase must match what the registry lists.
- Include ALL indications the trial evaluates; always include at least the primary.
- No explanations outside the JSON.
"""


def _normalize_conditions(raw_conditions) -> list[dict]:
    """Normalize a conditions value from Gemini into a clean list of dicts."""
    if not isinstance(raw_conditions, list):
        raw_conditions = [{"indication": str(raw_conditions), "rationale": ""}]

    normalized = []
    for c in raw_conditions:
        if isinstance(c, dict):
            normalized.append({
                "indication": (c.get("indication") or "").strip(),
                "rationale": (c.get("rationale") or "").strip(),
            })
        elif isinstance(c, str) and c.strip():
            normalized.append({"indication": c.strip(), "rationale": ""})

    seen, deduped = set(), []
    for c in normalized:
        key = (c["indication"] or "").lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _enrich_trial_batch(
    batch: list[dict],
    molecule_name: str,
    checkpoint: dict,
    cp_lock: threading.Lock,
) -> list[tuple]:
    """Extract indications for a batch of trials in a single Gemini call.

    Trials already in the checkpoint are skipped and returned directly.
    Returns a list of (trial_id, conditions, trial_title, phase, row) tuples
    in the same order as *batch*.
    """
    # Split into cached and uncached
    cached_results = {}
    uncached_rows = []
    with cp_lock:
        for row in batch:
            tid = row.get("trial_id")
            if tid in checkpoint:
                cp = checkpoint[tid]
                cached_results[tid] = (tid, cp["conditions"], cp["trial_title"], cp.get("phase", ""), row)
            else:
                uncached_rows.append(row)

    if not uncached_rows:
        return [cached_results[row.get("trial_id")] for row in batch]

    # Single Gemini call for all uncached trials in the batch
    batch_results: dict[str, tuple] = {}
    try:
        text = _gemini_generate(
            _build_batch_trial_prompt(uncached_rows),
            system_instruction=(
                "You are a clinical trial data assistant. "
                "Search ClinicalTrials.gov for each trial ID to get its exact title. "
                "Return ONLY valid JSON."
            ),
            use_search=True,
        )
        if not text:
            raise ValueError("Empty response from Gemini")

        data = _extract_json(text)
        trial_entries = data.get("trials", [])

        # Index by trial_id
        gemini_map = {e.get("trial_id"): e for e in trial_entries if e.get("trial_id")}

        for row in uncached_rows:
            tid = row.get("trial_id")
            entry = gemini_map.get(tid)
            if not entry:
                logger.warning("Batch response missing trial %s — recording as empty", tid)
                batch_results[tid] = (tid, [], "N/A", "", row)
                continue

            conditions = _normalize_conditions(entry.get("conditions", []))
            trial_title = (entry.get("trial_title") or "N/A").strip()
            extracted_phase = (entry.get("phase") or "").strip()

            logger.info("%s: %s", tid,
                        ", ".join(c["indication"] for c in conditions) or "no indications")

            with cp_lock:
                checkpoint[tid] = {
                    "conditions": conditions,
                    "trial_title": trial_title,
                    "phase": extracted_phase,
                }
            batch_results[tid] = (tid, conditions, trial_title, extracted_phase, row)

        # Save checkpoint once for the whole batch
        with cp_lock:
            _save_checkpoint(molecule_name, checkpoint)

    except Exception as e:
        logger.error("Batch Gemini call failed (%s) — marking %d trial(s) as empty", e, len(uncached_rows))
        for row in uncached_rows:
            tid = row.get("trial_id")
            if tid not in batch_results:
                batch_results[tid] = (tid, [], str(e), "", row)

    # Return in original batch order
    final = []
    for row in batch:
        tid = row.get("trial_id")
        final.append(cached_results.get(tid) or batch_results.get(tid) or (tid, [], "", "", row))
    return final


def _gemini_enrich_trials(rows: list[dict], molecule_name: str) -> list[dict]:
    """Extract indications from every trial (parallel + GCS-checkpointed),
    classify them, and score Ep/Et."""
    total = len(rows)
    checkpoint = _load_checkpoint(molecule_name)
    cp_lock = threading.Lock()

    batch_size = max(1, config.TRIAL_BATCH_SIZE)
    batches = [rows[i:i + batch_size] for i in range(0, total, batch_size)]

    logger.info("Checkpoint: %d trial(s) already completed", len(checkpoint))
    logger.info(
        "Processing %d trial(s) in %d batch(es) of up to %d (parallelism: %d workers)",
        total, len(batches), batch_size, config.MAX_WORKERS_TRIALS,
    )

    # ══ STEP 1: Parallel batch extraction ════════════════════════════════════
    # Each worker handles one batch (= one Gemini call for TRIAL_BATCH_SIZE trials).
    batch_outputs: list = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_TRIALS) as executor:
        futures = {
            executor.submit(_enrich_trial_batch, batch, molecule_name, checkpoint, cp_lock): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                batch_outputs[i] = future.result()
            except Exception as e:
                logger.error("Unexpected error in batch %d: %s", i, e)
                batch_outputs[i] = [
                    (row.get("trial_id"), [], str(e), "", row) for row in batches[i]
                ]

    # Flatten batches in original order
    results = [item for batch in batch_outputs for item in batch]

    # ── Collect all raw indications and batch-standardise in one call ────
    all_raws = [
        c.get("indication", "")
        for _, conditions, _, _, _ in results
        for c in (conditions or [])
        if c.get("indication", "")
    ]
    standardization_map = standardize_indications(all_raws)

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
                "trial_size": row.get("trial_size"),
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
            std = standardization_map.get(raw, raw)
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
                "trial_size": row.get("trial_size"),
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

    # TA-I (Therapy Area - Indication) — only for Secondary indications
    for row in flat_rows:
        if (row.get("indication_type") or "").lower() == "secondary":
            ta = (row.get("therapy_area") or "").strip()
            ind = (row.get("indication") or "").strip()
            row["TA-I"] = f"{ta} - {ind}" if ta and ind else ""
        else:
            row["TA-I"] = ""

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
                f'Return ONLY JSON: {{"company": "<n>", "brand_names": ["name1"]}}'
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
            rationale = (e.get("rationale") or e.get("detail") or "").strip()

            rows.append({
                "molecule_name": molecule.title(),
                "company_name": company,
                "indication": raw,      # standardised in batch by _innovator_research
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


# ══════════════════════════════════════════════════════════════════════════
#  APPROVED SECONDARY INDICATIONS LOOKUP (Gemini + Search)
# ══════════════════════════════════════════════════════════════════════════
def _search_approved_secondary_indications(molecule: str, company: str) -> list[dict]:
    """Search online for approved secondary (label-expansion) indications
    for the given drug and return rows with their trial phases."""
    logger.info("Searching for approved secondary indications for '%s'", molecule)
    prompt = f"""
You are a pharmaceutical regulatory analyst.

Drug / Molecule: {molecule}
Innovator Company: {company}

Search the web for ALL approved secondary indications (label expansions) for
"{molecule}". These are indications that were approved AFTER the original /
primary approval.

For EACH approved secondary indication, provide:
  - The exact indication name
  - The phase at which it is (e.g. "Approved", "Phase 3", "Phase 2", "Phase 1")
  - The approval year (if approved)
  - The regulatory body (FDA, EMA, etc.)
  - A source URL

Return ONLY valid JSON — no markdown, no explanation:
{{
  "approved_secondary_indications": [
    {{
      "indication": "<canonical indication name>",
      "phase": "<Approved / Phase 3 / Phase 2 / Phase 1>",
      "approval_year": "<year or empty string if not yet approved>",
      "regulatory_body": "<FDA / EMA / etc.>",
      "source_url": "<URL>",
      "rationale": "<brief explanation of the label expansion>"
    }}
  ]
}}
"""
    rows = []
    try:
        text = _gemini_generate(
            prompt,
            system_instruction=(
                "You are a pharmaceutical regulatory analyst. "
                "Search the web for approved label expansions. "
                "Return ONLY valid JSON."
            ),
            use_search=True,
        )
        if not text:
            logger.warning("Empty response for approved secondary indications")
            return []
        data = _extract_json(text)
        entries = data.get("approved_secondary_indications", [])
        logger.info("Found %d approved secondary indication(s)", len(entries))

        for e in entries:
            raw = (e.get("indication") or "").strip()
            if not raw:
                continue
            phase = (e.get("phase") or "").strip()
            approval_year = (e.get("approval_year") or "").strip()
            reg_body = (e.get("regulatory_body") or "").strip()
            rationale = (e.get("rationale") or "").strip()
            detail = f"{reg_body} {approval_year}".strip() if reg_body or approval_year else ""
            if detail and rationale:
                rationale = f"{detail} — {rationale}"
            elif detail:
                rationale = detail

            rows.append({
                "molecule_name": molecule.title(),
                "company_name": company,
                "indication": raw,
                "rationale": rationale,
                "indication_type": "",
                "therapy_area": "",
                "trial_title": f"Approved Label Expansion ({reg_body} {approval_year})".strip(),
                "trial_id": "",
                "phase": phase if phase else "Approved",
                "source_url": e.get("source_url", ""),
                "data_source": "Approved Secondary Indications",
            })
    except Exception as ex:
        logger.error("Approved secondary indications search failed: %s", ex)

    return rows


# ══════════════════════════════════════════════════════════════════════════
#  MOA LOOKUP (BigQuery)
# ══════════════════════════════════════════════════════════════════════════
def _fetch_moa_from_bq(molecule_name: str) -> list[str]:
    """Fetch the Mechanism(s) of Action for *molecule_name* from MOA_LOOKUP_TABLE.

    Returns a list of individual MOA strings (already split on ';').
    """
    if not config.MOA_LOOKUP_TABLE:
        logger.warning("MOA_LOOKUP_TABLE is not set — skipping MOA lookup")
        return []

    table_id = f"{config.PROJECT_ID}.{config.BQ_DATASET_ID}.{config.MOA_LOOKUP_TABLE}"
    query = f"""
        SELECT DISTINCT
            Cleaned_Generic_Name,
            moa_individual AS Mechanism_of_Action
        FROM `{table_id}`,
             UNNEST(SPLIT(Mechanism_of_Action, ';')) AS moa_individual
        WHERE LOWER(Cleaned_Generic_Name) = LOWER(@molecule)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("molecule", "STRING", molecule_name)]
    )

    logger.info("Querying MOA from %s for molecule '%s'", table_id, molecule_name)
    try:
        client = gcp_utils.get_bq_client()
        results = client.query(query, job_config=job_config).result()
        moas = []
        for row in results:
            moa = (row.get("Mechanism_of_Action") or "").strip()
            if moa:
                moas.append(moa)
        logger.info("Retrieved %d MOA(s) for '%s': %s", len(moas), molecule_name, moas)
        return moas
    except Exception as e:
        logger.error("Failed to fetch MOA from BigQuery: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════
#  OPENTARGETS INTEGRATION
# ══════════════════════════════════════════════════════════════════════════
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _find_opentargets_target_id(moa: str) -> str | None:
    """Use Gemini + Search to find the Ensembl gene ID (ENSG...) for a given
    Mechanism of Action target name, suitable for querying OpenTargets."""
    prompt = f"""
You are a bioinformatics expert.

Mechanism of Action (MOA): "{moa}"

From this MOA, identify the primary molecular target (protein / gene).
Then find its Ensembl Gene ID (format: ENSG followed by 11 digits,
e.g. ENSG00000169083) which is used on the OpenTargets Platform
(https://platform.opentargets.org/).

Search the web to find the correct Ensembl Gene ID for this target.

Return ONLY valid JSON — no markdown, no explanation:
{{"target_name": "<gene/protein name>", "ensembl_id": "<ENSG...>"}}
If you cannot determine the Ensembl ID, return:
{{"target_name": "<gene/protein name>", "ensembl_id": ""}}
"""
    try:
        text = _gemini_generate(
            prompt,
            system_instruction="Return ONLY valid JSON.",
            use_search=True,
        )
        if not text:
            return None
        data = _extract_json(text)
        ensembl_id = (data.get("ensembl_id") or "").strip()
        target_name = (data.get("target_name") or "").strip()
        if ensembl_id and ensembl_id.startswith("ENSG"):
            logger.info("MOA '%s' → target '%s' → Ensembl ID: %s", moa, target_name, ensembl_id)
            return ensembl_id
        else:
            logger.warning("Could not resolve Ensembl ID for MOA '%s' (target: '%s')", moa, target_name)
            return None
    except Exception as e:
        logger.warning("Failed to find OpenTargets target ID for MOA '%s': %s", moa, e)
        return None


def _query_opentargets_diseases(ensembl_id: str, max_diseases: int = 200) -> list[dict]:
    """Query the OpenTargets GraphQL API for diseases associated with a target.

    Returns list of dicts with keys: disease_id, disease_name, association_score,
    therapy_areas.
    """
    query_string = """
    query targetDiseases($ensemblId: String!, $size: Int!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        associatedDiseases(page: {size: $size, index: 0}) {
          count
          rows {
            score
            disease {
              id
              name
              therapeuticAreas {
                id
                name
              }
            }
          }
        }
      }
    }
    """
    variables = {"ensemblId": ensembl_id, "size": max_diseases}

    logger.info("Querying OpenTargets for diseases associated with %s", ensembl_id)
    try:
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query_string, "variables": variables},
            timeout=30,
        )
        resp.raise_for_status()
        api_data = resp.json()

        target_data = api_data.get("data", {}).get("target")
        if not target_data:
            logger.warning("No target data returned from OpenTargets for %s", ensembl_id)
            return []

        assoc = target_data.get("associatedDiseases", {})
        total = assoc.get("count", 0)
        rows = assoc.get("rows", [])
        logger.info(
            "OpenTargets: %s (%s) — %d associated diseases (returned %d)",
            target_data.get("approvedSymbol", "?"),
            ensembl_id,
            total,
            len(rows),
        )

        diseases = []
        for row in rows:
            disease = row.get("disease", {})
            ta_list = disease.get("therapeuticAreas", []) or []
            ta_names = [ta.get("name", "") for ta in ta_list if ta.get("name")]
            diseases.append({
                "disease_id": disease.get("id", ""),
                "disease_name": disease.get("name", ""),
                "association_score": round(row.get("score", 0.0), 4),
                "therapy_areas": ta_names,
            })
        return diseases

    except requests.RequestException as e:
        logger.error("OpenTargets API request failed: %s", e)
        return []
    except Exception as e:
        logger.error("Unexpected error querying OpenTargets: %s", e)
        return []


def _build_opentargets_rows(
    molecule: str,
    company: str,
    moa: str,
    ensembl_id: str,
    diseases: list[dict],
) -> list[dict]:
    """Convert OpenTargets disease associations into row dicts matching the
    common format for excel_writer.py."""
    rows = []
    for d in diseases:
        disease_name = d.get("disease_name", "")
        if not disease_name:
            continue
        ta_names = d.get("therapy_areas", [])
        ta_str = ", ".join(ta_names) if ta_names else ""
        score = d.get("association_score", 0.0)

        rows.append({
            "molecule_name": molecule.title(),
            "company_name": company,
            "indication": disease_name,
            "rationale": f"OpenTargets association (score: {score}) via MOA: {moa}",
            "indication_type": "",
            "therapy_area": ta_str,
            "trial_title": f"OpenTargets: {ensembl_id}",
            "trial_id": d.get("disease_id", ""),
            "phase": "",
            "source_url": f"https://platform.opentargets.org/target/{ensembl_id}",
            "data_source": "OpenTargets",
            "moa": moa,
            "opentargets_score": score,
        })
    return rows


def _match_indications_to_opentargets(
    pipeline_indications: list[str],
    ot_disease_names: list[str],
) -> dict[str, str]:
    """Use Gemini to match pipeline indication names to OpenTargets disease names.

    Returns dict: pipeline_indication(lower) → matched OpenTargets disease name.
    Only contains entries where a confident match was found.
    """
    if not pipeline_indications or not ot_disease_names:
        return {}

    # Deduplicate inputs
    unique_pipeline = sorted(set(pipeline_indications))
    unique_ot = sorted(set(ot_disease_names))

    pipeline_json = json.dumps(unique_pipeline, indent=2)
    ot_json = json.dumps(unique_ot, indent=2)

    prompt = f"""
You are a biomedical terminology expert.

I have two lists of disease/indication names. List A comes from clinical
trials and regulatory sources. List B comes from the OpenTargets platform.

Your task: for EACH indication in List A, find the BEST matching disease
name in List B. Diseases may use different names for the same condition
(e.g. "T2DM" matches "type II diabetes mellitus", "CKD" matches
"chronic kidney disease", "NASH" matches "non-alcoholic steatohepatitis",
"CV Risk Reduction" matches "cardiovascular disease").

Match by medical equivalence, not just string similarity. Only match if
the conditions are clinically the same or one is a clear subset of the other.
If there is no confident match in List B, set matched_ot_name to "".

List A (pipeline indications):
{pipeline_json}

List B (OpenTargets diseases):
{ot_json}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "matches": [
    {{"indication": "<exact name from List A>", "matched_ot_name": "<exact name from List B or empty>"}}
  ]
}}
"""
    try:
        text = _gemini_generate(
            prompt,
            system_instruction=(
                "You are a biomedical terminology expert. "
                "Return ONLY valid JSON."
            ),
            use_search=False,
        )
        if not text:
            return {}
        data = _extract_json(text)
        matches = data.get("matches", [])

        result = {}
        ot_set_lower = {n.lower(): n for n in unique_ot}
        for m in matches:
            ind = (m.get("indication") or "").strip()
            matched = (m.get("matched_ot_name") or "").strip()
            if ind and matched and matched.lower() in ot_set_lower:
                # Use the exact OT name (preserving original casing from OT)
                result[ind.lower()] = ot_set_lower[matched.lower()]
        logger.info("Matched %d/%d pipeline indications to OpenTargets diseases",
                     len(result), len(unique_pipeline))
        return result
    except Exception as e:
        logger.warning("Indication-to-OpenTargets matching failed: %s", e)
        return {}


def _apply_opentargets_scores(
    rows: list[dict],
    opentargets_rows: list[dict],
    moa_list: list[str],
) -> None:
    """Stamp opentargets_score and moa onto existing indication rows by
    matching their indication names to OpenTargets disease names.

    Also renames matched indications to use the OpenTargets disease name
    so names are consistent across all sheets.
    """
    if not opentargets_rows:
        return

    # Build lookup: ot_disease_name(lower) → best score
    ot_score_lookup: dict[str, float] = {}
    for otr in opentargets_rows:
        name = (otr.get("indication") or "").lower()
        score = otr.get("opentargets_score", 0.0)
        if name and isinstance(score, (int, float)):
            if name not in ot_score_lookup or score > ot_score_lookup[name]:
                ot_score_lookup[name] = score

    ot_disease_names = list({otr["indication"] for otr in opentargets_rows if otr.get("indication")})

    # Collect unique pipeline indications
    pipeline_indications = list({
        r["indication"] for r in rows
        if r.get("indication") and r["indication"] != "No indication found"
    })

    # First pass: direct lowercase match
    direct_matched = set()
    for row in rows:
        ind_lower = (row.get("indication") or "").lower()
        if ind_lower in ot_score_lookup:
            row["opentargets_score"] = ot_score_lookup[ind_lower]
            if moa_list and not row.get("moa"):
                row["moa"] = "; ".join(moa_list)
            direct_matched.add(ind_lower)

    # Remaining unmatched indications → use Gemini for fuzzy matching
    unmatched = [ind for ind in pipeline_indications if ind.lower() not in direct_matched]
    if unmatched and ot_disease_names:
        gemini_map = _match_indications_to_opentargets(unmatched, ot_disease_names)

        for row in rows:
            ind = (row.get("indication") or "")
            if ind.lower() in direct_matched:
                continue  # already matched directly
            ot_name = gemini_map.get(ind.lower())
            if ot_name:
                score = ot_score_lookup.get(ot_name.lower(), "")
                row["opentargets_score"] = score
                # Rename indication to the OpenTargets disease name for consistency
                row["indication"] = ot_name
                if moa_list and not row.get("moa"):
                    row["moa"] = "; ".join(moa_list)

    # Set moa on all rows if available
    if moa_list:
        moa_str = "; ".join(moa_list)
        for row in rows:
            if not row.get("moa"):
                row["moa"] = moa_str


def _run_opentargets_pipeline(molecule: str, company: str) -> tuple[list[dict], list[str]]:
    """Full OpenTargets pipeline: fetch MOA → resolve Ensembl IDs → query diseases.

    Returns (opentargets_rows, moa_list).
    """
    moas = _fetch_moa_from_bq(molecule)
    if not moas:
        logger.info("No MOA found for '%s' — skipping OpenTargets lookup", molecule)
        return [], []

    all_ot_rows = []
    for moa in moas:
        ensembl_id = _find_opentargets_target_id(moa)
        if not ensembl_id:
            continue
        diseases = _query_opentargets_diseases(ensembl_id)
        if diseases:
            ot_rows = _build_opentargets_rows(molecule, company, moa, ensembl_id, diseases)
            all_ot_rows.extend(ot_rows)
            logger.info("OpenTargets: %d indication(s) from MOA '%s'", len(ot_rows), moa)

    logger.info("OpenTargets total: %d indication(s) from %d MOA(s)", len(all_ot_rows), len(moas))
    return all_ot_rows, moas


# ══════════════════════════════════════════════════════════════════════════
#  MODULE B ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════
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

    # ── Search for approved secondary indications ─────────────────────
    approved_secondary_rows = _search_approved_secondary_indications(molecule, company)
    all_rows.extend(approved_secondary_rows)

    # ── Batch-standardise all raw indications in one Gemini + Search call ─
    raw_indications = [row["indication"] for row in all_rows if row["indication"]]
    standardization_map = standardize_indications(raw_indications)
    for row in all_rows:
        raw = row["indication"]
        std = standardization_map.get(raw, raw)
        if not std or std.lower() in ("error", "n/a", "none", "no indication found"):
            std = ""
        row["indication"] = std

    # Remove rows whose indication could not be resolved
    all_rows = [r for r in all_rows if r["indication"]]

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
        # Only overwrite rationale if it was empty
        if not row.get("rationale"):
            row["rationale"] = cls.get("rationale", "")

    # Override: Primary + non-Metabolic therapy area → Secondary
    for row in unique:
        if ((row.get("indication_type") or "").lower() == "primary"
                and (row.get("therapy_area") or "").lower() != "metabolic"):
            row["indication_type"] = "Secondary"

    # TA-I (Therapy Area - Indication) — only for Secondary indications
    for row in unique:
        if (row.get("indication_type") or "").lower() == "secondary":
            ta = (row.get("therapy_area") or "").strip()
            ind = (row.get("indication") or "").strip()
            row["TA-I"] = f"{ta} - {ind}" if ta and ind else ""
        else:
            row["TA-I"] = ""

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


def run_indication_research(
    molecule: str, company: str | None = None
) -> tuple[list[dict], list[dict], list[str]]:
    """Module B: research what the innovator company says about the drug.

    Returns
    -------
    tuple of (indication_rows, opentargets_rows, moa_list)
        indication_rows: standard Drug Indication Research rows.
        opentargets_rows: rows from OpenTargets disease associations.
        moa_list: list of MOA strings found for this drug.
    """
    logger.info("── Module B: Drug Indication Research ──")
    if not company:
        logger.info("Identifying innovator company")
        company = _identify_company(molecule) or ""

    indication_rows = _innovator_research(molecule, company)

    # ── OpenTargets pipeline (MOA → target → diseases) ─────────────────
    opentargets_rows, moa_list = _run_opentargets_pipeline(molecule, company)

    # ── Map OT scores back onto indication rows ───────────────────────
    if opentargets_rows:
        logger.info("Mapping OpenTargets association scores onto indication rows")
        _apply_opentargets_scores(indication_rows, opentargets_rows, moa_list)

    return indication_rows, opentargets_rows, moa_list
