"""indication_standardizer.py — Canonical indication name normalizer.

Exposes one public function:

    standardize_indications(raws: list[str]) -> dict[str, str]

Pass every raw indication string extracted in a run; get back a map of
raw → canonical name. All strings are sent to Gemini + Google Search in
one batched call so there is exactly one API round-trip per module run
(with a per-string fallback only for any that Gemini misses).

Canonical names follow the form used in FDA labels and ClinicalTrials.gov
(e.g. "Type 2 Diabetes Mellitus", "Heart Failure with Preserved Ejection
Fraction (HFpEF)"), including the standard abbreviation in parentheses
where one is widely used.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

from google import genai
from google.genai import types as genai_types

import medical_potential.config as config

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
#  IN-PROCESS CACHE  (raw.lower() → canonical, thread-safe)
#  Shared across both modules so the same raw string is never resolved twice
#  in a single run, even when Module A and Module B are called sequentially.
# ══════════════════════════════════════════════════════════════════════════
_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

# Sentinel values that callers treat as "skip this row"
_SENTINELS = {"error", "n/a", "none", "no indication found", ""}


# ══════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT  (mirrors research_modules.py plumbing exactly)
# ══════════════════════════════════════════════════════════════════════════
_gemini_client: genai.Client | None = None
_gemini_client_lock = threading.Lock()


def _get_gemini_client() -> genai.Client:
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


def _gemini_generate(prompt: str) -> str:
    """Call Gemini with Google Search grounding, falling back to no-search on empty/error."""
    client = _get_gemini_client()
    model = config.GEMINI_FLASH_PREVIEW_MODEL
    max_retries = config.GEMINI_MAX_RETRIES
    base_delay = config.GEMINI_RETRY_BASE_DELAY_SECONDS
    system = "You are a medical terminology expert. Return ONLY valid JSON. No markdown."

    configs = [
        genai_types.GenerateContentConfig(
            temperature=0,
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            system_instruction=system,
        ),
        genai_types.GenerateContentConfig(
            temperature=0,
            system_instruction=system,
        ),
    ]

    for i, cfg in enumerate(configs):
        last_err = None
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
                text = _safe_response_text(resp)
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                if text:
                    return text
                break  # empty — fall through to next config
            except Exception as e:
                last_err = e
                if _is_transient_error(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "[standardizer] Gemini error (%s) — retrying in %ss (%d/%d)",
                        e, delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                elif i == 0:
                    logger.info(
                        "[standardizer] Search-grounded call failed (%s) — retrying without grounding", e
                    )
                    break
                else:
                    raise
        else:
            if i == 0:
                logger.info(
                    "[standardizer] Retries exhausted with Search grounding — retrying without grounding"
                )
                continue
            if last_err:
                raise last_err

        if i == 0:
            logger.info(
                "[standardizer] Empty response with Search grounding — retrying without grounding"
            )

    return ""


# ══════════════════════════════════════════════════════════════════════════
#  BATCH STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════
def _build_batch_prompt(raws: list[str]) -> str:
    raws_json = json.dumps(raws, indent=2)
    return f"""
You are a medical terminology expert.

Below is a list of raw indication strings extracted from clinical trial
records and pharmaceutical company sources. They may be abbreviations,
informal names, inconsistent capitalisations, or partial descriptions of
the same underlying disease.

Raw indications:
{raws_json}

For EACH raw string:
1. Use Google Search to identify the exact disease or medical condition it refers to.
2. Return the single canonical, internationally recognised name — the form
   used in FDA prescribing information, EMA SmPC, or ClinicalTrials.gov
   (e.g. "Type 2 Diabetes Mellitus", "Heart Failure with Preserved Ejection
   Fraction (HFpEF)", "Non-Alcoholic Steatohepatitis (NASH)").
3. Include the standard abbreviation in parentheses when one is widely used.
4. If a raw string is genuinely not a medical indication (or cannot be
   resolved), set canonical_name to "no indication found".

Return ONLY valid JSON — no markdown, no explanation:
{{
  "standardized": [
    {{"raw": "<exact string from input list>", "canonical_name": "<canonical name>"}},
    ...
  ]
}}
"""


def _batch_standardize_via_gemini(raws: list[str]) -> dict[str, str]:
    """Send all raws to Gemini in one call. Returns raw.lower() → canonical map."""
    logger.info("[standardizer] Batch standardising %d indication(s) via Gemini + Search", len(raws))
    try:
        text = _gemini_generate(_build_batch_prompt(raws))
        if not text:
            raise ValueError("Empty response from Gemini")
        data = json.loads(text)
        result = {}
        for entry in data.get("standardized", []):
            raw = (entry.get("raw") or "").strip()
            canonical = (entry.get("canonical_name") or "").strip()
            if raw:
                result[raw.lower()] = canonical
        return result
    except Exception as e:
        logger.warning("[standardizer] Batch call failed (%s) — will fall back per-string", e)
        return {}


def _standardize_single_via_gemini(raw: str) -> str:
    """Fallback: resolve one raw string individually."""
    prompt = f"""
You are a medical terminology expert.

Raw indication text: "{raw}"

Search the web and return the single canonical, internationally recognised
name for this medical condition (as used in FDA labels or ClinicalTrials.gov),
including the standard abbreviation in parentheses where widely used.

If this is not a medical indication or cannot be resolved, return:
  {{"canonical_name": "no indication found"}}

Return ONLY valid JSON — no markdown, no explanation:
{{"canonical_name": "<canonical indication name>"}}
"""
    try:
        text = _gemini_generate(prompt)
        if not text:
            return ""
        data = json.loads(text)
        return (data.get("canonical_name") or "").strip()
    except Exception as e:
        logger.warning("[standardizer] Single fallback failed for '%s': %s", raw, e)
        return ""


# ══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════
def standardize_indications(raws: list[str]) -> dict[str, str]:
    """Return a map of raw indication string → canonical name.

    Pass all raw strings extracted in a module run at once. The function:
      1. Filters out blanks and known sentinels immediately.
      2. Checks the in-process cache for strings already resolved this run.
      3. Sends all remaining strings to Gemini + Google Search in ONE batch call.
      4. Falls back to individual Gemini calls for any strings Gemini missed.
      5. Falls back to the raw string itself if Gemini cannot resolve it,
         so rows are never silently dropped due to a standardisation failure.
      6. Updates the cache with all newly resolved strings.

    The returned dict covers every non-blank input string. Callers should
    skip rows where the value is in ("no indication found", "error", "n/a",
    "none", "").
    """
    result: dict[str, str] = {}
    to_resolve: list[str] = []

    for raw in raws:
        stripped = (raw or "").strip()
        if not stripped:
            result[raw] = ""
            continue
        key = stripped.lower()
        if key in _SENTINELS:
            result[raw] = stripped
            continue
        with _cache_lock:
            if key in _cache:
                result[raw] = _cache[key]
                continue
        to_resolve.append(stripped)

    if not to_resolve:
        return result

    # Process in batches of INDICATION_BATCH_SIZE
    batch_size = max(1, config.INDICATION_BATCH_SIZE)
    # Deduplicate to_resolve (preserving order) so we don't send the same string twice
    seen_resolve = set()
    unique_to_resolve = []
    for r in to_resolve:
        if r.lower() not in seen_resolve:
            seen_resolve.add(r.lower())
            unique_to_resolve.append(r)

    batches = [unique_to_resolve[i:i + batch_size] for i in range(0, len(unique_to_resolve), batch_size)]
    logger.info(
        "[standardizer] Standardising %d unique indication(s) in %d batch(es) of up to %d",
        len(unique_to_resolve), len(batches), batch_size,
    )

    full_batch_map: dict[str, str] = {}
    for i, batch in enumerate(batches, 1):
        logger.info("[standardizer] Standardisation batch %d/%d (%d indications)", i, len(batches), len(batch))
        batch_map = _batch_standardize_via_gemini(batch)

        # Individual fallback for any missed by the batch
        missed = [r for r in batch if r.lower() not in batch_map]
        if missed:
            logger.info("[standardizer] %d indication(s) missing from batch %d — resolving individually", len(missed), i)
        for raw in missed:
            canonical = _standardize_single_via_gemini(raw)
            if not canonical:
                canonical = raw  # preserve raw rather than silently dropping
                logger.debug("[standardizer] Could not resolve '%s' — keeping raw", raw)
            batch_map[raw.lower()] = canonical

        full_batch_map.update(batch_map)

    # Merge into result and update cache
    with _cache_lock:
        for raw in to_resolve:
            key = raw.lower()
            canonical = full_batch_map.get(key, raw)  # fallback to raw
            result[raw] = canonical
            _cache[key] = canonical
            logger.debug("[standardizer] '%s' → '%s'", raw, canonical)

    return result
