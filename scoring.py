"""
scoring.py — Targeted OpenTargets Scoring + Maturity Weight Scoring
=====================================================================

Standalone, re-runnable step. Given an already-generated label-expansion
workbook (produced once by the full label_expansion pipeline — Module A
and Module B need NOT be re-run), this script:

  1. Reads "Clinical Efficacy" and "Drug Indication Research" directly
     from the workbook.
  2. For every row whose indication_type is "Secondary", resolves the
     row's MOA to an Ensembl target ID (Gemini + Search, cached per MOA)
     and looks up ONLY that one indication against the OpenTargets
     Platform:
       - a single disease `search` call to align the indication name to
         OpenTargets' canonical disease name and get its EFO ID, then
       - a single target-disease association query filtered to that
         exact EFO ID (`Bs: [efoId]`) to fetch just that one score.
     This never bulk-fetches "all associated diseases" for a target —
     only the specific indications already present in our own data are
     looked up. Primary rows are left completely untouched: no
     OpenTargets call is made for them and any pre-existing
     opentargets_score on a non-Secondary row is cleared.
  3. Rebuilds "All Combined" from the (now updated) Clinical Efficacy +
     Drug Indication Research rows only — the old bulk "OpenTargets
     Indications" dump is no longer folded in, so this sheet only ever
     contains indications actually observed in trials / innovator
     research, not every disease OpenTargets happens to associate with
     the target.
  4. Rebuilds the "Maturity Weight Scoring" sheet from that new "All
     Combined" data, appending the Maturity_Weight column as before.

Every step is idempotent — sheets are updated/replaced in place, not
appended to — and OpenTargets/Gemini lookups are cached in-memory per
run so the same MOA or indication is never looked up twice in one pass.

Usage (standalone)
-------------------
    python -m medical_potential.label_expansion.scoring path/to/workbook.xlsx

Programmatic usage
-------------------
    from medical_potential.label_expansion.scoring import run_scoring
    run_scoring("output/semaglutide_label_expansion.xlsx")
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

import openpyxl
import requests
from google import genai
from google.genai import types as genai_types
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import medical_potential.config as config
from medical_potential.label_expansion.excel_writer import (
    HEADERS,
    COL_WIDTHS,
    _write_standard_sheet,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  GEMINI CLIENT + PLUMBING
# ══════════════════════════════════════════════════════════════════════════
# Duplicated (trimmed) from research_modules.py rather than imported, so
# this file has no dependency on the rest of the pipeline and can be run
# completely on its own against an existing workbook.
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
    """Safely extract text from a Gemini response (see research_modules.py
    for the full explanation of why this walks the candidates/parts tree)."""
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
    """Call Gemini with optional Google Search grounding, with retries."""
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
                break
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
    """Extract and parse the first JSON object or array from *text*."""
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
#  TARGETED OPENTARGETS LOOKUPS (no bulk "all diseases" fetch)
# ══════════════════════════════════════════════════════════════════════════
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _resolve_ensembl_id_for_moa(moa: str) -> str | None:
    """Use Gemini + Search to find the Ensembl gene ID (ENSG...) for a
    given Mechanism of Action target name."""
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
        text = _gemini_generate(prompt, system_instruction="Return ONLY valid JSON.", use_search=True)
        if not text:
            return None
        data = _extract_json(text)
        ensembl_id = (data.get("ensembl_id") or "").strip()
        target_name = (data.get("target_name") or "").strip()
        if ensembl_id and ensembl_id.startswith("ENSG"):
            logger.info("MOA '%s' → target '%s' → Ensembl ID: %s", moa, target_name, ensembl_id)
            return ensembl_id
        logger.warning("Could not resolve Ensembl ID for MOA '%s' (target: '%s')", moa, target_name)
        return None
    except Exception as e:
        logger.warning("Failed to resolve Ensembl ID for MOA '%s': %s", moa, e)
        return None


def _search_opentargets_disease(indication: str) -> tuple[str, str] | None:
    """Look up ONE specific indication by name against OpenTargets' search
    endpoint. Returns (efo_id, canonical_name) for the best disease hit,
    or None if nothing matched. This is a single targeted lookup — it does
    not fetch or page through OpenTargets' full disease list."""
    query_string = """
    query diseaseSearch($q: String!) {
      search(queryString: $q, entityNames: ["disease"], page: {index: 0, size: 5}) {
        hits { id name entity score }
      }
    }
    """
    try:
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query_string, "variables": {"q": indication}},
            timeout=20,
        )
        resp.raise_for_status()
        hits = resp.json().get("data", {}).get("search", {}).get("hits", []) or []
        for h in hits:
            if h.get("entity") == "disease" and h.get("id") and h.get("name"):
                return h["id"], h["name"]
        logger.info("No OpenTargets disease match for indication '%s'", indication)
        return None
    except requests.RequestException as e:
        logger.warning("OpenTargets disease search failed for '%s': %s", indication, e)
        return None


def _get_target_disease_score(ensembl_id: str, efo_id: str) -> float | None:
    """Fetch the association score for exactly one (target, disease) pair.

    Uses the `Bs` filter on `target.associatedDiseases` to restrict the
    result to the single disease ID we care about, instead of paging
    through every disease associated with the target."""
    query_string = """
    query targetDiseaseScore($ensemblId: String!, $diseaseIds: [String!]) {
      target(ensemblId: $ensemblId) {
        associatedDiseases(Bs: $diseaseIds, page: {index: 0, size: 5}) {
          rows { score disease { id name } }
        }
      }
    }
    """
    variables = {"ensemblId": ensembl_id, "diseaseIds": [efo_id]}
    try:
        resp = requests.post(
            OPENTARGETS_GRAPHQL_URL,
            json={"query": query_string, "variables": variables},
            timeout=20,
        )
        resp.raise_for_status()
        target = resp.json().get("data", {}).get("target")
        if not target:
            return None
        for row in (target.get("associatedDiseases", {}).get("rows", []) or []):
            if row.get("disease", {}).get("id") == efo_id:
                return round(row.get("score", 0.0), 4)
        return None
    except requests.RequestException as e:
        logger.warning("OpenTargets target-disease lookup failed (%s, %s): %s", ensembl_id, efo_id, e)
        return None


# ══════════════════════════════════════════════════════════════════════════
#  SHEET READ / UPDATE — Clinical Efficacy & Drug Indication Research
# ══════════════════════════════════════════════════════════════════════════
def _read_sheet_rows(ws) -> tuple[list[str], list[dict], dict[str, int]]:
    """Read a standard-layout sheet (title row 1, metadata row 2, headers
    row 3, data from row 4). Returns (headers, row_dicts, header_to_col)."""
    header_row, data_start = 3, 4
    headers = []
    for col in range(1, ws.max_column + 1):
        val = ws.cell(row=header_row, column=col).value
        headers.append(str(val) if val is not None else f"col_{col}")
    header_to_col = {h: i + 1 for i, h in enumerate(headers)}

    rows = []
    row_numbers = []
    for r in range(data_start, ws.max_row + 1):
        row_data = {h: ws.cell(row=r, column=header_to_col[h]).value for h in headers}
        if any(v is not None and str(v).strip() for v in row_data.values()):
            rows.append(row_data)
            row_numbers.append(r)

    for row, r in zip(rows, row_numbers):
        row["_row_number"] = r
    return headers, rows, header_to_col


def _update_secondary_rows_with_opentargets(
    ws,
    header_to_col: dict[str, int],
    rows: list[dict],
    ensembl_cache: dict[str, str | None],
    disease_cache: dict[str, tuple[str, str] | None],
) -> None:
    """Mutate *ws* in place: for Secondary rows, resolve MOA → Ensembl ID,
    look up that one indication on OpenTargets, align its name, and stamp
    opentargets_score. For every other row, clear any stale
    opentargets_score. Cell writes are targeted (only the indication and
    opentargets_score columns) so all other formatting is untouched."""
    ind_col = header_to_col.get("indication")
    itype_col = header_to_col.get("indication_type")
    score_col = header_to_col.get("opentargets_score")
    moa_col = header_to_col.get("moa")

    if not (ind_col and itype_col and score_col):
        logger.warning("Sheet is missing indication/indication_type/opentargets_score columns — skipping")
        return

    for row in rows:
        r = row["_row_number"]
        itype = str(row.get("indication_type", "") or "").strip().lower()

        if itype != "secondary":
            # Not a Secondary row — make sure no stale score lingers here.
            if row.get("opentargets_score") not in ("", None):
                ws.cell(row=r, column=score_col, value="")
                row["opentargets_score"] = ""
            continue

        indication = str(row.get("indication", "") or "").strip()
        if not indication or indication.lower() == "no indication found":
            continue

        moa_field = str(row.get("moa", "") or "").strip()
        if not moa_field:
            logger.info("Secondary row '%s' has no MOA — cannot resolve an OpenTargets target", indication)
            continue

        ensembl_id = None
        for moa in [m.strip() for m in moa_field.split(";") if m.strip()]:
            if moa not in ensembl_cache:
                ensembl_cache[moa] = _resolve_ensembl_id_for_moa(moa)
            ensembl_id = ensembl_cache[moa]
            if ensembl_id:
                break
        if not ensembl_id:
            continue

        cache_key = indication.lower()
        if cache_key not in disease_cache:
            disease_cache[cache_key] = _search_opentargets_disease(indication)
        hit = disease_cache[cache_key]
        if not hit:
            continue
        efo_id, ot_name = hit

        score = _get_target_disease_score(ensembl_id, efo_id)
        if score is None:
            continue

        # Align the indication name to OpenTargets' canonical name.
        ws.cell(row=r, column=ind_col, value=ot_name)
        row["indication"] = ot_name

        score_cell = ws.cell(row=r, column=score_col, value=score)
        score_cell.number_format = '0.0000'
        row["opentargets_score"] = score

        if moa_col and not row.get("moa"):
            ws.cell(row=r, column=moa_col, value=moa_field)


def _process_sheet(wb, sheet_name: str, ensembl_cache: dict, disease_cache: dict) -> list[dict]:
    """Read *sheet_name*, apply targeted OpenTargets scoring to its
    Secondary rows in place, and return the resulting row dicts (ready
    for use elsewhere, e.g. rebuilding "All Combined")."""
    if sheet_name not in wb.sheetnames:
        logger.warning("Sheet '%s' not found in workbook — skipping", sheet_name)
        return []
    ws = wb[sheet_name]
    _headers, rows, header_to_col = _read_sheet_rows(ws)
    _update_secondary_rows_with_opentargets(ws, header_to_col, rows, ensembl_cache, disease_cache)
    for row in rows:
        row.pop("_row_number", None)
    return rows


# ══════════════════════════════════════════════════════════════════════════
#  PHASE → MATURITY WEIGHT MAPPING
# ══════════════════════════════════════════════════════════════════════════
def _phase_to_maturity_weight(phase) -> float:
    """Map a phase string to its maturity weight.

    Rules:
      - "Not available", "Preclinical", empty, or unrecognised → 0.05
      - "Phase 1" (or contains "1")                           → 0.1
      - "Phase 2" (or contains "2")                           → 0.3
      - "Phase 3" (or contains "3")                           → 0.6
      - "Approved", "Phase 4", "Marketed", "Launched"         → 1.0
    """
    raw = str(phase).strip().lower() if phase else ""

    if not raw or raw in ("not available", "preclinical", "n/a", "none", ""):
        return 0.05

    if any(kw in raw for kw in ("approved", "marketed", "launched")):
        return 1.0

    nums = re.findall(r"\d+", raw)
    if nums:
        max_phase = max(int(n) for n in nums)
        if max_phase >= 4:
            return 1.0
        elif max_phase == 3:
            return 0.6
        elif max_phase == 2:
            return 0.3
        elif max_phase == 1:
            return 0.1

    if "preclinical" in raw or "pre-clinical" in raw:
        return 0.05

    return 0.05


# ══════════════════════════════════════════════════════════════════════════
#  ALL COMBINED — rebuilt from Clinical Efficacy + Drug Indication
#  Research only (the bulk "OpenTargets Indications" dump is NOT folded
#  back in, so this sheet only ever reflects indications actually
#  observed in trials / innovator research).
# ══════════════════════════════════════════════════════════════════════════
def _rebuild_all_combined(wb, clinical_rows: list[dict], indication_rows: list[dict], molecule: str) -> None:
    combined = clinical_rows + indication_rows

    et_values = [r.get("Et") for r in combined if r.get("Et")]
    max_et = ""
    if et_values:
        try:
            max_et = str(max(int(v) for v in et_values if str(v).isdigit()))
        except ValueError:
            max_et = et_values[0]
    for row in combined:
        row["Et"] = max_et

    if "All Combined" in wb.sheetnames:
        idx = wb.sheetnames.index("All Combined")
        del wb["All Combined"]
    else:
        idx = len(wb.sheetnames)

    ws = wb.create_sheet("All Combined")
    _write_standard_sheet(ws, combined, molecule, "All Sources Combined", headers=HEADERS, col_widths=COL_WIDTHS)
    wb.move_sheet("All Combined", offset=idx - (len(wb.sheetnames) - 1))


# ══════════════════════════════════════════════════════════════════════════
#  MATURITY WEIGHT SHEET WRITER
# ══════════════════════════════════════════════════════════════════════════
def add_maturity_weight_sheet(excel_path: str | Path) -> Path:
    """Open the label-expansion workbook, read the "All Combined" sheet,
    and add a new "Maturity Weight Scoring" sheet with a Maturity_Weight
    column appended.

    Parameters
    ----------
    excel_path : str or Path
        Path to the existing .xlsx workbook.

    Returns
    -------
    Path to the (updated) workbook.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    wb = openpyxl.load_workbook(str(excel_path))

    source_sheet_name = "All Combined"
    if source_sheet_name not in wb.sheetnames:
        logger.warning("'%s' sheet not found in workbook — cannot add maturity weights", source_sheet_name)
        return excel_path

    src = wb[source_sheet_name]

    header_row = 3
    data_start = 4

    headers = []
    for col in range(1, src.max_column + 1):
        val = src.cell(row=header_row, column=col).value
        headers.append(str(val) if val is not None else f"col_{col}")

    data_rows = []
    for row_idx in range(data_start, src.max_row + 1):
        row_data = {}
        for col_idx, header in enumerate(headers, 1):
            row_data[header] = src.cell(row=row_idx, column=col_idx).value
        if any(v is not None and str(v).strip() for v in row_data.values()):
            data_rows.append(row_data)

    if not data_rows:
        logger.warning("No data rows found in '%s' — skipping maturity weight sheet", source_sheet_name)
        return excel_path

    phase_col = None
    for h in headers:
        if h.lower() == "phase":
            phase_col = h
            break
    if not phase_col:
        logger.warning("No 'phase' column found in '%s' — cannot compute maturity weights", source_sheet_name)
        return excel_path

    for row in data_rows:
        phase_val = row.get(phase_col, "")
        row["Maturity_Weight"] = _phase_to_maturity_weight(phase_val)

    # ── Defensive cleanup: opentargets_score should only ever be present
    # on Secondary rows. It should already be true by construction (see
    # _update_secondary_rows_with_opentargets), but this guards against a
    # workbook that was hand-edited or produced by an older run. ─────────
    indication_type_col = None
    for h in headers:
        if h.lower() == "indication_type":
            indication_type_col = h
            break

    if indication_type_col and "opentargets_score" in headers:
        cleared = 0
        for row in data_rows:
            itype = str(row.get(indication_type_col, "") or "").strip().lower()
            if itype != "secondary":
                row["opentargets_score"] = ""
                cleared += 1
        logger.info(
            "Cleared opentargets_score on %d non-Secondary row(s); kept it only for Secondary rows",
            cleared,
        )
    elif not indication_type_col:
        logger.warning(
            "No 'indication_type' column found in '%s' — opentargets_score left as-is for all rows",
            source_sheet_name,
        )

    new_headers = headers + ["Maturity_Weight"]

    target_sheet_name = "Maturity Weight Scoring"
    if target_sheet_name in wb.sheetnames:
        del wb[target_sheet_name]

    ws = wb.create_sheet(target_sheet_name)

    thin = Border(
        bottom=Side(style="thin", color="DDDDDD"),
        right=Side(style="thin", color="DDDDDD"),
    )
    hfill = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")
    ncols = len(new_headers)

    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value = "Maturity Weight Scoring"
    c.font = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    ws.merge_cells(f"A2:{get_column_letter(ncols)}2")
    c = ws["A2"]
    c.value = (
        f"Rows: {len(data_rows)}  |  Weight mapping: Preclinical=0.05, Ph1=0.1, Ph2=0.3, "
        f"Ph3=0.6, Approved=1.0  |  opentargets_score shown for Secondary indications only"
    )
    c.font = Font(name="Arial", italic=True, size=9, color="888888")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 16

    for col, h in enumerate(new_headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin
    ws.row_dimensions[3].height = 22

    for idx, row in enumerate(data_rows, start=4):
        fill = alt_fill if idx % 2 == 0 else None
        for col, key in enumerate(new_headers, 1):
            val = row.get(key, "")
            c = ws.cell(row=idx, column=col, value=val)
            c.font = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            c.border = thin
            if fill:
                c.fill = fill

            if key == "Maturity_Weight" and isinstance(val, (int, float)):
                c.number_format = '0.00'
                c.alignment = Alignment(horizontal="center", vertical="top")

            if key == "opentargets_score" and isinstance(val, (int, float)):
                c.number_format = '0.0000'

        ws.row_dimensions[idx].height = 18

    ws.auto_filter.ref = f"A3:{get_column_letter(ncols)}{3 + len(data_rows)}"
    ws.freeze_panes = "A4"

    standard_widths = {
        "molecule_name": 18, "company_name": 22, "indication": 28,
        "rationale": 40, "indication_type": 16, "therapy_area": 18,
        "TA-I": 36, "trial_title": 50, "trial_id": 18, "trial_size": 12,
        "phase": 10, "source_url": 40, "data_source": 16, "Ep": 8, "Et": 8,
        "moa": 28, "opentargets_score": 18, "Maturity_Weight": 16,
    }
    for i, h in enumerate(new_headers, 1):
        ws.column_dimensions[get_column_letter(i)].width = standard_widths.get(h, 16)

    wb.save(str(excel_path))
    logger.info("Maturity Weight Scoring sheet added → %s", excel_path)
    return excel_path


# ══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATION — the standalone, re-runnable entry point
# ══════════════════════════════════════════════════════════════════════════
def run_scoring(excel_path: str | Path) -> Path:
    """Full standalone scoring step. Run this on its own, any time, on an
    already-generated label-expansion workbook — it never touches
    BigQuery, GCS, or re-runs Module A / Module B.

    1. Apply targeted OpenTargets scoring to Secondary rows in "Clinical
       Efficacy" and "Drug Indication Research" (in place).
    2. Rebuild "All Combined" from just those two sheets.
    3. Rebuild "Maturity Weight Scoring" from the new "All Combined".
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Workbook not found: {excel_path}")

    wb = openpyxl.load_workbook(str(excel_path))

    ensembl_cache: dict[str, str | None] = {}
    disease_cache: dict[str, tuple[str, str] | None] = {}

    clinical_rows = _process_sheet(wb, "Clinical Efficacy", ensembl_cache, disease_cache)
    indication_rows = _process_sheet(wb, "Drug Indication Research", ensembl_cache, disease_cache)

    molecule = ""
    for row in clinical_rows + indication_rows:
        if row.get("molecule_name"):
            molecule = str(row["molecule_name"])
            break

    _rebuild_all_combined(wb, clinical_rows, indication_rows, molecule)
    wb.save(str(excel_path))
    logger.info(
        "Targeted OpenTargets scoring applied and 'All Combined' rebuilt (%d clinical + %d indication rows) → %s",
        len(clinical_rows), len(indication_rows), excel_path,
    )

    return add_maturity_weight_sheet(excel_path)


# ══════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    if len(sys.argv) < 2:
        print("Usage: python -m medical_potential.label_expansion.scoring <path_to_xlsx>")
        sys.exit(1)

    path = sys.argv[1]
    run_scoring(path)


if __name__ == "__main__":
    main()
