"""Writes the single-drug label-expansion workbook.

Takes the row lists produced by `research_modules.py` (Clinical Efficacy
rows, Drug Indication Research rows) and writes them to one .xlsx with
four sheets: Clinical Efficacy, Drug Indication Research, All Combined,
and Summary. This module knows nothing about BigQuery, GCS, or Gemini —
it only formats rows that are already plain dicts.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import medical_potential.config as config

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  COMMON OUTPUT FORMAT
# ══════════════════════════════════════════════════════════════════════════
HEADERS = [
    "molecule_name", "company_name", "indication", "rationale",
    "indication_type", "therapy_area", "TA-I", "trial_title", "trial_id",
    "phase", "source_url", "data_source", "Ep", "Et",
]
COL_WIDTHS = [18, 22, 28, 40, 16, 18, 36, 50, 18, 10, 40, 16, 8, 8]


def _thin_border() -> Border:
    return Border(bottom=Side(style="thin", color="DDDDDD"), right=Side(style="thin", color="DDDDDD"))


def _write_standard_sheet(ws, rows: list[dict], molecule: str, sheet_title: str) -> None:
    thin = _thin_border()
    hfill = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")
    ncols = len(HEADERS)

    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value = f"{sheet_title}  —  {molecule.title()}"
    c.font = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    ws.merge_cells(f"A2:{get_column_letter(ncols)}2")
    c = ws["A2"]
    c.value = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   Rows: {len(rows)}"
    c.font = Font(name="Arial", italic=True, size=9, color="888888")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 16

    for col, h in enumerate(HEADERS, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin
    ws.row_dimensions[3].height = 22

    for idx, row in enumerate(rows, start=4):
        fill = alt_fill if idx % 2 == 0 else None
        for col, key in enumerate(HEADERS, 1):
            c = ws.cell(row=idx, column=col, value=row.get(key, ""))
            c.font = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            c.border = thin
            if fill:
                c.fill = fill
        ws.row_dimensions[idx].height = 18

    ws.auto_filter.ref = f"A3:{get_column_letter(ncols)}{3 + len(rows)}"
    ws.freeze_panes = "A4"
    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _write_summary_sheet(wb, rows: list[dict], title_label: str = "Summary") -> None:
    """Add a 'Summary' sheet: Drug Name | No. of Indications | No. of Therapy Areas."""
    ws = wb.create_sheet("Summary")
    thin = _thin_border()
    hfill = PatternFill("solid", start_color="1A1A2E")
    alt = PatternFill("solid", start_color="F4F7FB")

    # Keyed case-insensitively: BigQuery rows and innovator-research rows
    # can carry different casing for the same drug (e.g. "Testdrug" vs
    # "TestDrug"), and both should still roll up into one summary row.
    drug_data = defaultdict(lambda: {"display_name": "", "indications": set(), "therapy_areas": set()})
    for r in rows:
        name = (r.get("molecule_name") or "").strip()
        ind = r.get("indication", "")
        ta = r.get("therapy_area", "")
        if not name or ind == "No indication found":
            continue
        entry = drug_data[name.lower()]
        if not entry["display_name"]:
            entry["display_name"] = name
        entry["indications"].add(ind)
        if ta and ta.lower() not in ("other", ""):
            entry["therapy_areas"].add(ta)

    summary_headers = ["Drug Name", "No. of Indications", "No. of Therapy Areas"]
    summary_widths = [28, 20, 22]

    ws.merge_cells("A1:C1")
    c = ws["A1"]
    c.value = f"Summary  —  {title_label}"
    c.font = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    for col, h in enumerate(summary_headers, 1):
        c = ws.cell(row=2, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = hfill
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = thin
    ws.row_dimensions[2].height = 22

    for idx, (_, info) in enumerate(sorted(drug_data.items()), start=3):
        fill = alt if idx % 2 == 1 else None
        vals = [info["display_name"], len(info["indications"]), len(info["therapy_areas"])]
        for col, val in enumerate(vals, 1):
            c = ws.cell(row=idx, column=col, value=val)
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(horizontal="center" if col > 1 else "left", vertical="center")
            c.border = thin
            if fill:
                c.fill = fill
        ws.row_dimensions[idx].height = 20

    for i, w in enumerate(summary_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _parse_max_phase(phase) -> int:
    """Extract the highest phase number from a phase string. Returns 0 for
    None/Preclinical, -1 for unrecognised strings."""
    phase_str = str(phase).strip().lower() if phase else ""
    if not phase_str or phase_str in ("preclinical", "pre-clinical", "none", "n/a", ""):
        return 0
    if any(kw in phase_str for kw in ("approved", "launched", "marketed")):
        return 99  # sentinel for already-approved
    nums = re.findall(r"\d+", phase_str)
    if nums:
        return max(int(n) for n in nums)
    return -1


def _phase_maturity_weight(max_phase: int, is_approved: bool) -> float:
    """Map a parsed phase to its maturity weight.

    Phase mapping:
        None / Preclinical (0)  → 0.05
        Phase 1            (1)  → 0.1
        Phase 2            (2)  → 0.3
        Phase 3            (3)  → 0.6  (or 1.0 if the indication is approved)
        Approved           (99) → 1.0
    """
    if is_approved or max_phase >= 99:
        return 1.0
    if max_phase <= 0:
        return 0.05
    if max_phase == 1:
        return 0.1
    if max_phase == 2:
        return 0.3
    if max_phase >= 3:
        return 0.6
    return 0.05  # fallback for unrecognised


def _write_score_calculation_sheet(
    wb,
    combined_rows: list[dict],
    molecule: str,
) -> None:
    """Add a 'Score Calculation' sheet.

    Logic:
      1. Filter to rows where TA-I is non-empty (i.e. Secondary indications).
      2. Deduplicate by TA-I — keep the row with the highest phase.
      3. For each row, compute Phase Maturity Weight from the phase.
         Phase 3 rows get an extra Gemini+Search check: if the secondary
         indication is already approved, the weight becomes 1.0 instead of 0.6.
      4. Effective Indications = sum of all Phase Maturity Weights for the drug
         (same value on every row).
    """
    from medical_potential.label_expansion.research_modules import check_phase3_approvals

    ws = wb.create_sheet("Score Calculation")
    thin = _thin_border()
    hfill = PatternFill("solid", start_color="1A1A2E")
    alt_fill = PatternFill("solid", start_color="F4F7FB")

    # ── Step 1: rows with non-empty TA-I ────────────────────────────────
    tai_rows = [
        r for r in combined_rows
        if (r.get("TA-I") or "").strip()
    ]
    if not tai_rows:
        # Write an empty sheet with a note
        ws["A1"] = "No Secondary indications with TA-I found."
        ws["A1"].font = Font(name="Arial", italic=True, size=10, color="888888")
        return

    # ── Step 2: deduplicate by TA-I — keep highest phase ────────────────
    best_by_tai: dict[str, dict] = {}
    for row in tai_rows:
        tai = (row.get("TA-I") or "").strip()
        max_ph = _parse_max_phase(row.get("phase"))
        existing = best_by_tai.get(tai)
        if existing is None or max_ph > _parse_max_phase(existing.get("phase")):
            best_by_tai[tai] = dict(row)  # shallow copy

    unique_rows = list(best_by_tai.values())

    # ── Step 3: check approval status for Phase 3 indications ───────────
    phase3_indications = [
        r["indication"]
        for r in unique_rows
        if _parse_max_phase(r.get("phase")) == 3 and r.get("indication")
    ]
    approval_map: dict[str, bool] = {}
    if phase3_indications:
        logger.info("Checking approval status for %d Phase 3 secondary indication(s)", len(phase3_indications))
        approval_map = check_phase3_approvals(molecule, list(set(phase3_indications)))

    # ── Compute weights ─────────────────────────────────────────────────
    for row in unique_rows:
        max_ph = _parse_max_phase(row.get("phase"))
        is_approved = approval_map.get(row.get("indication", ""), False)
        row["phase_maturity_weight"] = _phase_maturity_weight(max_ph, is_approved)

    # ── Step 4: effective indications = sum of all weights ──────────────
    effective_indications = round(sum(r["phase_maturity_weight"] for r in unique_rows), 4)
    for row in unique_rows:
        row["effective_indications"] = effective_indications

    # ── Write sheet ─────────────────────────────────────────────────────
    score_headers = [
        "molecule_name", "company_name", "indication", "therapy_area",
        "TA-I", "phase", "data_source", "Phase Maturity Weight",
        "Effective Indications",
    ]
    score_widths = [18, 22, 28, 18, 36, 12, 16, 22, 22]
    ncols = len(score_headers)

    # Title row
    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value = f"Score Calculation  —  {molecule.title()}"
    c.font = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    # Subtitle
    ws.merge_cells(f"A2:{get_column_letter(ncols)}2")
    c = ws["A2"]
    c.value = (
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}   |   "
        f"Unique TA-I rows: {len(unique_rows)}   |   "
        f"Effective Indications: {effective_indications}"
    )
    c.font = Font(name="Arial", italic=True, size=9, color="888888")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 16

    # Header row
    for col, h in enumerate(score_headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin
    ws.row_dimensions[3].height = 22

    # Data rows
    for idx, row in enumerate(unique_rows, start=4):
        fill = alt_fill if idx % 2 == 0 else None
        for col, key in enumerate(score_headers, 1):
            if key == "Phase Maturity Weight":
                val = row.get("phase_maturity_weight", "")
            elif key == "Effective Indications":
                val = row.get("effective_indications", "")
            else:
                val = row.get(key, "")
            c = ws.cell(row=idx, column=col, value=val)
            c.font = Font(name="Arial", size=9)
            c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            c.border = thin
            if fill:
                c.fill = fill
        ws.row_dimensions[idx].height = 18

    ws.auto_filter.ref = f"A3:{get_column_letter(ncols)}{3 + len(unique_rows)}"
    ws.freeze_panes = "A4"
    for i, w in enumerate(score_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    logger.info(
        "Score Calculation sheet: %d unique TA-I row(s), Effective Indications = %s",
        len(unique_rows), effective_indications,
    )


def write_excel(molecule: str, clinical_rows: list[dict], indication_rows: list[dict]) -> Path:
    """Write the single-drug workbook: Clinical Efficacy, Drug Indication
    Research, All Combined, and Summary sheets.

    Writes directly into config.OUTPUT_DIR (created once, reused across
    runs) as `<drug>_label_expansion.xlsx` — no per-run output folder.
    """
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = molecule.lower().replace(" ", "_")
    out_file = output_dir / f"{slug}_label_expansion.xlsx"

    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = "Clinical Efficacy"
    _write_standard_sheet(ws1, clinical_rows, molecule, "Clinical Efficacy")

    ws2 = wb.create_sheet("Drug Indication Research")
    _write_standard_sheet(ws2, indication_rows, molecule, "Drug Indication Research")

    combined = clinical_rows + indication_rows

    # Et in "All Combined" = highest Et across both modules
    et_values = [r.get("Et") for r in combined if r.get("Et")]
    max_et = ""
    if et_values:
        try:
            max_et = str(max(int(v) for v in et_values if str(v).isdigit()))
        except ValueError:
            max_et = et_values[0]
    for row in combined:
        row["Et"] = max_et

    ws3 = wb.create_sheet("All Combined")
    _write_standard_sheet(ws3, combined, molecule, "All Sources Combined")

    _write_summary_sheet(wb, combined, molecule.title())

    _write_score_calculation_sheet(wb, combined, molecule)

    wb.save(str(out_file))
    logger.info("Excel workbook saved → %s", out_file)
    return out_file
