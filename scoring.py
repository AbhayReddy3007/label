"""
scoring.py — Maturity Weight Scoring
=====================================

Reads the label-expansion Excel workbook and adds a new sheet called
"Maturity Weight Scoring". The sheet copies all data from the "All Combined"
sheet and appends a "Maturity_Weight" column derived from each row's phase:

    Phase                      Maturity_Weight
    ─────────────────────────  ───────────────
    Not available / Preclinical       0.05
    Phase 1                           0.1
    Phase 2                           0.3
    Phase 3                           0.6
    Approved                          1.0

Usage (standalone)
-------------------
    python -m medical_potential.label_expansion.scoring path/to/workbook.xlsx

Programmatic usage
-------------------
    from medical_potential.label_expansion.scoring import add_maturity_weight_sheet
    add_maturity_weight_sheet("output/semaglutide_label_expansion.xlsx")
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)


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

    # Check for "approved" / "marketed" / "launched" keywords first
    if any(kw in raw for kw in ("approved", "marketed", "launched")):
        return 1.0

    # Extract numeric phase values
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

    # Fallback for "preclinical" or anything unrecognised
    if "preclinical" in raw or "pre-clinical" in raw:
        return 0.05

    return 0.05


# ══════════════════════════════════════════════════════════════════════════
#  SHEET WRITER
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

    # ── Read headers (row 3 in the standard layout) and data ──────────
    # The standard sheet layout is:
    #   Row 1: title (merged)
    #   Row 2: metadata (merged)
    #   Row 3: headers
    #   Row 4+: data
    header_row = 3
    data_start = 4

    headers = []
    for col in range(1, src.max_column + 1):
        val = src.cell(row=header_row, column=col).value
        if val is not None:
            headers.append(str(val))
        else:
            headers.append(f"col_{col}")

    # Read data rows
    data_rows = []
    for row_idx in range(data_start, src.max_row + 1):
        row_data = {}
        for col_idx, header in enumerate(headers, 1):
            row_data[header] = src.cell(row=row_idx, column=col_idx).value
        # Skip completely empty rows
        if any(v is not None and str(v).strip() for v in row_data.values()):
            data_rows.append(row_data)

    if not data_rows:
        logger.warning("No data rows found in '%s' — skipping maturity weight sheet", source_sheet_name)
        return excel_path

    # ── Find the phase column ────────────────────────────────────────
    phase_col = None
    for h in headers:
        if h.lower() == "phase":
            phase_col = h
            break
    if not phase_col:
        logger.warning("No 'phase' column found in '%s' — cannot compute maturity weights", source_sheet_name)
        return excel_path

    # ── Compute maturity weights ─────────────────────────────────────
    for row in data_rows:
        phase_val = row.get(phase_col, "")
        row["Maturity_Weight"] = _phase_to_maturity_weight(phase_val)

    # ── Build the new sheet ──────────────────────────────────────────
    new_headers = headers + ["Maturity_Weight"]

    # Remove existing sheet if present (for re-runs)
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

    # Title row
    ws.merge_cells(f"A1:{get_column_letter(ncols)}1")
    c = ws["A1"]
    c.value = "Maturity Weight Scoring"
    c.font = Font(name="Arial", bold=True, size=14, color="1A1A2E")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 32

    # Metadata row
    ws.merge_cells(f"A2:{get_column_letter(ncols)}2")
    c = ws["A2"]
    c.value = f"Rows: {len(data_rows)}  |  Weight mapping: Preclinical=0.05, Ph1=0.1, Ph2=0.3, Ph3=0.6, Approved=1.0"
    c.font = Font(name="Arial", italic=True, size=9, color="888888")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[2].height = 16

    # Header row
    for col, h in enumerate(new_headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = hfill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin
    ws.row_dimensions[3].height = 22

    # Data rows
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

            # Format the Maturity_Weight column as a number
            if key == "Maturity_Weight" and isinstance(val, (int, float)):
                c.number_format = '0.00'
                c.alignment = Alignment(horizontal="center", vertical="top")

            # Format the opentargets_score column as a number
            if key == "opentargets_score" and isinstance(val, (int, float)):
                c.number_format = '0.0000'

        ws.row_dimensions[idx].height = 18

    # Auto-filter and freeze
    ws.auto_filter.ref = f"A3:{get_column_letter(ncols)}{3 + len(data_rows)}"
    ws.freeze_panes = "A4"

    # Set column widths — reuse standard widths where possible
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
    add_maturity_weight_sheet(path)


if __name__ == "__main__":
    main()
