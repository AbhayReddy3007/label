"""Writes the single-drug label-expansion workbook.

Takes the row lists produced by `research_modules.py` (Clinical Efficacy
rows, Drug Indication Research rows) and writes them to one .xlsx with
four sheets: Clinical Efficacy, Drug Indication Research, All Combined,
and Summary. This module knows nothing about BigQuery, GCS, or Gemini —
it only formats rows that are already plain dicts.
"""

from __future__ import annotations

import logging
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
    "indication_type", "therapy_area", "trial_title", "trial_id",
    "phase", "source_url", "data_source", "Ep", "Et",
]
COL_WIDTHS = [18, 22, 28, 40, 16, 18, 50, 18, 10, 40, 16, 8, 8]


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

    wb.save(str(out_file))
    logger.info("Excel workbook saved → %s", out_file)
    return out_file
