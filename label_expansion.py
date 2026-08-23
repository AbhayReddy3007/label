"""
label_expansion.py — Label Expansion Pipeline (entry point)
====================================================================

The main entry point for the label-expansion pipeline. This file contains
ONLY orchestration: read config, call the two research modules, write the
Excel output, and handle the CLI. All of the actual work lives elsewhere:

    medical_potential/config.py                         all tunables
    medical_potential/gcp_utils.py                      BigQuery + GCS helpers
    medical_potential/label_expansion/research_modules.py  Module A + Module B
    medical_potential/label_expansion/excel_writer.py      builds .xlsx workbook
    medical_potential/label_expansion/indication_standardizer.py  batch normalizer

Usage (run as package)
----------------------
    python -m medical_potential.label_expansion
    python -m medical_potential.label_expansion tirzepatide
    python -m medical_potential.label_expansion tirzepatide --company "Eli Lilly"

Programmatic usage
-------------------
    from medical_potential.label_expansion.label_expansion import run
    output_path = run("tirzepatide")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import medical_potential.config as config
from medical_potential.label_expansion import excel_writer
from medical_potential.label_expansion import research_modules

logger = logging.getLogger(__name__)


def run(drug: str | None = None, company: str | None = None) -> Path:
    """Run the full label-expansion workflow for a single drug.

    Parameters
    ----------
    drug:
        Molecule/drug name to research. Defaults to `config.INPUT_DRUG`.
    company:
        Innovator/originator company. Defaults to `config.INPUT_COMPANY`
        (blank → auto-detected via Gemini + Google Search).

    Returns
    -------
    Path to the generated Excel workbook.
    """
    drug = (drug or config.INPUT_DRUG or "").strip()
    if not drug:
        raise ValueError("No drug provided — set INPUT_DRUG in config.py or pass run(drug=...)")
    company = company if company is not None else config.INPUT_COMPANY

    logger.info("=" * 70)
    logger.info("LABEL EXPANSION PIPELINE — %s", drug.title())
    logger.info("=" * 70)

    clinical_rows = research_modules.run_clinical_efficacy(drug)
    indication_rows = research_modules.run_indication_research(drug, company or None)

    if not clinical_rows and not indication_rows:
        logger.warning("No data found for '%s' from either Clinical Trials or Indication Research", drug)

    out_file = excel_writer.write_excel(drug, clinical_rows, indication_rows)

    logger.info("=" * 70)
    logger.info("DONE — Clinical Efficacy: %d | Indication Research: %d | Combined: %d",
                len(clinical_rows), len(indication_rows), len(clinical_rows) + len(indication_rows))
    logger.info("Output: %s", out_file)
    logger.info("=" * 70)
    return out_file


def _warn_if_missing_config() -> None:
    """Give the person an early, clear heads-up about likely-required
    config values that are still blank, without hard-failing (some may be
    legitimately blank, e.g. PROJECT_ID under Application Default
    Credentials)."""
    if not config.PROJECT_ID:
        logger.warning("config.PROJECT_ID is not set — BigQuery/GCS clients will fall back to ADC's default project.")
    if not config.BQ_DATASET_ID:
        logger.warning("config.BQ_DATASET_ID is not set — the clinical-trials query will fail.")
    if not config.GCS_BUCKET:
        logger.warning("config.GCS_BUCKET is not set — checkpoints cannot be read/written to GCS.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run the label-expansion pipeline for a single drug.",
    )
    parser.add_argument(
        "drug", nargs="?", default=None,
        help="Optional: override INPUT_DRUG from medical_potential/config.py for this run only.",
    )
    parser.add_argument(
        "--company", default=None,
        help="Optional: override INPUT_COMPANY from medical_potential/config.py for this run only.",
    )
    args = parser.parse_args()

    if not config.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. Add it to your .env file or environment.")
        sys.exit(1)

    _warn_if_missing_config()

    try:
        run(drug=args.drug, company=args.company)
    except Exception:
        logger.exception("Label-expansion pipeline failed for '%s'", args.drug or config.INPUT_DRUG)
        sys.exit(1)


if __name__ == "__main__":
    main()
