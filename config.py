"""
Central configuration for the medical_potential package and the
label_expansion pipeline.

All tunables are defined here as plain module-level constants.
Secrets and environment-specific values are sourced from the environment
or a local .env file — nothing sensitive is hardcoded.

Edit the values below, or override via environment variables / .env.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load a local .env file if present (does not error if missing).
load_dotenv(override=True)


def _env_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to *default* if unset/invalid."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ══════════════════════════════════════════════════════════════════════════
#  GOOGLE CLOUD — CREDENTIALS & PROJECT
# ══════════════════════════════════════════════════════════════════════════
# Path to a service-account JSON key file. Leave blank ("") to fall back to
# Application Default Credentials (e.g. on Cloud Run / GCE / a machine that
# already ran `gcloud auth application-default login`).
GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

PROJECT_ID: str = os.getenv("PROJECT_ID", "")

# ══════════════════════════════════════════════════════════════════════════
#  GOOGLE CLOUD — BIGQUERY
# ══════════════════════════════════════════════════════════════════════════
BQ_DATASET_ID: str = os.getenv("BQ_DATASET_ID", "")

# Table that clinical-trial rows are fetched from (used by label_expansion).
# Fully qualified as `{PROJECT_ID}.{BQ_DATASET_ID}.{CLINICAL_TRIALS_TABLE}`
# inside research_modules.py.
CLINICAL_TRIALS_TABLE: str = os.getenv("CLINICAL_TRIALS_TABLE", "clinical_efficacy")

# Table used by gcp_utils.append_dimension_score_to_bigquery (medical_potential pipeline).
DIM_SCORES_TABLE: str = os.getenv("DIM_SCORES_TABLE", "")

# ══════════════════════════════════════════════════════════════════════════
#  GOOGLE CLOUD — STORAGE (GCS)
# ══════════════════════════════════════════════════════════════════════════
GCS_BUCKET: str = os.getenv("GCS_BUCKET", "")

# Used by gcp_utils for medical_potential dimension report uploads.
GCS_REPORT_BASE_PATH: str = os.getenv("GCS_REPORT_BASE_PATH", "")

# Subfolder name for the medical_potential pillar inside GCS paths.
GCS_MEDICAL_POTENTIAL_SUBFOLDER: str = os.getenv("GCS_MEDICAL_POTENTIAL_SUBFOLDER", "")

# Root path for pipeline payload cache uploads (medical_potential pipeline).
GCS_PIPELINE_CACHE_BASE_PATH: str = os.getenv("GCS_PIPELINE_CACHE_BASE_PATH", "")

# Folder (prefix) inside GCS_BUCKET where label_expansion checkpoint JSON
# files are read from / written to.
GCS_LE_CHECKPOINTS_PATH: str = os.getenv("GCS_LE_CHECKPOINTS_PATH", "label_expansion/checkpoints")

# ══════════════════════════════════════════════════════════════════════════
#  DRUG INPUT — label_expansion pipeline
# ══════════════════════════════════════════════════════════════════════════
# The one molecule to run (single drug only). Change before each run,
# e.g. INPUT_DRUG = "semaglutide", or set the env var.
INPUT_DRUG: str = os.getenv("INPUT_DRUG", "semaglutide")

# Optional: innovator/originator company for INPUT_DRUG.
# Leave blank ("") to have it auto-detected via Gemini + Google Search.
INPUT_COMPANY: str = os.getenv("INPUT_COMPANY", "")

# ══════════════════════════════════════════════════════════════════════════
#  GEMINI
# ══════════════════════════════════════════════════════════════════════════
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Gemini model used for every extraction/classification call in the pipeline.
# Replace the default with whichever Flash preview model you intend to use.
GEMINI_FLASH_PREVIEW_MODEL: str = os.getenv("GEMINI_FLASH_PREVIEW_MODEL", "gemini-2.5-flash")

# Retry behaviour for transient Gemini errors (503 / 429 / timeouts / etc).
GEMINI_MAX_RETRIES: int = _env_int("GEMINI_MAX_RETRIES", 4)
GEMINI_RETRY_BASE_DELAY_SECONDS: int = _env_int("GEMINI_RETRY_BASE_DELAY_SECONDS", 5)

# ══════════════════════════════════════════════════════════════════════════
#  CONCURRENCY — label_expansion pipeline
# ══════════════════════════════════════════════════════════════════════════
MAX_WORKERS_TRIALS: int = _env_int("MAX_WORKERS_TRIALS", 10)      # clinical-trial extraction
MAX_WORKERS_INNOVATOR: int = _env_int("MAX_WORKERS_INNOVATOR", 5)  # innovator-source research

# ══════════════════════════════════════════════════════════════════════════
#  OUTPUT — label_expansion pipeline
# ══════════════════════════════════════════════════════════════════════════
# Fixed output directory, reused across runs. The Excel file for INPUT_DRUG
# is written here as `<drug>_label_expansion.xlsx`.
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
