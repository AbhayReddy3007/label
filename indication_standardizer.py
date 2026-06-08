"""
Indication Standardizer
========================
Shared module for normalizing drug indication names and classifying
them as primary or secondary.

Usage:
    from indication_standardizer import standardize_indication, classify_indication, process_indications
"""

import re

# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZATION MAP
#  Keys: lowercase pattern → standardized name
#  Order matters — first match wins, so put more specific patterns first.
# ══════════════════════════════════════════════════════════════════════════════

NORMALIZATION_RULES: list[tuple[str, str]] = [
    # ── Diabetes ──────────────────────────────────────────────────────────────
    (r"type\s*2\s*diabetes\s*mellitus",            "T2DM"),
    (r"type\s*2\s*diabetes",                       "T2DM"),
    (r"type\s*ii\s*diabetes",                      "T2DM"),
    (r"\bt2dm\b",                                  "T2DM"),
    (r"\bt2d\b",                                   "T2DM"),
    (r"type\s*1\s*diabetes\s*mellitus",            "T1DM"),
    (r"type\s*1\s*diabetes",                       "T1DM"),
    (r"type\s*i\s*diabetes",                       "T1DM"),
    (r"\bt1dm\b",                                  "T1DM"),
    (r"\bt1d\b",                                   "T1DM"),
    (r"gestational\s*diabetes\s*mellitus",         "Gestational Diabetes"),
    (r"gestational\s*diabetes",                    "Gestational Diabetes"),
    (r"\bgdm\b",                                   "Gestational Diabetes"),
    (r"diabetic\s*kidney\s*disease",               "Diabetic Kidney Disease"),
    (r"diabetic\s*nephropathy",                    "Diabetic Kidney Disease"),
    (r"diabetic\s*retinopathy",                    "Diabetic Retinopathy"),
    (r"diabetic\s*neuropathy",                     "Diabetic Neuropathy"),
    (r"diabetic\s*foot\s*ulcer",                   "Diabetic Foot Ulcer"),
    (r"diabetes\s*mellitus",                       "Diabetes Mellitus"),

    # ── Obesity / Weight ──────────────────────────────────────────────────────
    (r"overweight\s*(and|or|/)\s*obesity",          "Obesity"),
    (r"obesity\s*(and|or|/)\s*overweight",          "Obesity"),
    (r"overweight\s*\/\s*obesity",                  "Obesity"),
    (r"\bobesity\b",                                "Obesity"),
    (r"\boverweight\b",                             "Obesity"),
    (r"weight\s*management",                        "Obesity"),
    (r"weight\s*loss",                              "Obesity"),
    (r"weight\s*reduction",                         "Obesity"),
    (r"body\s*weight\s*management",                 "Obesity"),
    (r"anti[\s-]?obesity",                          "Obesity"),

    # ── Cardiovascular ────────────────────────────────────────────────────────
    (r"non[\s-]?alcoholic\s*steatohepatitis",       "NASH"),
    (r"nonalcoholic\s*steatohepatitis",             "NASH"),
    (r"\bnash\b",                                   "NASH"),
    (r"metabolic[\s-]?dysfunction[\s-]?associated\s*steatohepatitis", "MASH"),
    (r"\bmash\b",                                   "MASH"),
    (r"non[\s-]?alcoholic\s*fatty\s*liver\s*disease", "NAFLD"),
    (r"nonalcoholic\s*fatty\s*liver",               "NAFLD"),
    (r"\bnafld\b",                                  "NAFLD"),
    (r"metabolic[\s-]?associated\s*fatty\s*liver",  "MAFLD"),
    (r"\bmafld\b",                                  "MAFLD"),
    (r"heart\s*failure\s*with\s*preserved\s*ejection\s*fraction", "HFpEF"),
    (r"\bhfpef\b",                                  "HFpEF"),
    (r"heart\s*failure\s*with\s*reduced\s*ejection\s*fraction", "HFrEF"),
    (r"\bhfref\b",                                  "HFrEF"),
    (r"chronic\s*heart\s*failure",                  "Heart Failure"),
    (r"congestive\s*heart\s*failure",               "Heart Failure"),
    (r"\bheart\s*failure\b",                        "Heart Failure"),
    (r"\bchf\b",                                    "Heart Failure"),
    (r"major\s*adverse\s*cardiovascular\s*event",   "MACE Reduction"),
    (r"\bmace\b",                                   "MACE Reduction"),
    (r"cardiovascular\s*risk\s*reduction",          "CV Risk Reduction"),
    (r"cardiovascular\s*disease",                   "Cardiovascular Disease"),
    (r"\bcvd\b",                                    "Cardiovascular Disease"),
    (r"atherosclerotic\s*cardiovascular",           "ASCVD"),
    (r"\bascvd\b",                                  "ASCVD"),
    (r"peripheral\s*arterial?\s*disease",           "Peripheral Artery Disease"),
    (r"\bpad\b",                                    "Peripheral Artery Disease"),
    (r"coronary\s*artery\s*disease",               "Coronary Artery Disease"),
    (r"\bcad\b",                                    "Coronary Artery Disease"),
    (r"atrial\s*fibrillation",                     "Atrial Fibrillation"),
    (r"\baf\b(?!.)",                               "Atrial Fibrillation"),
    (r"hypertension",                              "Hypertension"),
    (r"high\s*blood\s*pressure",                   "Hypertension"),
    (r"pulmonary\s*arterial\s*hypertension",       "Pulmonary Arterial Hypertension"),
    (r"\bpah\b",                                   "Pulmonary Arterial Hypertension"),
    (r"myocardial\s*infarction",                   "Myocardial Infarction"),
    (r"\bmi\b",                                    "Myocardial Infarction"),
    (r"stroke\s*prevention",                       "Stroke Prevention"),
    (r"\bstroke\b",                                "Stroke"),

    # ── Kidney ────────────────────────────────────────────────────────────────
    (r"chronic\s*kidney\s*disease",                "CKD"),
    (r"\bckd\b",                                   "CKD"),
    (r"end[\s-]?stage\s*renal\s*disease",          "ESRD"),
    (r"\besrd\b",                                  "ESRD"),
    (r"acute\s*kidney\s*injury",                   "Acute Kidney Injury"),
    (r"\baki\b",                                   "Acute Kidney Injury"),

    # ── Respiratory ───────────────────────────────────────────────────────────
    (r"chronic\s*obstructive\s*pulmonary\s*disease", "COPD"),
    (r"\bcopd\b",                                  "COPD"),
    (r"\basthma\b",                                "Asthma"),
    (r"obstructive\s*sleep\s*apn[eo]a",            "OSA"),
    (r"sleep\s*apn[eo]a",                          "OSA"),
    (r"\bosa\b",                                   "OSA"),
    (r"idiopathic\s*pulmonary\s*fibrosis",         "IPF"),
    (r"\bipf\b",                                   "IPF"),

    # ── Neuro / Psych ─────────────────────────────────────────────────────────
    (r"alzheimer.?s?\s*disease",                   "Alzheimer's Disease"),
    (r"\bad\b",                                    "Alzheimer's Disease"),
    (r"parkinson.?s?\s*disease",                   "Parkinson's Disease"),
    (r"major\s*depressive\s*disorder",             "MDD"),
    (r"\bmdd\b",                                   "MDD"),
    (r"\bdepression\b",                            "Depression"),
    (r"multiple\s*sclerosis",                      "Multiple Sclerosis"),
    (r"\bms\b",                                    "Multiple Sclerosis"),
    (r"epilepsy",                                  "Epilepsy"),
    (r"migraine",                                  "Migraine"),
    (r"schizophrenia",                             "Schizophrenia"),
    (r"bipolar\s*disorder",                        "Bipolar Disorder"),
    (r"attention[\s-]?deficit",                    "ADHD"),
    (r"\badhd\b",                                  "ADHD"),
    (r"substance\s*use\s*disorder",                "Substance Use Disorder"),
    (r"alcohol\s*use\s*disorder",                  "Alcohol Use Disorder"),
    (r"opioid\s*use\s*disorder",                   "Opioid Use Disorder"),
    (r"addiction",                                 "Addiction"),

    # ── Oncology ──────────────────────────────────────────────────────────────
    (r"non[\s-]?small\s*cell\s*lung\s*cancer",     "NSCLC"),
    (r"\bnsclc\b",                                 "NSCLC"),
    (r"small\s*cell\s*lung\s*cancer",              "SCLC"),
    (r"\bsclc\b",                                  "SCLC"),
    (r"hepatocellular\s*carcinoma",                "HCC"),
    (r"\bhcc\b",                                   "HCC"),
    (r"renal\s*cell\s*carcinoma",                  "RCC"),
    (r"\brcc\b",                                   "RCC"),
    (r"breast\s*cancer",                           "Breast Cancer"),
    (r"prostate\s*cancer",                         "Prostate Cancer"),
    (r"colorectal\s*cancer",                       "Colorectal Cancer"),
    (r"pancreatic\s*cancer",                       "Pancreatic Cancer"),
    (r"ovarian\s*cancer",                          "Ovarian Cancer"),
    (r"bladder\s*cancer",                          "Bladder Cancer"),
    (r"gastric\s*cancer",                          "Gastric Cancer"),
    (r"stomach\s*cancer",                          "Gastric Cancer"),
    (r"melanoma",                                  "Melanoma"),
    (r"leukemia",                                  "Leukemia"),
    (r"lymphoma",                                  "Lymphoma"),
    (r"myeloma",                                   "Myeloma"),
    (r"glioblastoma",                              "Glioblastoma"),

    # ── Autoimmune / Inflammatory ─────────────────────────────────────────────
    (r"rheumatoid\s*arthritis",                    "Rheumatoid Arthritis"),
    (r"\bra\b",                                    "Rheumatoid Arthritis"),
    (r"osteoarthritis",                            "Osteoarthritis"),
    (r"\boa\b",                                    "Osteoarthritis"),
    (r"psoriatic\s*arthritis",                     "Psoriatic Arthritis"),
    (r"ankylosing\s*spondylitis",                  "Ankylosing Spondylitis"),
    (r"systemic\s*lupus\s*erythematosus",          "SLE"),
    (r"\bsle\b",                                   "SLE"),
    (r"\blupus\b",                                 "SLE"),
    (r"crohn.?s?\s*disease",                       "Crohn's Disease"),
    (r"ulcerative\s*colitis",                      "Ulcerative Colitis"),
    (r"inflammatory\s*bowel\s*disease",            "IBD"),
    (r"\bibd\b",                                   "IBD"),
    (r"irritable\s*bowel\s*syndrome",              "IBS"),
    (r"\bibs\b",                                   "IBS"),
    (r"psoriasis",                                 "Psoriasis"),
    (r"atopic\s*dermatitis",                       "Atopic Dermatitis"),
    (r"\beczema\b",                                "Atopic Dermatitis"),

    # ── Metabolic ─────────────────────────────────────────────────────────────
    (r"metabolic\s*syndrome",                      "Metabolic Syndrome"),
    (r"dyslipid[ae]mia",                           "Dyslipidemia"),
    (r"hyperlipid[ae]mia",                         "Dyslipidemia"),
    (r"hypercholesterol[ae]mia",                   "Hypercholesterolemia"),
    (r"hypertriglycerid[ae]mia",                   "Hypertriglyceridemia"),
    (r"gout",                                      "Gout"),
    (r"hyperuric[ae]mia",                          "Hyperuricemia"),
    (r"osteoporosis",                              "Osteoporosis"),
    (r"polycystic\s*ovary",                        "PCOS"),
    (r"\bpcos\b",                                  "PCOS"),

    # ── Infectious ────────────────────────────────────────────────────────────
    (r"hepatitis\s*b",                             "Hepatitis B"),
    (r"\bhbv\b",                                   "Hepatitis B"),
    (r"hepatitis\s*c",                             "Hepatitis C"),
    (r"\bhcv\b",                                   "Hepatitis C"),
    (r"hiv",                                       "HIV"),
    (r"tuberculosis",                              "Tuberculosis"),
    (r"\btb\b",                                    "Tuberculosis"),

    # ── Eye ───────────────────────────────────────────────────────────────────
    (r"age[\s-]?related\s*macular\s*degeneration", "AMD"),
    (r"\bamd\b",                                   "AMD"),
    (r"diabetic\s*macular\s*edema",                "DME"),
    (r"\bdme\b",                                   "DME"),
    (r"glaucoma",                                  "Glaucoma"),
]

# Compile once
_COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), std) for pat, std in NORMALIZATION_RULES]


# ══════════════════════════════════════════════════════════════════════════════
#  PRIMARY INDICATION MAP
#  drug (lowercase) → set of standardized indication names that are primary
# ══════════════════════════════════════════════════════════════════════════════

PRIMARY_INDICATIONS: dict[str, set[str]] = {
    # ── GLP-1 / incretin drugs ────────────────────────────────────────────────
    "semaglutide":      {"T2DM", "Obesity"},
    "liraglutide":      {"T2DM", "Obesity"},
    "tirzepatide":      {"T2DM", "Obesity"},
    "dulaglutide":      {"T2DM"},
    "exenatide":        {"T2DM"},
    "albiglutide":      {"T2DM"},
    "orforglipron":     {"T2DM", "Obesity"},
    "survodutide":      {"Obesity", "NASH"},
    "retatrutide":      {"T2DM", "Obesity"},
    "pemvidutide":      {"Obesity", "NASH"},
    "mazdutide":        {"T2DM", "Obesity"},
    "cagrilintide":     {"Obesity"},
    "amycretin":        {"Obesity"},
    "ecnoglutide":      {"T2DM", "Obesity"},
    "danuglipron":      {"T2DM", "Obesity"},
    "efinopegdutide":   {"Obesity", "NASH"},

    # ── SGLT2 inhibitors ──────────────────────────────────────────────────────
    "empagliflozin":    {"T2DM", "Heart Failure"},
    "dapagliflozin":    {"T2DM", "Heart Failure", "CKD"},
    "canagliflozin":    {"T2DM"},
    "ertugliflozin":    {"T2DM"},
    "sotagliflozin":    {"T2DM", "Heart Failure"},

    # ── DPP-4 inhibitors ──────────────────────────────────────────────────────
    "sitagliptin":      {"T2DM"},
    "linagliptin":      {"T2DM"},
    "saxagliptin":      {"T2DM"},
    "alogliptin":       {"T2DM"},
    "vildagliptin":     {"T2DM"},

    # ── Insulin ───────────────────────────────────────────────────────────────
    "insulin glargine": {"T2DM", "T1DM"},
    "insulin lispro":   {"T2DM", "T1DM"},
    "insulin aspart":   {"T2DM", "T1DM"},
    "insulin degludec": {"T2DM", "T1DM"},
    "icodec":           {"T2DM"},
    "insulin icodec":   {"T2DM"},

    # ── Other diabetes ────────────────────────────────────────────────────────
    "metformin":        {"T2DM"},
    "pioglitazone":     {"T2DM"},
    "glimepiride":      {"T2DM"},
    "glyburide":        {"T2DM"},
    "acarbose":         {"T2DM"},

    # ── Cardiovascular ────────────────────────────────────────────────────────
    "sacubitril/valsartan": {"Heart Failure"},
    "entresto":         {"Heart Failure"},
    "evolocumab":       {"Hypercholesterolemia", "ASCVD"},
    "alirocumab":       {"Hypercholesterolemia", "ASCVD"},
    "inclisiran":       {"Hypercholesterolemia"},
    "bempedoic acid":   {"Hypercholesterolemia"},
    "icosapent ethyl":  {"Hypertriglyceridemia", "CV Risk Reduction"},

    # ── NASH / liver ──────────────────────────────────────────────────────────
    "resmetirom":       {"NASH"},
    "obeticholic acid": {"NASH"},
    "lanifibranor":     {"NASH"},

    # ── Respiratory ───────────────────────────────────────────────────────────
    "dupilumab":        {"Atopic Dermatitis", "Asthma"},
    "benralizumab":     {"Asthma"},
    "mepolizumab":      {"Asthma"},
    "tezepelumab":      {"Asthma"},
    "itepekimab":       {"Asthma"},
    "omalizumab":       {"Asthma"},
    "pirfenidone":      {"IPF"},
    "nintedanib":       {"IPF"},

    # ── Alzheimer's ───────────────────────────────────────────────────────────
    "lecanemab":        {"Alzheimer's Disease"},
    "donanemab":        {"Alzheimer's Disease"},
    "aducanumab":       {"Alzheimer's Disease"},

    # ── Oncology (selected) ───────────────────────────────────────────────────
    "pembrolizumab":    {"NSCLC", "Melanoma"},
    "nivolumab":        {"NSCLC", "Melanoma", "RCC"},
    "atezolizumab":     {"NSCLC", "Breast Cancer"},
    "trastuzumab":      {"Breast Cancer", "Gastric Cancer"},
    "bevacizumab":      {"Colorectal Cancer", "NSCLC"},
    "osimertinib":      {"NSCLC"},
    "olaparib":         {"Ovarian Cancer", "Breast Cancer"},
    "palbociclib":      {"Breast Cancer"},
    "ribociclib":       {"Breast Cancer"},
    "abemaciclib":      {"Breast Cancer"},
    "lenvatinib":       {"HCC", "RCC"},
    "sorafenib":        {"HCC", "RCC"},
    "abiraterone":      {"Prostate Cancer"},
    "enzalutamide":     {"Prostate Cancer"},

    # ── Autoimmune ────────────────────────────────────────────────────────────
    "adalimumab":       {"Rheumatoid Arthritis", "Psoriasis", "Crohn's Disease"},
    "infliximab":       {"Rheumatoid Arthritis", "Crohn's Disease", "Ulcerative Colitis"},
    "secukinumab":      {"Psoriasis", "Psoriatic Arthritis", "Ankylosing Spondylitis"},
    "ixekizumab":       {"Psoriasis", "Psoriatic Arthritis"},
    "ustekinumab":      {"Psoriasis", "Crohn's Disease"},
    "risankizumab":     {"Psoriasis", "Crohn's Disease"},
    "guselkumab":       {"Psoriasis", "Psoriatic Arthritis"},
    "tofacitinib":      {"Rheumatoid Arthritis", "Ulcerative Colitis"},
    "baricitinib":      {"Rheumatoid Arthritis", "Atopic Dermatitis"},
    "upadacitinib":     {"Rheumatoid Arthritis", "Atopic Dermatitis", "Crohn's Disease"},
    "belimumab":        {"SLE"},
    "anifrolumab":      {"SLE"},
    "vedolizumab":      {"Ulcerative Colitis", "Crohn's Disease"},
    "ozanimod":         {"Ulcerative Colitis", "Multiple Sclerosis"},

    # ── OSA ───────────────────────────────────────────────────────────────────
    "solriamfetol":     {"OSA"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def standardize_indication(raw: str) -> str:
    """Normalize a single indication string to its standardized form."""
    text = raw.strip()
    if not text:
        return text
    for pattern, standard in _COMPILED_RULES:
        if pattern.search(text):
            return standard
    # Title-case fallback for unrecognized indications
    return text.title()


def classify_indication(drug: str, indication: str) -> str:
    """Return 'Primary' or 'Secondary' for a given drug + standardized indication."""
    drug_lower = drug.strip().lower()
    primary_set = PRIMARY_INDICATIONS.get(drug_lower, set())
    if not primary_set:
        # Unknown drug — all are secondary by default
        return "Secondary"
    return "Primary" if indication in primary_set else "Secondary"


def process_indications(drug: str, raw_indications: list[str]) -> list[dict]:
    """
    Take a list of raw indication strings, standardize, deduplicate,
    and classify each as Primary/Secondary.

    Returns a list of dicts:
        [{"indication": "T2DM", "indication_type": "Primary"}, ...]
    """
    seen = set()
    results = []
    for raw in raw_indications:
        std = standardize_indication(raw)
        if not std or std.lower() in ("error", "n/a", "no indication found", "none"):
            continue
        if std in seen:
            continue
        seen.add(std)
        results.append({
            "indication": std,
            "indication_type": classify_indication(drug, std),
        })
    return results
