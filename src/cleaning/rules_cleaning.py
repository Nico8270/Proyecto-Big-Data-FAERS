"""
src/cleaning/rules_cleaning.py
================================
Reglas de limpieza centralizadas — sin magia hardcodeada.

Cualquier cambio de regla se modifica aquí, no hay que tocar los scripts
individuales s01_…s07_.
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA     = PROJECT_ROOT / "data" / "raw" / "faers"
CLEAN_DATA   = PROJECT_ROOT / "data" / "clean_data"

FAERS_SEP      = "$"
FAERS_ENCODING = "latin-1"


# ── Reglas DEMO ───────────────────────────────────────────────────────────────

RULES_DEMO = {
    # Columnas que se eliminan completamente si superan el umbral de nulos
    "drop_columns_threshold_pct": 60,

    # Valores de edad válidos
    "edad_min":       0,
    "edad_max":      120,

    # Códigos de sexo aceptados
    "sexo_validos":  {"F", "M", "U"},
    "sexo_default":  "U",

    # Códigos de outcome aceptados (FAERS oficial)
    "outcome_validos": {
        "DE", "LT", "HO", "DS", "CA", "RI", "OT", "",
    },
}

# ── Reglas DRUG ───────────────────────────────────────────────────────────────

RULES_DRUG = {
    "drop_columns_threshold_pct": 90,
    "ruta_default":               "UNKNOWN",
    "dosis_min":                  0,
    "dosis_max":                10_000,
    # Códigos de rol aceptados
    "rol_validos":   {"PS", "SS", "C", "I"},
    "rol_default":   "UNK",
    # Códigos de suspensión/reinicio
    "dechal_validos": {"Y", "N", "U"},
    "rechal_validos": {"Y", "N", "U"},
}

# ── Reglas REAC ───────────────────────────────────────────────────────────────

RULES_REAC = {
    # Términos sinónimos → término canónico MedDRA PT
    "sinonimos": {
        "NAUSEA":      "Nausea",
        "NAUSEA/VOMITING": "Nausea",
        "DYSPNOEA":    "Dyspnea",
        "DYSPNEA":     "Dyspnea",
        "DYSPNOE":     "Dyspnea",
        "HEADACHE":    "Cefalea",
        "HEAD ACHE":   "Cefalea",
        "DIZZINESS":   "Mareo",
        "DIZZY":       "Mareo",
        "FATIGUE":     "Fatiga",
        "FATIGUED":    "Fatiga",
        "PAIN":        "Dolor",
        "VOMITING":    "Vomito",
        "RASH":        "Exantema",
        "SKIN RASH":   "Exantema",
        "PRURITUS":    "Prurito",
        "ITCHING":     "Prurito",
        "ANXIETY":     "Ansiedad",
        "ANXIEDAD":    "Ansiedad",
        "INSOMNIA":    "Insomnio",
        "SLEEP DISORDER": "Insomnio",
        "FEVER":       "Fiebre",
        "PYREXIA":     "Fiebre",
        "ARTHRALGIA":  "Artralgia",
        "JOINT PAIN":  "Artralgia",
        "MYALGIA":     "Mialgia",
        "MUSCLE PAIN": "Mialgia",
        "DIARRHOEA":   "Diarrea",
        "DIARRHEA":    "Diarrea",
        "DIARRHEA/DYSPNOEA": "Diarrea",
        "CONSTIPATION":"Estrenimiento",
        "ABDOMINAL PAIN": "Dolor abdominal",
        "ABDOMINAL DISCOMFORT": "Dolor abdominal",
        "BACK PAIN":   "Dolor lumbar",
        "CHEST PAIN":  "Dolor toracico",
        "CHEST PAIN/PRESSURE": "Dolor toracico",
        "COUGH":       "Tos",
        "DYSPHAGIA":   "Disfagia",
        "DYSPNEA/CHEST PAIN": "Dificultad respiratoria",
        "PALPITATIONS": "Palpitaciones",
        "PALPITATION": "Palpitaciones",
        "SYNCOPE":     "Sincope",
        "TREMOR":      "Temblor",
        "TREMORS":     "Temblor",
        "DEPRESSION":  "Depresion",
        "DEPRESSIVE DISORDER": "Depresion",
        "CONFUSION":   "Confusion",
        "CONFUSIONAL STATE": "Confusion",
        "MEMORY LOSS": "Perdida de memoria",
        "MEMORY IMPAIRMENT": "Perdida de memoria",
        "VERTIGO":     "Vertigo",
        "BLURRED VISION": "Vision borrosa",
        "BLURRY VISION": "Vision borrosa",
        "VISUAL IMPAIRMENT": "Vision borrosa",
        "TINNITUS":    "Tinitus",
        "HYPOacusis":  "Hipoacusia",
        "HEARING LOSS": "Hipoacusia",
        "DEAFNESS":    "Hipoacusia",
        "PNEUMONIA":   "Neumonia",
        "BRONCHITIS":  "Bronquitis",
        "BRONCHOSPASM": "Broncoespasmo",
        "ASTHMA":      "Asma",
        "COPD":        "EPOC",
        "UTI":         "Infeccion urinaria",
        "URINARY TRACT INFECTION": "Infeccion urinaria",
        "CELLULITIS":  "Celulitis",
        "CELLULITE":   "Celulitis",
        "INFECTION":   "Infeccion",
        "SEPTICAEMIA": "Septicemia",
        "SEPSIS":      "Septicemia",
        "SEPTIC SHOCK": "Choque septico",
        "ANAPHYLAXIS": "Anafilaxia",
        "ANGIOEDEMA":  "Angioedema",
        "URTICARIA":   "Urticaria",
        "ANGINA":      "Angina",
        "ARRHYTHMIA":  "Arritmia",
        "ATRIAL FIBRILLATION": "Fibrilacion auricular",
        "AF":          "Fibrilacion auricular",
        "HYPERTENSION": "Hipertension",
        "HIGH BLOOD PRESSURE": "Hipertension",
        "HYPOTENSION": "Hipotension",
        "LOW BLOOD PRESSURE": "Hipotension",
        "STROKE":      "Accidente cerebrovascular",
        "CVA":         "Accidente cerebrovascular",
        "AFIB":        "Fibrilacion auricular",
    },
    "pt_min_len":  3,  # Filtrar PT con menos de 3 caracteres (ruido)
}

# ── Reglas INDI ───────────────────────────────────────────────────────────────

RULES_INDI = {
    "columna_texto":   "indi_pt",
    "pt_min_len":      3,
    "default_valor":   "NO INFORMADA",
}

# ── Reglas OUTC ───────────────────────────────────────────────────────────────

RULES_OUTC = {
    # Códigos de outcome válidos oficialmente FAERS
    "validos":       {"DE", "LT", "HO", "DS", "CA", "RI", "OT"},
    "default_code":  "OT",           # código residual seguro
    "default_label": "NO_INFORMADO",  # para cuando outc_cod está vacío
}

# ── Reglas CONSILIDAR ─────────────────────────────────────────────────────────

RULES_CONSOLIDAR = {
    "join_type": "inner",         # inner join por primaryid
}
