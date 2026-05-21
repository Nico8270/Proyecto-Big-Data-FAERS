"""
config_eda.py
========================
Configuracion centralizada para todo el flujo EDA de FAERS.
Importar desde aqui — no hay constantes duplicadas en otros scripts.
"""

from pathlib import Path

# ── Rutas (relativas al proyecto) ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"  / "raw"  / "faers"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "eda_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── FAERS ────────────────────────────────────────────────────────────────────
FAERS_SEP      = "$"
FAERS_ENCODING = "latin-1"
FIGURE_DPI     = 120
FIGURE_SIZE    = (12, 6)

# ── Mapeos semanticos ───────────────────────────────────────────────────────
DRUG_ROLE_MAP = {
    "PS": "Sospechoso primario",
    "SS": "Sospechoso secundario",
    "C":  "Concomitante",
    "I":  "Interactuante",
}

OUTCOME_MAP = {
    "DE": "Muerte",
    "LT": "Amenaza de vida",
    "HO": "Hospitalizacion",
    "DS": "Discapacidad / Incapacidad",
    "CA": "Anomalia congenita",
    "RI": "Intervencion requerida",
    "OT": "Otro",
}

AGE_CODE_MAP = {
    "DEC": "Decadas", "YR": "Anos", "MO": "Meses",
    "WK":  "Semanas",  "DY": "Dias",  "HR": "Horas",
}

# ── Nombres de archivo FAERS ─────────────────────────────────────────────────
FAERS_FILES = {
    "DEMO": "DEMO25Q4.txt",
    "DRUG": "DRUG25Q4.txt",
    "REAC": "REAC25Q4.txt",
    "OUTC": "OUTC25Q4.txt",
    "INDI": "INDI25Q4.txt",
    "THER": "THER25Q4.txt",
    "RPSR": "RPSR25Q4.txt",
}

# ── Paletas de graficos ─────────────────────────────────────────────────────
PALETTE_MAIN   = "viridis"
PALETTE_CATEG  = "Set2"
PALETTE_SEQUEN = "coolwarm"
PALETTE_DANGER = "Reds_r"
