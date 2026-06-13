"""
config_balance.py
=================
Configuracion centralizada para todo el flujo de balanceo FAERS.
Importar desde aqui — no hay constantes duplicadas en otros scripts.
"""

from pathlib import Path

# ── Rutas (relativas al proyecto) ───────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEAN_DATA   = PROJECT_ROOT / "data" / "clean_data"
CONSOLIDADO  = CLEAN_DATA / "dataset_consolidado.parquet"
BALANCED_DIR = PROJECT_ROOT / "data" / "balanced_data"
BAL_REPORTS  = PROJECT_ROOT / "outputs" / "balance_reports"
BALANCE_DIR  = Path(__file__).parent
LOGS_DIR     = BALANCE_DIR / "logs"

# Crear carpetas de salida si no existen
for d in [BAL_REPORTS, BALANCED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Entradas del EDA ────────────────────────────────────────────────────────────

EDA_OUTPUTS  = PROJECT_ROOT / "outputs" / "eda_results"
PLAN_CSV     = EDA_OUTPUTS / "s09_resumen_completo.csv"
BALANCEO_CSV = EDA_OUTPUTS / "s09_balanceo.csv"

# ── FAERS ──────────────────────────────────────────────────────────────────────

FAERS_SEP      = "$"
FAERS_ENCODING = "latin-1"

# ── Variable objetivo ──────────────────────────────────────────────────────────

TARGET_COL      = "severity_level"
TARGET_VALUES   = {1, 2, 3, 4, 5}
ID_COL          = "primaryid"

# ── Semillas ───────────────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── Parámetros SMOTE ───────────────────────────────────────────────────────────

SMOTE_K_NEIGHBORS  = 5
SMOTE_SAMPLING     = "auto"

# ── Parámetros ADASYN ──────────────────────────────────────────────────────────

ADASYN_K_NEIGHBORS = 5

# ── Parámetros ENN ─────────────────────────────────────────────────────────────

ENN_N_NEIGHBORS    = 3
ENN_KIND_SEL       = "mode"

# ── Parámetros Tomek ───────────────────────────────────────────────────────────

TOMEK_SAMPLING     = "auto"

# ── Parámetros Borderline-SMOTE ────────────────────────────────────────────────

BORDERLINE_KIND    = "borderline-1"  # borderline-1 usa solo mayoria vecinas
BORDERLINE_K_NEIGH = 5

# ── Umbrales de diagnóstico ─────────────────────────────────────────────────────

TAMANO_PEQ_MAX        = 30_000      # datasets menores NO reciben undersampling
TAMANO_GRANDE_MIN     = 300_000     # datasets mayores: cualquier técnica

UMBRAL_SIN_ACCION     = 2.0         # ratio máximo sin intervención
UMBRAL_CATEGORICAS    = 0.30        # >30 % de columnas object → SMOTE-NC
UMBRAL_CLASE_MIN_PCT  = 0.01        # <1 % clase minoritaria → ADASYN
UMBRAL_CARDINALIDAD   = 50          # cardinalidad alta por columna categórica

# ── Salida ─────────────────────────────────────────────────────────────────────

ARCHIVO_SALIDA     = "dataset.parquet"
ARCHIVO_INFORME    = "informe_balanceo.txt"
ARCHIVO_METRICAS   = "comparativo.csv"
CACHE_DIAGNOSTICO  = BALANCE_DIR / ".cache_diagnostico.json"
