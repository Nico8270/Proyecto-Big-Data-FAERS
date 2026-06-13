"""
src/balance/rules_balance.py
=============================
Umbrales, catálogo de técnicas y funciones de decisión.
Modificar aquí para cambiar el comportamiento del orquestador sin tocar
ni balance_main.py ni los scripts individuales.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BAL_REPORTS  = PROJECT_ROOT / "outputs" / "balance_reports"
BAL_REPORTS.mkdir(parents=True, exist_ok=True)

# ── Umbrales ───────────────────────────────────────────────────────────────────

RATIO_SIN_ACCION    = 2.0
RATIO_SMOTE         = 5.0
RATIO_SMOTEENN      = 8.0

TAMANO_PEQ_MAX      = 30_000
TAMANO_GRANDE_MIN   = 300_000

PCT_CATEGORICAS_LIM = 0.30
CLASE_MUY_MIN_PCT   = 0.01
CARDINALIDAD_LIM    = 50

# ── Catálogo de técnicas ───────────────────────────────────────────────────────

TECNICAS = {
    "random_undersampling": {
        "label":    "Random Undersampling",
        "script":   "src/balance/undersampling/s01_random_undersampling.py",
        "familia":  "undersampling",
        "desc":     "Elimina mayoria al azar hasta igualar la clase minoritaria.",
    },
    "tomek": {
        "label":    "Tomek Links",
        "script":   "src/balance/undersampling/s02_tomek_links.py",
        "familia":  "undersampling",
        "desc":     "Elimina pares mayoria-minoria fronterizos sin generar datos nuevos.",
    },
    "smote": {
        "label":    "SMOTE",
        "script":   "src/balance/oversampling/s03_smote.py",
        "familia":  "oversampling",
        "desc":     "Genera sinteticas por interpolacion euclidiana en features numericas.",
    },
    "adasyn": {
        "label":    "ADASYN",
        "script":   "src/balance/oversampling/s04_adasyn.py",
        "familia":  "oversampling",
        "desc":     "Genera mas sinteticas en las muestras minoritarias mas dificiles de clasificar.",
    },
    "borderline_smote": {
        "label":    "Borderline-SMOTE",
        "script":   "src/balance/oversampling/s05_borderline_smote.py",
        "familia":  "oversampling",
        "desc":     "SMOTE solo sobre muestras fronterizas, evita generar en zonas limpias.",
    },
    "smote_nc": {
        "label":    "SMOTE-NC",
        "script":   "src/balance/oversampling/s06_smote_nc.py",
        "familia":  "oversampling",
        "desc":     "SMOTE en numericas; categoricas se preservan sin interpolar.",
    },
    "smoteenn": {
        "label":    "SMOTE + ENN",
        "script":   "src/balance/hibridas/s07_smoteenn.py",
        "familia":  "hibrida",
        "desc":     "SMOTE genera sinteticas y ENN elimina muestras ruidosas.",
    },
}

FAMILIAS = {
    "undersampling": ["random_undersampling", "tomek"],
    "oversampling":  ["smote", "adasyn", "borderline_smote", "smote_nc"],
    "hibrida":       ["smoteenn"],
}

# ── Funciones de consulta ───────────────────────────────────────────────────────

def ratio_ok(ratio: float) -> bool:
    return ratio <= RATIO_SIN_ACCION


def es_pequeno(n_filas: int) -> bool:
    return n_filas < TAMANO_PEQ_MAX


def es_grande(n_filas: int) -> bool:
    return n_filas >= TAMANO_GRANDE_MIN


# ── Seleccion de técnicas ───────────────────────────────────────────────────────

def recomendar_tecnicas(ratio: float, n_filas: int,
                          pct_categoricas: float = 0.0,
                          clase_min_pct:   float = 0.0,
                          card_prom:       float = 0.0) -> list:
    """
    Devuelve la lista de claves recomendadas.

    Criterios aplicados en orden:
      1. Tamano del dataset (chico → solo oversampling puro)
      2. Ratio de desbalance
      3. % de columnas categoricas
      4. Tamanio de la clase minoritaria mas chica
    """
    if ratio_ok(ratio):
        return []

    peq   = es_pequeno(n_filas)
    gran  = es_grande(n_filas)
    cat   = pct_categoricas >= PCT_CATEGORICAS_LIM
    min_  = clase_min_pct   < CLASE_MUY_MIN_PCT
    card  = card_prom       > CARDINALIDAD_LIM
    r_alto  = ratio >= RATIO_SMOTEENN
    r_medio = RATIO_SMOTE <= ratio < RATIO_SMOTEENN

    rec = []

    # Dataset chico → oversampling puro, sin undersampling de ningun tipo
    if peq:
        rec.append("adasyn" if min_ else "smote")
        if cat or card:
            rec.append("smote_nc")
        if r_alto:
            rec.append("smoteenn")

    # Dataset grande → cualquier tecnica es viable
    elif gran:
        rec.append("random_undersampling")
        if ratio >= RATIO_SMOTE:
            rec.append("smote")
        if r_alto:
            rec.append("smoteenn")
        if min_:
            rec.append("adasyn")
        if cat or card:
            rec.append("smote_nc")

    # Tamano medio (30 K – 300 K)
    else:
        rec.append("random_undersampling")
        if ratio >= RATIO_SMOTE:
            rec.append("smote")
        if r_alto:
            rec.append("smoteenn")
        if cat or card:
            rec.append("smote_nc")
        if min_:
            rec.append("adasyn")
        if r_medio:
            rec.append("borderline_smote")

    # Eliminar duplicados manteniendo el orden de insercion
    visto: dict = {}
    for k in rec:
        visto.setdefault(k, True)
    return list(visto)


# ── Tecnica ganadora ─────────────────────────────────────────────────────────────

def elegir_tecnica_ganadora(ratio: float, n_filas: int,
                             pct_categoricas: float = 0.0,
                             clase_min_pct:    float = 0.0,
                             card_prom:        float = 0.0) -> dict:
    """Elige una tecnica ganadora con justificacion textual."""
    rec = recomendar_tecnicas(ratio, n_filas, pct_categoricas,
                               clase_min_pct, card_prom)
    if not rec:
        return {"ganadora": None, "alternativas": [], "justificacion": "",
                "tecnica": ""}

    # Jerarquia: hibrida > oversampling especializado > oversampling estandar
    orden = ["smoteenn", "smote_nc", "adasyn",
             "borderline_smote", "smote", "tomek", "random_undersampling"]
    just = {
        "smoteenn":          "SMOTE+ENN genera sinteticas y limpia ruido posterior.",
        "smote_nc":          "SMOTE-NC evita interpolar valores categoricos inexistentes.",
        "adasyn":            "ADASYN se concentra en las muestras minoritarias mas dificiles.",
        "borderline_smote":  "Borderline-SMOTE evita sinteticas en zonas limpias de mayoria.",
        "smote":             "SMOTE genera sinteticas uniformes sobre la clase minoritaria.",
        "tomek":             "Tomek elimina pares fronterizos sin agregar datos.",
        "random_undersampling":"Random Undersampling recorta mayoria sin generar sinteticas.",
    }

    for clave in orden:
        if clave in rec:
            return {
                "ganadora":     clave,
                "alternativas": [k for k in rec if k != clave],
                "justificacion": just[clave],
                "tecnica":      TECNICAS[clave]["label"],
            }

    return {"ganadora": rec[0], "alternativas": rec[1:],
            "justificacion": "Primera recomendada.",
            "tecnica": TECNICAS[rec[0]]["label"]}

# ── Criterio de exito ────────────────────────────────────────────────────────────

def ratio_ok(ratio_desbalance: float) -> bool:
    return ratio_desbalance <= RATIO_SIN_ACCION
