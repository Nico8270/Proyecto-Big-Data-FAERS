"""
src/balance/balance_main.py
=============================
Orquestador del pipeline de balanceo FAERS.

Entrada:
  - data/clean_data/dataset_consolidado.parquet  (limpio, unido)
  - outputs/eda_results/s09_resumen_completo.csv (plan EDA — tipos de features)
  - outputs/eda_results/s09_balanceo.csv         (conteos por clase)

Salida:
  - data/balanced_data/{clave}/dataset.parquet
  - outputs/balance_reports/informe_balanceo.txt
  - outputs/balance_reports/comparativo.csv

Flujo:
  [PASO 0] Diagnostico inteligente — lee EDA + consolidado, NO modifica datos.
  [PASO 1] Recomendacion automatica basada en 4 criterios (tamano, ratio,
           % categoricas, clase minoritaria).
  [PASO 2] Submenu con la tecnica ganadora preseleccionada.
  [PASO 3] Ejecucion de las tecnicas elegidas.
  [PASO 4] Comparativo de resultados.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Rutas y constantes ─────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from balance.config_balance import (
    CONSOLIDADO, PLAN_CSV, BALANCEO_CSV,
    BAL_REPORTS, BALANCED_DIR, TARGET_COL, LOGS_DIR,
    CACHE_DIAGNOSTICO, TARGET_VALUES,
)
from balance.rules_balance import (
    TECNICAS, recomendar_tecnicas, elegir_tecnica_ganadora,
    ratio_ok, es_pequeno, RATIO_SIN_ACCION,
)
from balance.diagnostico import (
    medir_tamano, medir_desbalance, medir_features,
    cargar_plan_si_existe, imprimir as imprimir_diagnostico,
)

BALANCE_DIR   = Path(__file__).parent
BALANCE_CACHE = BALANCE_DIR / ".balance_timing_cache.json"

ARCHIVO_SALIDA = "dataset.parquet"
ARCHIVO_INFORME  = BAL_REPORTS / "informe_balanceo.txt"
ARCHIVO_METRICAS = BAL_REPORTS / "comparativo.csv"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Lectura del EDA ─────────────────────────────────────────────────────────────

def leer_indicadores_eda() -> dict:
    """Devuelve los indicadores del EDA en un solo dict."""
    ind = {}

    # Plan de limpieza
    plan = cargar_plan_si_existe()
    if plan is not None:
        ind["plan_existe"]    = True
        ind["plan_columnas"]  = int(len(plan))
        ind["plan_eliminar"]  = int(
            (plan["accion"] == "eliminar_columna").sum()
            if "accion" in plan.columns else 0
        )
    else:
        ind["plan_existe"] = False

    # Tabla de balanceo del EDA
    if BALANCEO_CSV.exists():
        try:
            df_bal = pd.read_csv(BALANCEO_CSV)
            ind["balanceo_existe"] = True
            ind["balanceo_clases"]  = int(df_bal["clase"].nunique())
            ind["balanceo_ratio"]   = float(df_bal["count"].max() / max(df_bal["count"].min(), 1))
        except Exception:
            ind["balanceo_existe"] = False
    else:
        ind["balanceo_existe"] = False

    return ind


# ── Diagnostico (PASO 0) ────────────────────────────────────────────────────────

def ejecutar_diagnostico() -> dict:
    """Ejecuta s00_diagnostico.py por subprocess y devuelve el cache."""
    script = BALANCE_DIR / "diagnostico.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(f"\n  [AVISO] Diagnostico fallo (codigo {result.returncode}).")
        if result.stderr:
            print(f"  {result.stderr.strip()[:300]}")

    # Leer el cache generado
    if CACHE_DIAGNOSTICO.exists():
        try:
            return json.loads(CACHE_DIAGNOSTICO.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def mostrar_diagnostico_resumido(cache: dict) -> None:
    """Imprime un resumen del diagnostico antes de la recomendacion."""
    n_f   = cache.get("n_filas", 0)
    pct_c = cache.get("pct_categoricas", 0.0)
    ratio = cache.get("ratio", 0.0)
    cl_pct= cache.get("clase_min_pct", 0.0)
    card  = cache.get("cardinalidad_promedio", 0.0)

    tam_str  = "chico"     if n_f  < 30_000 else "grande" if n_f >= 300_000 else "mediano"
    cat_str  = f"{pct_c*100:.0f}%"
    cl_str   = f"{cl_pct*100:.2f}%"
    card_str = f"card.{card:.0f}"

    print(f"\n   Resumen: {n_f:,} filas | ratio {ratio:.1f}x | "
          f"categoricas {cat_str} | clase min {cl_str} | {card_str}")
    if not ratio_ok(ratio):
        print(f"   -> Desbalance detectado. Seleccionando tecnica...")


# ── Presentacion de la tecnica ganadora ───────────────────────────────────────

def imprimir_tecnica_ganadora(seleccion: dict) -> None:
    """Muestra al usuario la tecnica ganadora con su justificacion."""
    if not seleccion or not seleccion.get("ganadora"):
        print("\n  [AVISO] No se pudo determinar una tecnica ganadora.")
        return

    print(f"\n  TECNICA RECOMENDADA")
    print(f"  {'─' * 56}")
    print(f"  -> {seleccion['tecnica']}")
    print(f"     {seleccion['justificacion']}")
    if seleccion.get("alternativas"):
        print(f"\n  Alternativas: {', '.join(TECNICAS[k]['label'] for k in seleccion['alternativas'])}")
    print(f"  {'─' * 56}")


# ── Submenu de seleccion ────────────────────────────────────────────────────────

def submenu_tecnica(seleccion: dict, recomendadas: list) -> list:
    """
    Submenu:
      1. Ejecutar la tecnica ganadora   ← recomendado
      2. Ver todas las tecnicas disponibles
      3. Ejecutar todas las recomendadas
      4. Salir sin ejecutar
    """
    if not seleccion or not seleccion.get("ganadora"):
        return []

    ganadora = seleccion["ganadora"]

    while True:
        print(f"\n  1. Ejecutar {seleccion['tecnica']}   (recomendado)")
        print("  2. Elegir otra tecnia")
        print("  3. Ejecutar todas las recomendadas")
        print("  4. Salir sin ejecutar")
        print(f"  {'─' * 56}")

        eleccion = input("\n  Seleccione opcion (1–4): ").strip()

        if eleccion == "1":
            return [ganadora]

        elif eleccion == "2":
            return _submenu_todas(recomendadas)

        elif eleccion == "3":
            return recomendadas

        elif eleccion == "4":
            return []

        else:
            print("  [AVISO] Opcion no valida. Intente 1, 2, 3 o 4.")


def _submenu_todas(recomendadas: list) -> list:
    """Muestra el listado completo y devuelve la(s) elegida(s)."""
    print("\n  TECNICAS DISPONIBLES")
    print(f"  {'─' * 56}")
    opciones = {str(i + 1): k for i, k in enumerate(recomendadas)}

    for num, clave in opciones.items():
        marca = "  (recomendada)" if clave in recomendadas else ""
        print(f"  {num}. {TECNICAS[clave]['label']:<30}{marca}")

    print(f"\n  0. Volver")
    print(f"  {'─' * 56}")

    while True:
        eleccion = input("\n  Seleccione una tecnica (0 para volver): ").strip()
        if eleccion == "0":
            return []
        if eleccion in opciones:
            return [opciones[eleccion]]
        print("  [AVISO] Opcion no valida.")


# ── Ejecucion de una tecnica ────────────────────────────────────────────────────

def ejecutar_tecnica(clave: str) -> dict:
    """Ejecuta el script de tecnica por subprocess y devuelve metricas."""
    info    = TECNICAS[clave]
    script  = info["script"]
    salida  = BALANCED_DIR / clave / ARCHIVO_SALIDA
    salida.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script,
         "--input",  str(CONSOLIDADO),
         "--output", str(salida)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    duracion = round(time.time() - t0, 1)

    metricas = {"tiempo_s": duracion}

    if result.returncode == 0:
        # Extraer la linea JSON del final del stdout
        for linea in reversed(result.stdout.strip().splitlines()):
            linea = linea.strip()
            if linea.startswith("{") and linea.endswith("}"):
                try:
                    metricas.update(json.loads(linea))
                except json.JSONDecodeError:
                    pass
                break

        if salida.exists():
            try:
                df_out = pd.read_parquet(salida)
                metricas.setdefault("filas_resultado", int(len(df_out)))
            except Exception:
                metricas["filas_resultado"] = "N/D"
    else:
        print(f"\n  [ERROR] {script} fallo (codigo {result.returncode})")
        if result.stderr:
            print(f"  {result.stderr.strip()[:400]}")
        metricas["filas_resultado"] = 0

    return metricas


# ── Guardado de salidas ─────────────────────────────────────────────────────────

def guardar_metricas(metricas: dict):
    filas = []
    for clave, m in metricas.items():
        filas.append({
            "tecnica":          TECNICAS[clave]["label"],
            "familia":          TECNICAS[clave]["familia"],
            "filas_originales": m.get("filas_originales", 0),
            "filas_resultado":  m.get("filas_resultado",  0),
            "tiempo_s":         m.get("tiempo_s",          0.0),
            "archivo":          str(BALANCED_DIR / clave / ARCHIVO_SALIDA),
        })
    if filas:
        pd.DataFrame(filas).to_csv(ARCHIVO_METRICAS, index=False)
        print(f"  [OK] Comparativo: {ARCHIVO_METRICAS}")


def guardar_cache(metricas: dict):
    """Guarda en cache los tiempos de ejecución y metadatos del balanceo."""
    data = {}
    for k, v in metricas.items():
        data[k] = {
            "tiempo_s": v.get("tiempo_s", 0.0),
            "filas_resultado": v.get("filas_resultado", 0)
        }
    try:
        BALANCE_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def guardar_informe(diag: dict, recomendadas: list,
                     ejecutadas: dict, errores: list) -> None:
    lineas = [
        "=" * 60,
        f"  INFORME DE BALANCEO — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
        "",
        "DIAGNOSTICO",
        f"  Ratio de desbalance  : {diag.get('ratio', '?')}x",
        f"  Clase mayoritaria    : {diag.get('clase_mayoritaria', '?')}",
        f"  Clase minoritaria    : {diag.get('clase_minoritaria', '?')}",
        "",
    ]
    for clase, cnt in sorted(diag.get("conteos", {}).items()):
        tot  = diag.get("filas_totales", 1)
        pct  = cnt / tot * 100 if tot else 0
        lineas.append(f"  Clase {clase}: {cnt:>8,}  ({pct:>5.1f}%)")

    lineas += ["", "TECNICAS RECOMENDADAS"]
    for clave in recomendadas:
        lineas.append(f"  * {TECNICAS[clave]['label']}")

    lineas += ["", "EJECUCION"]
    for clave, m in ejecutadas.items():
        lineas.append(f"  {TECNICAS[clave]['label']:<32}  {m.get('tiempo_s', 0):>6.1f}s")

    if errores:
        lineas += ["", "ERRORES"]
        for e in errores:
            lineas.append(f"  * {e}")

    lineas += ["=" * 60, ""]
    ARCHIVO_INFORME.write_text("\n".join(lineas), encoding="utf-8")
    print(f"  [LOG] Informe: {ARCHIVO_INFORME}")


# ── Pipeline principal ──────────────────────────────────────────────────────────

def run(tecnicas: list | None = None, forzar_submenu: bool = True) -> dict:
    """
    Ejecuta el pipeline completo de balanceo.

    Parametros
    ----------
    tecnicas : list | None
        Lista de claves de TECNICAS.  Si es None y forzar_submenu=True
        pregunta al usuario.  Si es lista vacia, no ejecuta nada.
    forzar_submenu : bool
        Si True y tecnicas es None, muestra el submenu interactivo.
    """
    print("=" * 60)
    print("  BALANCEO DE DATOS  —  FAERS Q4 2025")
    print("=" * 60)

    # ── PASO 0: Diagnóstico ────────────────────────────────────────────────────
    print("\n[PASO 0] Diagnosticando datos...")
    cache_diag   = ejecutar_diagnostico()
    mostrar_diagnostico_resumido(cache_diag)

    # Cargar dataset
    try:
        df  = pd.read_parquet(CONSOLIDADO)
        print(f"  Dataset: {len(df):,} filas x {df.shape[1]} columnas")
    except Exception as exc:
        print(f"\n  [ERROR] No se pudo cargar el consolidado: {exc}")
        return {"diagnostico": {}, "accion": "error", "errores": [str(exc)]}

    # Paso 0b: desbalance desde el propio dataset
    if TARGET_COL not in df.columns:
        print(f"\n  [ERROR] La columna objetivo '{TARGET_COL}' no existe.")
        return {"diagnostico": {}, "accion": "error", "errores": ["target no encontrado"]}

    conteos  = df[TARGET_COL].value_counts().sort_index()
    ratio    = float(conteos.max() / max(conteos.min(), 1))
    cl_may   = int(conteos.idxmax())
    cl_min   = int(conteos.idxmin())
    cl_min_pct = float(conteos.min() / conteos.sum())
    n_filas    = len(df)

    # Paso 0c: indicadores desde cache del diagnostico
    pct_categoricas  = float(cache_diag.get("pct_categoricas", 0.0))
    cardinalidad_prom = float(cache_diag.get("cardinalidad_promedio", 0.0))

    # ── Verificar si es necesario balancear ────────────────────────────────────
    if ratio_ok(ratio):
        print(f"\n  Ratio {ratio:.1f}x <= {RATIO_SIN_ACCION}x. Balance aceptable, no se requiere accion.")
        return {"diagnostico": {"ratio": ratio}, "accion": "no_necesario",
                "recomendadas": [], "ejecutadas": {}, "errores": []}

    print(f"\n  Ratio de desbalance: {ratio:.1f}x  "
          f"(clase {cl_may} mayoritaria, clase {cl_min} minoritaria)")

    # ── PASO 1: Seleccion de tecnicas ──────────────────────────────────────────
    recomendadas = recomendar_tecnicas(
        ratio, n_filas, pct_categoricas, cl_min_pct, cardinalidad_prom,
    )

    print(f"\n[PASO 1] Tecnicas recomendadas:")
    for clave in recomendadas:
        print(f"    * {TECNICAS[clave]['label']}")

    if not recomendadas:
        print("  [AVISO] No hay tecnicas recomendadas. No se ejecutara nada.")
        return {"diagnostico": {"ratio": ratio}, "accion": "ninguna",
                "recomendadas": [], "ejecutadas": {}, "errores": []}

    # Elegir ganadora
    seleccion = elegir_tecnica_ganadora(
        ratio, n_filas, pct_categoricas, cl_min_pct, cardinalidad_prom,
    )
    imprimir_tecnica_ganadora(seleccion)

    # ── PASO 2: Submenu ────────────────────────────────────────────────────────
    if tecnicas is None and forzar_submenu:
        tecnicas = submenu_tecnica(seleccion, recomendadas)

    if not tecnicas:
        print("\n  Sin tecnicas seleccionadas. Cancelando.")
        return {"diagnostico": {"ratio": ratio}, "accion": "cancelado",
                "recomendadas": recomendadas, "ejecutadas": {}, "errores": []}

    confirmar = [t for t in tecnicas if t in TECNICAS]
    print(f"\n[PASO 2] Ejecutando {len(confirmar)} tecnica(s)...")

    # ── PASO 3: Ejecucion ──────────────────────────────────────────────────────
    metricas: dict = {}
    errores:  list = []

    for clave in confirmar:
        label = TECNICAS[clave]["label"]
        print(f"\n  Ejecutando: {label}...")
        m = ejecutar_tecnica(clave)
        metricas[clave] = m

        filas_out = m.get("filas_resultado", 0)
        if filas_out not in (0, "N/D", None):
            print(f"  [OK] {label}: {n_filas:,} -> {filas_out:,} filas "
                  f"({m['tiempo_s']}s)")
        else:
            errores.append(f"{label}: no produjo salida valida")

    # ── PASO 4: Resumen final ──────────────────────────────────────────────────
    guardar_metricas(metricas)
    guardar_cache(metricas)

    diag = {"ratio": ratio, "clase_mayoritaria": cl_may,
            "clase_minoritaria": cl_min, "conteos": conteos.to_dict(),
            "filas_totales": n_filas}
    guardar_informe(diag, recomendadas, metricas, errores)

    print(f"\n  {chr(61)*60}")
    if errores:
        print(f"  BALANCEO CON {len(errores)} ADVERTENCIA(S)")
    else:
        print("  BALANCEO COMPLETADO")
    print(f"  Datos balanceados en: {BALANCED_DIR}")
    print(f"  Comparativo: {ARCHIVO_METRICAS.name}")
    print(f"{chr(61)*60}")

    return {
        "diagnostico":  diag,
        "recomendadas": recomendadas,
        "ejecutadas":   confirmar,
        "metricas":     metricas,
        "errores":      errores,
    }


# ── Main independiente ─────────────────────────────────────────────────────────

def main():
    run(tecnicas=None, forzar_submenu=True)
    input("\n  Presione Enter para volver...")


if __name__ == "__main__":
    main()
