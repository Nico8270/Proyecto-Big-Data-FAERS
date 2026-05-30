"""
src/cleaning/cleaning_main.py
================================
Orquestador del pipeline de limpieza — versión silenciosa, sin barra.

La barra de progreso la maneja src/main.py.  Este archivo solo ejecuta
los s0X_*.py secuencialmente y escribe el log de resultados.

Uso independiente (para testing):
    python src/cleaning/cleaning_main.py
"""

import shutil
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CLEANING_DIR = Path(__file__).parent
EDA_OUTPUTS   = PROJECT_ROOT / "outputs" / "eda_results"
CLEAN_DATA    = PROJECT_ROOT / "data" / "clean_data"
CLEANING_CACHE = CLEANING_DIR / ".cleaning_timing_cache.json"

SCRIPTS = [
    "s01_limpiar_demo.py",
    "s02_limpiar_drug.py",
    "s03_limpiar_reac.py",
    "s04_limpiar_indications.py",
    "s05_limpiar_outcomes.py",
    "s06_limpiar_therapy.py",
    "s07_limpiar_reporter.py",
    "s08_consolidar_clean.py",
]


# ─── Utilidades ────────────────────────────────────────────────────────────────

def limpiar_salida():
    """Borra clean_data para empezar de cero."""
    if CLEAN_DATA.exists():
        shutil.rmtree(CLEAN_DATA)
    CLEAN_DATA.mkdir(parents=True, exist_ok=True)


def cargar_plan():
    """Lee el CSV de plan de limpieza del EDA.  Si no existe, devuelve None."""
    plan_path = EDA_OUTPUTS / "s09_resumen_completo.csv"
    if not plan_path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(plan_path)
    except Exception:
        return None


def escribir_log(tiempos: dict, errores: list):
    path    = CLEANING_DIR / "logs" / f"cleaning_execution_{datetime.now():%Y-%m-%d}.txt"
    total   = sum(tiempos.values()) if tiempos else 0
    estados = ["OK" if e is None else f"FALLO: {e}" for e in errores]

    lineas = [
        "=" * 60,
        f"  LIMPIEZA — {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 60,
    ]
    for script, t in zip(SCRIPTS, [tiempos.get(s, 0) for s in SCRIPTS]):
        idx  = SCRIPTS.index(script)
        pct  = t / total * 100 if total else 0
        lineas.append(f"  {script:<48}  {estados[idx]:<8}  {t:>6.1f}s  ({pct:>5.1f}%)")
    lineas += [
        "",
        f"  {'TOTAL':<48}       {'':8}  {total:>6.1f}s",
        "=" * 60,
    ]
    path.write_text("\n".join(lineas), encoding="utf-8")
    return path


def guardar_cache(tiempos: dict):
    data = {k: round(v, 1) for k, v in tiempos.items()}
    data["last_run_total"] = round(sum(tiempos.values()), 1) if tiempos else 0
    try:
        CLEANING_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def cargar_cache() -> float:
    if CLEANING_CACHE.exists():
        try:
            data = json.loads(CLEANING_CACHE.read_text(encoding="utf-8"))
            return float(data.get("last_run_total", 60))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return 60      # estimado genérico: 60 s


# ─── Función principal (sin barra, solo retorna el resultado) ──────────────────

def run() -> tuple[dict, list]:
    """
    Ejecuta todos los scripts de limpieza en orden.

    Returns
    -------
    tiempos : dict  {nombre_script: duracion_segundos}
    errores : list  [None si OK, str con el error si falló]
    """
    limpiar_salida()
    plan = cargar_plan()

    if plan is None:
        print("  [AVISO] No se encontró el plan de limpieza del EDA "
              "(outputs/eda_results/s09_resumen_completo.csv).")
        print("  Se ejecutarán TODOS los scripts de limpieza.\n")

    tiempos  = {}
    errores  = []

    for i, script in enumerate(SCRIPTS, 1):
        script_path = CLEANING_DIR / script
        if not script_path.exists():
            print(f"  [ERROR] Script no encontrado: {script_path}")
            tiempos[script] = 0
            errores.append(f"No encontrado: {script}")
            continue

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        dur = round(time.time() - t0, 1)
        tiempos[script] = dur

        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            if result.stderr:
                print("\n" + "!" * 80)
                print(f"  [ERROR EN SCRIPT] {script} falló con el siguiente error detallado:")
                print("!" * 80)
                print(result.stderr.rstrip())
                print("!" * 80 + "\n")
            err = result.stderr.strip()[:300] if result.stderr else "sin detalle"
            errores.append(f"código {result.returncode} — {err}")

    guardar_cache(tiempos)
    return tiempos, errores


# ─── Main independiente ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  LIMPIEZA DE DATOS — FAERS Q4 2025")
    print("=" * 60)
    print(f"\n  Leyendo plan de limpieza...\n")

    tiempos, errores = run()

    print(f"\n{'=' * 60}")
    if not errores:
        print("  LIMPIEZA COMPLETADA")
    else:
        print(f"  LIMPIEZA COMPLETADA CON {len(errores)} ERROR(ES)")
    print(f"{'=' * 60}")
    print(f"\n  Datos limpios en: {CLEAN_DATA}")
    for f in sorted(CLEAN_DATA.iterdir()):
        if f.is_file():
            print(f"    {f.name:<50}  {f.stat().st_size / 1024:.0f} KB")
    if errores:
        print("\n  Errores:")
        for e in errores:
            print(f"    • {e}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
