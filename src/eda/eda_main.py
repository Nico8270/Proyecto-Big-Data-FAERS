"""
eda_main.py
===========
Orquestador principal del EDA FAERS.

Responsabilidades:
  1. Limpia outputs/eda_results/ antes de cada ejecucion.
  2. Ejecuta cada script en orden numerico estricto mediante subprocess.
  3. Guarda un log de ejecucion con tiempos.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eda.config_eda import DATA_DIR, OUTPUT_DIR, LOG_FILE

# Orden de ejecucion estricto
SCRIPTS = [
    "s01_carga_y_overview.py",
    "s02_demograficos.py",
    "s03_drogas.py",
    "s04_reacciones.py",
    "s05_outcomes.py",
    "s06_temporal.py",
    "s07_cruzado.py",
    "s08_duplicados.py",
    "s09_resumen.py",
]

TIMES = {}


def limpiar_outputs():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] Outputs limpiados: {OUTPUT_DIR}")


def ejecutar_script(nombre: str, idx: int, total: int) -> bool:
    script_rel = "src/eda/" + nombre
    inicio = time.time()

    print(f"\n[{idx:02d}/{total}] {nombre}")
    print("─" * 70)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script_rel)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(f"  [STDERR]\n{result.stderr.rstrip()}")

    elapsed  = time.time() - inicio
    ok       = result.returncode == 0
    TIMES[nombre] = elapsed

    print(f"\n  [{'OK' if ok else 'FALLO'}] {elapsed:.1f}s  codigo={result.returncode}")
    return ok


def escribir_log(todo_ok: bool):
    total = sum(TIMES.values())
    lineas = [
        "=" * 70,
        f"  EDA FAERS Q4 2025    {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 70,
    ]
    for nombre, t in TIMES.items():
        pct = t / total * 100 if total else 0
        lineas.append(f"  {nombre:<44} {t:>7.1f}s  ({pct:>5.1f}%)")
    lineas += [
        "",
        f"  {'TOTAL':<44} {total:>7.1f}s",
        "=" * 70,
        f"  Estado: {'EXITOSO' if todo_ok else 'CON ERRORES'}",
        "=" * 70,
    ]
    LOG_FILE.write_text("\n".join(lineas), encoding="utf-8")


def listar_outputs():
    print(f"\n{'=' * 70}")
    print(f"  Archivos generados en outputs/eda_results/")
    print(f"{'=' * 70}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        tam = f"{f.stat().st_size / 1024:.1f} KB" if f.is_file() else "<dir>"
        print(f"    {f.name:<60} {tam}")
    print(f"{'=' * 70}")


def main():
    print("=" * 70)
    print("  EDA FAERS Q4 2025 — ORQUESTADOR")
    print("=" * 70)
    print(f"\n  Dataset  : {DATA_DIR}")
    print(f"  Outputs  : {OUTPUT_DIR}")
    print(f"  Scripts  : {len(SCRIPTS)}")
    print()

    limpiar_outputs()

    todo_ok = True
    for i, script in enumerate(SCRIPTS, 1):
        if not ejecutar_script(script, i, len(SCRIPTS)):
            todo_ok = False
            print(f"\n  [ABORTANDO] Script fallido: {script}")
            break

    escribir_log(todo_ok)
    listar_outputs()

    print(f"\n{'=' * 70}")
    print("  ESTADO FINAL:", "EXITOSO" if todo_ok else "CON ERRORES")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
