"""
Script 00: Ejecución Completa del EDA – FAERS Q4 2025
Este script ejecuta todos los análisis en secuencia.
"""
import subprocess
import sys
from pathlib import Path

# Añadir directorio raíz al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

scripts = [
    "01_load_and_overview.py",
    "02_demographic_analysis.py",
    "03_drug_analysis.py",
    "04_reaction_analysis.py",
    "05_outcome_analysis.py",
    "06_temporal_analysis.py",
    "07_cross_analysis.py",
    "08_summary_report.py",
]

def main():
    print("=" * 70)
    print("  EDA COMPLETO – FAERS Q4 2025")
    print("=" * 70)
    print(f"\n  Directorio de datos  : {PROJECT_ROOT / 'data'}")
    print(f"  Directorio de salida: {PROJECT_ROOT / 'outputs' / 'eda_results'}")
    print("\n" + "=" * 70)

    for i, script in enumerate(scripts, 1):
        script_path = Path(__file__).parent / script
        if not script_path.exists():
            print(f"\n[ADVERTENCIA] Script no encontrado: {script}")
            continue

        print(f"\n[{i:02d}/{len(scripts)}] Ejecutando: {script}")
        print("-" * 70)

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            if result.returncode != 0:
                print(f"[ERROR] Script falló con código {result.returncode}")
                return
        except Exception as e:
            print(f"[ERROR] No se pudo ejecutar {script}: {e}")
            return

    print("\n" + "=" * 70)
    print("  EDA COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n  Todos los gráficos y tablas guardados en:")
    print(f"  {PROJECT_ROOT / 'outputs' / 'eda_results'}")
    print("\n  Archivos generados:")
    for file in sorted((PROJECT_ROOT / "outputs" / "eda_results").glob("*")):
        print(f"    • {file.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
