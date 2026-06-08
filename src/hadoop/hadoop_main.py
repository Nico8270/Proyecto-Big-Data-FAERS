"""
src/hadoop/hadoop_main.py
===========================
Orquestador del pipeline de Hadoop MapReduce.

# Pipeline: local batch processing, architected for horizontal scalability.
# To deploy on EMR: change runner to 'emr' in mrjob.conf and set up AWS credentials.

Secuencia:
1. Prepara las entradas convirtiendo los archivos parquet a JSON Lines tagged
   (ejecuta src/hadoop/s01_prepare_inputs.py).
2. Ejecuta el Job de MapReduce (src/hadoop/s02_mapreduce_join.py) localmente,
   guardando los resultados en la carpeta temporal data/staging/hadoop_join_output/.
3. Postprocesa el resultado de MapReduce convirtiendo lo a parquet y limpiando
   directorios temporales (ejecuta src/hadoop/s03_postprocess_output.py).

Argumentos CLI opcionales:
  --sample N   Limitar cada tabla a N filas en s01 (modo prueba).
"""

import argparse
import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "hadoop"))
import s01_prepare_inputs as s01
STAGING_INPUT = PROJECT_ROOT / "data" / "staging" / "hadoop_join_input"
STAGING_OUTPUT = PROJECT_ROOT / "data" / "staging" / "hadoop_join_output"

def run_script(script_name: str) -> bool:
    """Ejecuta un script de Python como subproceso con salida en vivo."""
    script_path = PROJECT_ROOT / "src" / "hadoop" / script_name
    print(f"\n>>> Ejecutando {script_name}...")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    duracion = time.time() - t0
    
    if result.returncode == 0:
        print(f">>> {script_name} completado con éxito en {duracion:.1f} segundos.")
        return True
    else:
        print(f"\n[ERROR] {script_name} falló con código de salida {result.returncode}.")
        return False

def main(sample_size: int | None = None):
    modo = f"PRUEBA ({sample_size:,} filas/tabla)" if sample_size else "COMPLETO"
    print("=" * 60)
    print(f"  ORQUESTADOR DE JOIN HADOOP MAPREDUCE")
    print(f"  Modo: {modo}")
    print("=" * 60)

    t_start = time.time()

    # 1. Preparar entradas (con o sin muestra)
    print(f"\n>>> Ejecutando s01_prepare_inputs...")
    t0 = time.time()
    try:
        s01.main(sample_size=sample_size)
    except SystemExit as e:
        if e.code != 0:
            print("\n[FALLO] Abortando el pipeline en la etapa de preparación de entradas.")
            sys.exit(1)
    duracion = time.time() - t0
    print(f">>> s01_prepare_inputs completado en {duracion:.1f} segundos.")

    # Verificar que se crearon los archivos jsonl
    if not STAGING_INPUT.exists() or not any(STAGING_INPUT.iterdir()):
        print("\n[ERROR] No se encontraron archivos preparados en staging para el join. Abortando.")
        sys.exit(1)

    # Asegurar que el directorio de salida de MapReduce no existe
    if STAGING_OUTPUT.exists():
        try:
            shutil.rmtree(STAGING_OUTPUT)
        except OSError as e:
            print(f"[AVISO] No se pudo limpiar {STAGING_OUTPUT}: {e}")

    # 2. Ejecutar Job de MapReduce
    print("\n>>> Ejecutando Job de MapReduce (MRJoinJob) en modo local...")
    t0 = time.time()

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "src" / "hadoop" / "s02_mapreduce_join.py"),
        "--output-dir",
        str(STAGING_OUTPUT),
        str(STAGING_INPUT)
    ]

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    dur = time.time() - t0

    if result.returncode != 0:
        print(f"\n[ERROR] El Job de MapReduce falló con código {result.returncode}.")
        if STAGING_INPUT.exists():
            shutil.rmtree(STAGING_INPUT)
        sys.exit(1)

    print(f"\n>>> Job de MapReduce completado con éxito en {dur:.1f} segundos.")

    # 3. Postprocesar salidas
    if not run_script("s03_postprocess_output.py"):
        print("\n[FALLO] Abortando el pipeline en la etapa de postprocesamiento.")
        sys.exit(1)

    # 4. Calcular señales de seguridad estadísticas
    if not run_script("s04_safety_signals.py"):
        print("\n[FALLO] Abortando el pipeline en la etapa de cálculo de señales.")
        sys.exit(1)

    total_dur = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  PIPELINE HADOOP MAPREDUCE COMPLETADO CON ÉXITO")
    print(f"  Modo: {modo}")
    print(f"  Duración total: {total_dur:.1f} segundos")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Limitar cada tabla a N filas (modo prueba)")
    args = parser.parse_args()
    main(sample_size=args.sample)
