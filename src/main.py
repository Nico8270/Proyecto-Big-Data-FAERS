# Punto de entrada principal -- menu interactivo + orquestación.


import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ─── Configuración de Java local portable si existe ───────────────────────────
JDK_LOCAL = PROJECT_ROOT / "vendor" / "jdk"
if JDK_LOCAL.exists():
    os.environ["JAVA_HOME"] = str(JDK_LOCAL)
    os.environ["PATH"] = str(JDK_LOCAL / "bin") + os.path.pathsep + os.environ.get("PATH", "")

from utils.bar import ProgressBar

# ─── Rutas ────────────────────────────────────────────────────────────────────

CONFIG_CACHE  = PROJECT_ROOT / "src" / "eda" / ".eda_timing_cache.json"
EDA_SCRIPTS   = [
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
EDA_DIR       = PROJECT_ROOT / "src" / "eda"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"
EDA_OUTPUTS   = OUTPUTS_DIR / "eda_results"

# ─── Menu ─────────────────────────────────────────────────────────────────────

MENU = """
╔═══════════════════════════════════════════════════════════════╗
║       FAERS PHARMACOVIGILANCE ANALYTICS PLATFORM              ║
╠═══════════════════════════════════════════════════════════════╣
║  0. Configurar entorno y descargar tecnologias                ║
║  1. Analisis exploratorio (EDA)                               ║
║  2. Limpiar datos                                             ║
║  3. Join con Hadoop y MapReduce                               ║
║  4. Balancear datos                                           ║
║  5. Entrenar modelo con Hadoop MapReduce                      ║
║  6. Correr modelo con Kafka                                   ║
║  7. Consultar modelo con caso clínico (Predicción ML)         ║
║  8. Salir                                                     ║
╚═══════════════════════════════════════════════════════════════╝
"""


# ─── Opción 0 -- Configuración de entorno ──────────────────────────────────────────────────

SEP = "─" * 63

def configurar_entorno() -> None:
    """Instala todas las dependencias del proyecto y verifica la estructura de carpetas."""

    print(f"\n  [0] CONFIGURACIÓN DE ENTORNO Y DEPENDENCIAS")
    print(f"  {SEP}\n")

    # ── Paso 1: Python ──────────────────────────────────────────────────────────
    res = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
    version = (res.stdout or res.stderr or "").strip()
    print(f"  [INFO] Python detectado  →  {version}\n")

    # ── Paso 2: Actualizar pip ──────────────────────────────────────────────────
    print(f"  {SEP}")
    print("  [1/4] Actualizando pip...")
    print(f"  {SEP}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        text=True,
    )

    # ── Paso 3: Verificar o instalar JDK de Java portable ───────────────────────
    print(f"\n  {SEP}")
    print("  [2/4] Verificando / Descargando JDK de Java portable...")
    print(f"  {SEP}")

    java_encontrado = False
    try:
        res_java = subprocess.run(["java", "-version"], capture_output=True, text=True)
        # Si tiene java y no es el local, lo usamos directamente
        if res_java.returncode == 0:
            java_encontrado = True
            print("  [INFO] Java ya está disponible en el sistema a nivel global.")
    except FileNotFoundError:
        pass

    if not java_encontrado:
        import urllib.request
        import zipfile
        
        jdk_dir = PROJECT_ROOT / "vendor" / "jdk"
        if jdk_dir.exists():
            print("  [INFO] JDK portable ya está descargado y configurado en vendor/jdk/")
            os.environ["JAVA_HOME"] = str(jdk_dir)
            os.environ["PATH"] = str(jdk_dir / "bin") + os.path.pathsep + os.environ.get("PATH", "")
        else:
            url = "https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse"
            zip_path = PROJECT_ROOT / "vendor" / "openjdk.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            
            print("  [Descarga] Descargando JDK 17 Portable de Eclipse Adoptium...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                    total_size = int(response.info().get('Content-Length', 0))
                    block_size = 1024 * 1024
                    descargado = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        descargado += len(buffer)
                        out_file.write(buffer)
                        if total_size:
                            pct = descargado / total_size * 100
                            sys.stdout.write(f"\r             Progreso: {pct:.1f}% ({descargado / (1024*1024):.1f} MB de {total_size / (1024*1024):.1f} MB)")
                            sys.stdout.flush()
                print("\n             Descarga completa.")
                
                print("  [Extracción] Descomprimiendo JDK...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    temp_extract = PROJECT_ROOT / "vendor" / "temp_jdk"
                    zip_ref.extractall(temp_extract)
                    carpetas_internas = list(temp_extract.iterdir())
                    if carpetas_internas:
                        shutil.move(str(carpetas_internas[0]), str(jdk_dir))
                    shutil.rmtree(temp_extract)
                zip_path.unlink()
                print("  [OK] JDK portable instalado con éxito en vendor/jdk/")
                os.environ["JAVA_HOME"] = str(jdk_dir)
                os.environ["PATH"] = str(jdk_dir / "bin") + os.path.pathsep + os.environ.get("PATH", "")
            except Exception as e:
                print(f"\n  [ERROR] Falló la descarga/extracción del JDK portable: {e}")
                if zip_path.exists():
                    zip_path.unlink()
    else:
        print("  [INFO] Omitiendo descarga del JDK portable.")

    # ── Paso 4: Instalar TODAS las dependencias ─────────────────────────────────
    req = PROJECT_ROOT / "requirements.txt"
    print(f"\n  {SEP}")
    print(f"  [3/4] Instalando dependencias del proyecto ({req.name})...")
    print( "        pandas · numpy · matplotlib · seaborn · pyarrow")
    print( "        scikit-learn · imbalanced-learn · mrjob · kafka-python · pyspark")
    print(f"  {SEP}")
    if req.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req)],
            text=True,
            cwd=str(PROJECT_ROOT),
        )
    else:
        print(f"  [ERROR] No se encontró {req}. Abortando instalación.")

    # ── Paso 4: Verificar estructura de carpetas ──────────────────────────────
    print(f"\n  {SEP}")
    print("  [3/4] Verificando estructura de carpetas del proyecto...")
    print(f"  {SEP}")
    carpetas = [
        PROJECT_ROOT / "data" / "raw" / "faers",
        PROJECT_ROOT / "data" / "staging",
        PROJECT_ROOT / "data" / "joined",
        PROJECT_ROOT / "data" / "aggregated",
        PROJECT_ROOT / "data" / "streaming_raw",
        PROJECT_ROOT / "data" / "alerts",
        PROJECT_ROOT / "data" / "balanced_data",
        PROJECT_ROOT / "data" / "clean_data",
        PROJECT_ROOT / "src" / "eda",
        PROJECT_ROOT / "src" / "balance" / "logs",
        PROJECT_ROOT / "outputs" / "eda_results",
        PROJECT_ROOT / "outputs" / "mapreduce_results",
        PROJECT_ROOT / "outputs" / "balance_reports",
    ]
    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"    [OK] {carpeta.relative_to(PROJECT_ROOT)}")

    # ── Paso 5: Limpiar outputs previos ───────────────────────────────────────
    print(f"\n  {SEP}")
    print("  [4/4] Limpiando outputs previos...")
    print(f"  {SEP}")
    for carpeta in [
        OUTPUTS_DIR / "eda_results",
        OUTPUTS_DIR / "mapreduce_results",
        OUTPUTS_DIR / "balance_reports",
    ]:
        if carpeta.exists():
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        print(f"    [OK] {carpeta.relative_to(PROJECT_ROOT)} limpiada")

    print(f"\n  {SEP}")
    print("  [OK] Entorno configurado correctamente.")
    print(f"  {SEP}\n")
    input("  Presione Enter para volver al menu...")


def _run_silent(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Ejecuta un comando en silencio (usado por EDA y otros módulos)."""
    return subprocess.run(cmd, capture_output=capture, text=True)


# ─── Opción 1 -- EDA con barra de progreso ────────────────────────────────────

# Descripción de cada archivo generado por el EDA
_EDA_FILE_DESC: dict[str, str] = {
    "s01_demo_columnas.csv":          "Calidad de columnas de la tabla DEMO (demográficos)",
    "s01_drug_columnas.csv":          "Calidad de columnas de la tabla DRUG (fármacos)",
    "s01_indi_columnas.csv":          "Calidad de columnas de la tabla INDI (indicaciones)",
    "s01_outc_columnas.csv":          "Calidad de columnas de la tabla OUTC (outcomes)",
    "s01_reac_columnas.csv":          "Calidad de columnas de la tabla REAC (reacciones)",
    "s01_rpsr_columnas.csv":          "Calidad de columnas de la tabla RPSR (fuente reporte)",
    "s01_ther_columnas.csv":          "Calidad de columnas de la tabla THER (terapias)",
    "s01_nulos_por_tabla.png":        "Heatmap de porcentaje de nulos por tabla y columna",
    "s02_edad.png":                   "Distribución de edad de pacientes reportados",
    "s02_genero.png":                 "Distribución de género de los pacientes",
    "s02_paises.png":                 "Top países de origen de los reportes",
    "s02_boxplot_edad_genero.png":    "Boxplot de edad cruzado con género",
    "s02_estadisticas_edad_por_genero.csv": "Estadísticas de edad (media, std, cuartiles) por género",
    "s03_top25_farmacos.png":         "Top 25 fármacos más frecuentes en los reportes",
    "s03_top25_farmacos.csv":         "Tabla con conteo de los top 25 fármacos",
    "s03_roles_farmaco.png":          "Distribución de roles del fármaco (PS, SS, C)",
    "s03_estadisticas_roles.csv":     "Conteo y porcentaje de cada rol de fármaco",
    "s03_vias_administracion.png":    "Top vías de administración más reportadas",
    "s03_dosis.png":                  "Distribución de dosis numéricas reportadas",
    "s04_top25_reacciones.png":       "Top 25 términos de reacción adversa (PT) más frecuentes",
    "s04_top25_reacciones.csv":       "Tabla con conteo de las top 25 reacciones",
    "s04_reacciones_por_caso.png":    "Distribución de cantidad de reacciones por caso",
    "s04_estadisticas_por_caso.csv":  "Estadísticas de reacciones por caso (media, máx, etc.)",
    "s05_outcomes.png":               "Distribución de outcomes clínicos (muerte, hospitalización...)",
    "s05_conteos_outcomes.csv":       "Conteo y porcentaje de cada código de outcome",
    "s06_serie_mensual.png":          "Serie temporal de reportes por mes",
    "s06_heatmap_pais_mes.png":       "Heatmap de reportes por país y mes",
    "s06_serie_mensual_reportes.csv": "Datos numéricos de la serie temporal mensual",
    "s07_heatmap_drug_reac.png":      "Heatmap de co-ocurrencia fármaco x reacción (top 20)",
    "s07_top20_pares_farmaco_reaccion.csv": "Top 20 pares fármaco-reacción más frecuentes",
    "s08_duplicados_por_tabla.csv":   "Conteo de duplicados completos y por PK en cada tabla",
    "s09_resumen_completo.csv":       "Plan de limpieza: acción a tomar por cada columna",
    "s09_balanceo.csv":               "Distribución de clases de severidad + estrategia recomendada",
}


def ejecutar_eda() -> None:
    """Ejecuta el pipeline EDA completo con barra de progreso y ETA."""
    _limpiar_outputs()
    _cargar_cache()

    bar = ProgressBar(
        total     = len(EDA_SCRIPTS),
        label     = "EDA FAERS",
        bar_width = 20,
    )
    bar.start()

    tiempos        = {}
    error_ocurrido = False

    for i, script in enumerate(EDA_SCRIPTS, 1):
        script_path = EDA_DIR / script
        if not script_path.exists():
            print(f"\n  [ERROR] Script no encontrado: {script_path}")
            bar.next(label=f"[ERROR] {script}")
            error_ocurrido = True
            continue

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        duracion = round(time.time() - t0, 2)
        tiempos[script] = duracion

        if result.returncode != 0:
            print(f"\n  [ERROR] {script} falló (código {result.returncode})")
            if result.stderr:
                print(f"  {result.stderr.strip()[:300]}")
            bar.next(label=f"[FALLO] {script}")
            error_ocurrido = True
            continue

        bar.next(label=script)

    bar.finish()
    _guardar_cache(tiempos)

    estado = "CON ERRORES" if error_ocurrido else "COMPLETADO"
    print(f"\n  [OK] EDA {estado}.\n")
    _mostrar_resumen_eda()
    input("\n  Presione Enter para volver al menu...")


def _limpiar_outputs():
    if EDA_OUTPUTS.exists():
        shutil.rmtree(EDA_OUTPUTS)
    EDA_OUTPUTS.mkdir(parents=True, exist_ok=True)


def _mostrar_resumen_eda() -> None:
    """Muestra los archivos generados y tablas de resumen del EDA."""
    import csv
    from collections import Counter
    sep = "─" * 63

    # ── Archivos generados ────────────────────────────────────────────────
    print(f"  {sep}")
    print("  ARCHIVOS GENERADOS")
    print(f"  {sep}")
    archivos = sorted(EDA_OUTPUTS.iterdir()) if EDA_OUTPUTS.exists() else []
    for f in archivos:
        desc = _EDA_FILE_DESC.get(f.name, "Archivo de salida del EDA")
        print(f"  {f.name:<45}  {desc}")
    if not archivos:
        print("  (No se encontraron archivos en outputs/eda_results/)")
        return

    # ── Resumen duplicados (s08) ───────────────────────────────────────
    dups_csv = EDA_OUTPUTS / "s08_duplicados_por_tabla.csv"
    if dups_csv.exists():
        try:
            print(f"\n  {sep}")
            print("  DUPLICADOS POR TABLA")
            print(f"  {sep}")
            print(f"  {'Tabla':<8} {'Filas':>10} {'Dups':>8} {'% Dups':>8} {'PKs únicos':>12} {'% Dup PK':>9}")
            print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*9}")
            with open(dups_csv, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    filas  = int(row.get("filas", 0))
                    dups   = int(row.get("duplicados", 0))
                    pct_d  = float(row.get("pct_duplicados", 0))
                    pks    = int(row.get("pk_unicos", 0))
                    pct_pk = round((filas - pks) / filas * 100, 2) if filas > 0 else 0
                    print(f"  {row.get('tabla','?'):<8} {filas:>10,} {dups:>8,} {pct_d:>7.2f}% {pks:>12,} {pct_pk:>8.2f}%")
        except Exception:
            pass

    # ── Plan de limpieza (s09_resumen_completo) ──────────────────────────
    resumen_csv = EDA_OUTPUTS / "s09_resumen_completo.csv"
    if resumen_csv.exists():
        try:
            acciones: Counter = Counter()
            tablas_vistas: set = set()
            total_cols = 0
            with open(resumen_csv, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    total_cols += 1
                    tablas_vistas.add(row.get("tabla", "?"))
                    acciones[row.get("accion", "?")] += 1
            print(f"\n  {sep}")
            print("  PLAN DE LIMPIEZA — ACCIONES POR COLUMNA")
            print(f"  {sep}")
            print(f"  {'Acción':<35} {'# Columnas':>12}")
            print(f"  {'-'*35} {'-'*12}")
            for accion, n in sorted(acciones.items(), key=lambda x: -x[1]):
                print(f"  {accion:<35} {n:>12}")
            print(f"  {'-'*35} {'-'*12}")
            print(f"  {'TOTAL':<35} {total_cols:>12}")
            print(f"\n  Tablas: {len(tablas_vistas)}  ({', '.join(sorted(tablas_vistas))})")
        except Exception:
            pass

    # ── Balanceo de clases (s09_balanceo) ─────────────────────────────
    balanceo_csv = EDA_OUTPUTS / "s09_balanceo.csv"
    if balanceo_csv.exists():
        try:
            print(f"\n  {sep}")
            print("  DISTRIBUCIÓN DE CLASES — VARIABLE OBJETIVO (severity)")
            print(f"  {sep}")
            print(f"  {'Clase':<7} {'Descripción':<14} {'Count':>10} {'%':>7} {'Ratio':>7}  Estrategia")
            print(f"  {'-'*7} {'-'*14} {'-'*10} {'-'*7} {'-'*7}  {'-'*28}")
            with open(balanceo_csv, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    count  = int(float(row.get("count", 0)))
                    pct    = float(row.get("porcentaje", 0))
                    ratio  = float(row.get("ratio", 1))
                    estrat = row.get("estrategia", "")
                    print(f"  {row.get('clase','?'):<7} {row.get('descripcion','?'):<14} {count:>10,} {pct:>6.1f}% {ratio:>6.1f}x  {estrat}")
        except Exception:
            pass

    print(f"\n  {sep}")
    print(f"  Carpeta: outputs/eda_results/  ({len(archivos)} archivos generados)")
    print(f"  {sep}")


# ─── Opción 2 -- Limpieza de datos ─────────────────────────────────────────────

def ejecutar_limpieza() -> None:
    """Ejecuta el pipeline de limpieza mostrando su salida en vivo."""
    print("\n  Iniciando limpieza de datos...\n")
    result = subprocess.run(
        [sys.executable, "src/cleaning/cleaning_main.py"],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("\n  [OK] Limpieza completada. Datos en: data/clean_data/\n")
    else:
        print(f"\n  [ERROR] Limpieza falló (código {result.returncode}).\n")
    input("  Presione Enter para volver al menu...")


def _cargar_cache() -> dict:
    if CONFIG_CACHE.exists():
        try:
            return json.loads(CONFIG_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _guardar_cache(tiempos: dict):
    data = dict(tiempos)
    data["last_run_total"] = round(sum(tiempos.values()), 1)
    data["runs"] = _cargar_cache().get("runs", 0) + 1
    try:
        CONFIG_CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


SUBMENU_JOIN = """
  ┌─────────────────────────────────────────────────────────────┐
  │           JOIN CON HADOOP Y MAPREDUCE                       │
  ├─────────────────────────────────────────────────────────────┤
  │  1. Probar con 10,000 datos  (rapido, para verificar)       │
  │  2. Join completo            (todos los datos)              │
  │  0. Volver al menu principal                                │
  └─────────────────────────────────────────────────────────────┘
"""

def ejecutar_join() -> None:
    """Muestra submenú y ejecuta el pipeline de join con Hadoop MapReduce."""
    while True:
        _limpiar_pantalla()
        print(SUBMENU_JOIN)
        sub = input("  Seleccione opcion: ").strip()

        if sub == "1":
            print("\n  [3.1] JOIN CON HADOOP - PRUEBA (10,000 datos)\n")
            result = subprocess.run(
                [sys.executable, "src/hadoop/hadoop_main.py", "--sample", "10000"],
                capture_output=False,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                print("\n  [OK] Join de prueba completado. Datos en: data/joined/\n")
            else:
                print(f"\n  [ERROR] Join de prueba falló (código {result.returncode}).\n")
            input("  Presione Enter para continuar...")
            break

        elif sub == "2":
            print("\n  [3.2] JOIN CON HADOOP - COMPLETO\n")
            result = subprocess.run(
                [sys.executable, "src/hadoop/hadoop_main.py"],
                capture_output=False,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                print("\n  [OK] Join completo. Datos en: data/joined/\n")
            else:
                print(f"\n  [ERROR] Join completo falló (código {result.returncode}).\n")
            input("  Presione Enter para continuar...")
            break

        elif sub == "0":
            break

        else:
            print(f"\n  Opcion '{sub}' no valida. Elija 0, 1 o 2.\n")
            input("  Presione Enter para continuar...")



# ─── Opción 5 -- Entrenar modelo con Hadoop MapReduce ───────────────────────────

def ejecutar_entrenamiento() -> None:
    """Ejecuta el pipeline de entrenamiento con Hadoop MapReduce."""
    print("\n  [5] ENTRENAR MODELO CON HADOOP MAPREDUCE\n")
    result = subprocess.run(
        [sys.executable, "src/hadoop/training/training_main.py"],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("\n  [OK] Entrenamiento completado.\n")
    else:
        print(f"\n  [ERROR] Entrenamiento falló (código {result.returncode}).\n")
    input("  Presione Enter para volver al menu...")


# ─── Opción 6 -- Correr modelo con Kafka ────────────────────────────────────────

def ejecutar_kafka() -> None:
    """Lanza el pipeline de inferencia en streaming con Kafka."""
    print("\n  [6] CORRER MODELO CON KAFKA\n")
    try:
        result = subprocess.run(
            [sys.executable, "src/faers_kafka/main_kafka.py"],
            capture_output=False,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            print("\n  [OK] Kafka pipeline completado.\n")
        else:
            print(f"\n  [ERROR] Kafka pipeline falló (código {result.returncode}).\n")
    except KeyboardInterrupt:
        # Se captura el Ctrl+C de forma limpia; main_kafka.py ya muestra su propio resumen final
        print("\n\n  [*] Volviendo al menú principal...")
        time.sleep(1.0)
    input("  Presione Enter para volver al menu...")


def ejecutar_consulta() -> None:
    """Lanza la consulta interactiva con el modelo RandomForest y señales de MapReduce."""
    print("\n  [7] CONSULTAR MODELO CON CASO CLÍNICO (PREDICCIÓN ML)\n")
    result = subprocess.run(
        [sys.executable, "src/utils/interactive_menu.py"],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"\n  [ERROR] Falló al lanzar la consulta interactiva (código {result.returncode}).\n")


# ─── Opción 4 -- Balanceo de datos ──────────────────────────────────────────────

def ejecutar_balanceo() -> None:
    """Ejecuta el pipeline de balanceo mostrando su salida en vivo."""
    print("\n  Iniciando balanceo de datos...\n")
    result = subprocess.run(
        [sys.executable, "src/balance/balance_main.py"],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("\n  [OK] Balanceo completado. Datos en: data/balanced_data/\n")
    else:
        print(f"\n  [ERROR] Balanceo falló (código {result.returncode}).\n")
    input("  Presione Enter para volver al menu...")


# ─── Opción 7 -- Salir ──────────────────────────────────────────────────────────

def salir() -> None:
    print("\n  Hasta luego.\n")
    sys.exit(0)


# ─── Main -- bucle de menu ─────────────────────────────────────────────────────

def _limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    while True:
        _limpiar_pantalla()
        print(MENU)
        opcion = input("  Seleccione opcion: ").strip()

        if opcion == "0":
            configurar_entorno()

        elif opcion == "1":
            ejecutar_eda()

        elif opcion == "2":
            ejecutar_limpieza()

        elif opcion == "3":
            ejecutar_join()

        elif opcion == "4":
            ejecutar_balanceo()

        elif opcion == "5":
            ejecutar_entrenamiento()

        elif opcion == "6":
            ejecutar_kafka()

        elif opcion == "7":
            ejecutar_consulta()

        elif opcion == "8":
            salir()

        else:
            print(f"\n  Opcion '{opcion}' no valida. Elija del 0 al 8.\n")
            input("  Presione Enter para continuar...")


if __name__ == "__main__":
    main()
