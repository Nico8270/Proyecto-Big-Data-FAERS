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
║  3. Hacer join de datos limpios                               ║
║  4. Balancear datos                                           ║
║  5. Salir                                                     ║
╚═══════════════════════════════════════════════════════════════╝
"""


# ─── Opción 0 -- Configuración de entorno ──────────────────────────────────────

def configurar_entorno() -> None:
    """Instala Python, dependencias y verifica estructura de carpetas."""
    print(MENU.replace("╚══", "╠══").replace("╝", "║").rstrip())
    print("\n  Configurando entorno...\n")

    pasos = [
        ("Verificando Python...",
         _check_python),
        ("Instalando/actualizando pip...",
         _install_pip),
        ("Instalando dependencias EDA...",
         _install_requirements, "requirements.txt"),
        ("Instalando dependencias Hadoop...",
         _install_requirements, "requirements_hadoop.txt"),
        ("Instalando mrjob...",
         _install_mrjob),
        ("Instalando imbalanced-learn (para balanceo)...",
         _install_imblearn),
        ("Verificando estructura de carpetas...",
         _verify_structure),
        ("Limpiando outputs previos...",
         _clear_outputs),
    ]

    bar = ProgressBar(total=len(pasos), label="Configuracion de entorno")
    bar.start()

    for label, fn, *extra in pasos:
        try:
            fn(*extra)
        except Exception as exc:
            print(f"\n  [AVISO] {label}: {exc}")
        bar.next(label=label)

    bar.finish()
    print("\n  [OK] Entorno configurado. Listo para ejecutar el EDA.\n")
    input("  Presione Enter para volver al menu...")


def _run_silent(cmd: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Ejecuta un comando en silencio, sin flood de salida."""
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )


def _check_python():
    result = _run_silent([sys.executable, "--version"])
    version = (result.stdout or result.stderr or "").strip()
    print(f"\n    Python: {version}")


def _install_pip():
    _run_silent([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])


def _install_requirements(file_: str):
    req = PROJECT_ROOT / file_
    if req.exists():
        _run_silent([sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"])
    else:
        raise FileNotFoundError(f"No encontrado: {req}")


def _install_mrjob():
    _run_silent([sys.executable, "-m", "pip", "install", "mrjob", "--quiet"])


def _install_imblearn():
    _run_silent([sys.executable, "-m", "pip", "install", "imbalanced-learn", "--quiet"])


def _verify_structure():
    carpetas = [
        PROJECT_ROOT / "data" / "raw" / "faers",
        PROJECT_ROOT / "data" / "staging",
        PROJECT_ROOT / "data" / "joined",
        PROJECT_ROOT / "data" / "aggregated",
        PROJECT_ROOT / "data" / "streaming_raw",
        PROJECT_ROOT / "data" / "alerts",
        PROJECT_ROOT / "data" / "balanced_data",
        PROJECT_ROOT / "src" / "hadoop" / "mapper",
        PROJECT_ROOT / "src" / "hadoop" / "reducer",
        PROJECT_ROOT / "src" / "training",
        PROJECT_ROOT / "src" / "eda",
        PROJECT_ROOT / "src" / "balance" / "logs",
    ]
    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)


def _clear_outputs():
    for carpeta in [OUTPUTS_DIR / "eda_results", OUTPUTS_DIR / "mapreduce_results",
                    OUTPUTS_DIR / "balance_reports"]:
        if carpeta.exists():
            shutil.rmtree(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)


# ─── Opción 1 -- EDA con barra de progreso ────────────────────────────────────

def ejecutar_eda() -> None:
    """Ejecuta el pipeline EDA completo con barra de progreso y ETA."""
    _limpiar_outputs()
    cache = _cargar_cache()
    estimado_inicial = cache.get("last_run_total", len(EDA_SCRIPTS) * 30)

    bar = ProgressBar(
        total              = len(EDA_SCRIPTS),
        label              = "EDA FAERS Q4 2025",
        bar_width          = 30,
    )
    bar.start()

    tiempos = {}
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
        duracion = round(time.time() - t0, 1)
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

    # Guardar caché para la próxima ejecución
    _guardar_cache(tiempos)

    estado = "CON ERRORES" if error_ocurrido else "COMPLETADO"
    print(f"\n  [OK] EDA {estado}. Resultados en: {EDA_OUTPUTS}\n")
    input("  Presione Enter para volver al menu...")


def _limpiar_outputs():
    if EDA_OUTPUTS.exists():
        shutil.rmtree(EDA_OUTPUTS)
    EDA_OUTPUTS.mkdir(parents=True, exist_ok=True)


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


# ─── Opción 3 -- Join de datos ──────────────────────────────────────────────────

def ejecutar_join() -> None:
    """Ejecuta el pipeline de consolidación (join) mostrando su salida en vivo."""
    print("\n  Iniciando join de datos limpios...\n")
    result = subprocess.run(
        [sys.executable, "src/cleaning/s08_consolidar_clean.py"],
        capture_output=False,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print("\n  [OK] Join completado. Datos en: data/joined/\n")
    else:
        print(f"\n  [ERROR] Join falló (código {result.returncode}).\n")
    input("  Presione Enter para volver al menu...")


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


# ─── Opción 5 -- Salir ──────────────────────────────────────────────────────────

def salir() -> None:
    print("\n  Hasta luego.\n")
    sys.exit(0)


# ─── Main -- bucle de menu ─────────────────────────────────────────────────────

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
            salir()

        else:
            print(f"\n  Opcion '{opcion}' no valida. Elija 0, 1, 2, 3, 4 o 5.\n")
            input("  Presione Enter para continuar...")


@staticmethod
def _limpiar_pantalla():
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
