"""
src/kafka/config.py
===================
CONFIGURACIONES GENERALES DEL PIPELINE DE STREAMING REAL-TIME

Define los parámetros de conexión para Apache Kafka, rutas a datos históricos y crudos,
y especificaciones del modo simulado para depuración local.
"""

from pathlib import Path

# --- Rutas del Proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "faers"
SIGNALS_FILE = PROJECT_ROOT / "outputs" / "mapreduce_results" / "top_safety_signals.csv"
ALERTS_DIR   = PROJECT_ROOT / "data" / "alerts"
ALERTS_FILE  = ALERTS_DIR / "realtime_alerts.jsonl"

# --- Ingesta y Simulación ---
STREAMING_RAW_DIR    = PROJECT_ROOT / "data" / "streaming_raw"
SIMULATED_QUEUE_FILE = STREAMING_RAW_DIR / "simulated_queue.jsonl"
SIMULATION_DELAY     = 1.5  # Segundos entre eventos para la consola en vivo

# --- Logs del Sistema ---
LOGS_DIR  = PROJECT_ROOT / "logs"
LOGS_FILE = LOGS_DIR / "streaming_pipeline.log"

# --- Parámetros de Kafka ---
BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC_NAME        = "faers-raw-reports"

# Asegurar directorios críticos
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
STREAMING_RAW_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
