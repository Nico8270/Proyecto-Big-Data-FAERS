"""
src/kafka/consumer.py
=====================
CONSUMIDOR Y ANALIZADOR EN TIEMPO REAL FAERS

Consume los eventos del stream (Kafka o cola compartida), carga el modelo
entrenado (outputs/mapreduce_results/top_safety_signals.csv) y evalúa
cada combinación fármaco-reacción en base a gravedad clínica y significancia estadística.
"""

import json
import os
import time
import sys
import queue
from pathlib import Path
import pandas as pd

from kafka.errors import NoBrokersAvailable
from kafka import KafkaConsumer

from faers_kafka.config import BOOTSTRAP_SERVERS, TOPIC_NAME, SIGNALS_FILE, ALERTS_FILE

# Códigos ANSI para dar estética premium en consola
RED     = '\033[91m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
MAGENTA = '\033[95m'
CYAN    = '\033[96m'
BOLD    = '\033[1m'
RESET   = '\033[0m'
import logging
from faers_kafka.config import LOGS_FILE

logging.basicConfig(
    filename=str(LOGS_FILE),
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

def log_info(message: str):
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_msg = ansi_escape.sub('', message)
    logging.info(clean_msg)

def safe_print(*args, **kwargs):
    text = " ".join(str(arg) for arg in args)
    try:
        sys.stdout.write(text + kwargs.get("end", "\n"))
        sys.stdout.flush()
    except UnicodeEncodeError:
        text_safe = text.replace("🚨", "[CRITICO]").replace("⚠️", "[ALERTA]").replace("─", "-").replace("•", "*")
        try:
            sys.stdout.write(text_safe + kwargs.get("end", "\n"))
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(text_safe.encode('ascii', errors='ignore').decode('ascii') + kwargs.get("end", "\n"))
            sys.stdout.flush()

print = safe_print

class FAERSConsumer:
    def __init__(self, shared_queue: queue.Queue = None):
        self.shared_queue = shared_queue
        self.consumer = None
        self.simulated = False
        
        # Estadísticas en tiempo real
        self.stats = {
            "total_consumidos": 0,
            "total_alertas": 0,
            "alertas_criticas": 0,
            "alertas_altas": 0,
            "alertas_conocidas": 0,
            "alertas_nuevas": 0
        }
        
        # Cargar modelo entrenado
        self.modelo_senales = self._cargar_modelo_entrenado()
        
        # Intentar conexión con Kafka
        try:
            print(f"[*] Conectando al consumidor Kafka en {BOOTSTRAP_SERVERS}...")
            self.consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                consumer_timeout_ms=1000,
                request_timeout_ms=5000
            )
            print("[OK] Consumidor Kafka inicializado con éxito.")
        except (NoBrokersAvailable, Exception) as e:
            print(f"[!] AVISO: No se pudo conectar el Consumidor al Broker Kafka ({e}).")
            print("[*] Iniciando Consumidor en MODO SIMULADO (Local Queue).")
            self.simulated = True

    def _cargar_modelo_entrenado(self) -> dict:
        """Carga el CSV de señales y lo indexa en memoria para búsquedas O(1)."""
        senales = {}
        if not SIGNALS_FILE.exists():
            print(f"{YELLOW}[AVISO] No se encontró el modelo de señales ({SIGNALS_FILE.name}).")
            print(f"        El enriquecimiento se hará sin datos históricos. Realice el entrenamiento (opción 5).{RESET}")
            return senales
        
        try:
            df = pd.read_csv(SIGNALS_FILE)
            for _, row in df.iterrows():
                drug = str(row["drugname"]).upper().strip()
                pt = str(row["pt"]).upper().strip()
                senales[(drug, pt)] = {
                    "prr": float(row["prr"]),
                    "chi2": float(row["chi2"]),
                    "count_a": int(row["count_a"])
                }
            print(f"[OK] Modelo de entrenamiento cargado: {len(senales):,} señales activas indexadas.")
        except Exception as e:
            print(f"{RED}[ERROR] Al cargar el modelo entrenado: {e}{RESET}")
        return senales

    def _evaluar_gravedad(self, outcomes: list) -> tuple[int, str, str]:
        """Calcula el nivel de gravedad del caso en base a los outcomes reportados."""
        # outcomes oficiales: DE (Death), LT (Life-Threatening), HO (Hospitalization), DS (Disability)...
        outcomes_set = set(str(o).upper().strip() for o in outcomes if o)
        
        if "DE" in outcomes_set or "LT" in outcomes_set:
            return 4, "CRÍTICA", RED
        elif "HO" in outcomes_set or "DS" in outcomes_set or "CA" in outcomes_set:
            return 3, "ALTA", YELLOW
        elif "RI" in outcomes_set or "OT" in outcomes_set:
            return 2, "MEDIA", CYAN
        else:
            return 1, "BAJA", GREEN

    def procesar_caso(self, caso: dict):
        """Aplica la lógica del modelo e infiere alertas para un reporte de paciente."""
        self.stats["total_consumidos"] += 1
        
        pid = caso.get("primaryid", "DESCONOCIDO")
        sex = caso.get("sex", "U")
        age = caso.get("age", "N/A")
        country = caso.get("reporter_country", "UNKNOWN")
        drugs = caso.get("drugs", [])
        reactions = caso.get("reactions", [])
        outcomes = caso.get("outcomes", [])
        
        severity_level, severity_label, severity_color = self._evaluar_gravedad(outcomes)
        
        # Enviar logs detallados del caso al archivo de logs en disco
        log_info(f"================================================================================")
        log_info(f"[INGESTA] Caso: {pid} | Sexo: {sex} | Edad: {age} | País: {country}")
        log_info(f"          Fármacos: {', '.join(drugs)}")
        log_info(f"          Reacciones: {', '.join(reactions)}")
        log_info(f"          Gravedad Caso: {severity_label} (Outcomes: {', '.join(outcomes) or 'Ninguno'})")
        log_info(f"--------------------------------------------------------------------------------")
        
        alertas_caso = []
        
        # Evaluar cruzando parejas fármaco-reacción
        for drug in drugs:
            drug_up = drug.upper().strip()
            for pt in reactions:
                pt_up = pt.upper().strip()
                
                # Buscar en el modelo histórico
                key = (drug_up, pt_up)
                match = self.modelo_senales.get(key)
                
                is_known_signal = False
                prr = None
                chi2 = None
                count_a = 0
                
                if match:
                    prr = match["prr"]
                    chi2 = match["chi2"]
                    count_a = match["count_a"]
                    is_known_signal = (prr >= 2.0) and (chi2 >= 3.84)
                
                # Decisión de alerta en tiempo real:
                # 1. Gravedad Crítica/Alta del paciente por este reporte.
                # 2. Combinación estadísticamente significativa en el modelo histórico.
                trigger_alert = False
                alert_reason = ""
                
                if severity_level >= 3:
                    trigger_alert = True
                    alert_reason = "Gravedad del Desenlace Clínico"
                elif is_known_signal:
                    trigger_alert = True
                    alert_reason = "Señal de Seguridad Activa en Modelo Histórico"
                elif chi2 and chi2 > 10.0:
                    trigger_alert = True
                    alert_reason = "Alta Correlación Estadística (Chi2 > 10.0)"
                
                if trigger_alert:
                    # Clasificar alerta
                    alert_type = "NUEVA_ASOCIACION_SEVERA"
                    if is_known_signal:
                        alert_type = "SEÑAL_SEGURIDAD_CONOCIDA"
                        self.stats["alertas_conocidas"] += 1
                    else:
                        self.stats["alertas_nuevas"] += 1
                        
                    alert_item = {
                        "alert_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "primaryid": pid,
                        "sex": sex,
                        "age": age,
                        "reporter_country": country,
                        "drugname": drug_up,
                        "pt": pt_up,
                        "severity_level": severity_level,
                        "severity_label": severity_label,
                        "is_known_signal": is_known_signal,
                        "prr": prr,
                        "chi2": chi2,
                        "alert_type": alert_type,
                        "alert_reason": alert_reason,
                        "outcomes": outcomes
                    }
                    
                    alertas_caso.append(alert_item)
                    self.stats["total_alertas"] += 1
                    if severity_level == 4:
                        self.stats["alertas_criticas"] += 1
                    elif severity_level == 3:
                        self.stats["alertas_altas"] += 1
                        
                    # 1. LOG AL ARCHIVO DETALLADO
                    log_info(f"[ALERTA DETECTADA] {drug_up} + {pt_up} | Tipo: {alert_type} ({alert_reason})")
                    if match:
                        log_info(f"                  Métricas: PRR = {prr:.2f} | Chi2 = {chi2:.2f} | Histórico = {count_a} casos")
                    else:
                        log_info(f"                  Métricas: No registradas históricamente (Nueva Co-ocurrencia!)")
                    
                else:
                    # Información de control normal (SOLO al archivo de logs)
                    log_info(f"[Control] {drug_up} + {pt_up}: Sin alerta activa (PRR: {prr}, Chi2: {chi2})")
                    
        # Imprimir una única línea consolidada e inteligente en consola si hay alertas para este caso
        # Guardar alertas del caso a disco
        if alertas_caso:
            try:
                with open(ALERTS_FILE, "a", encoding="utf-8") as f:
                    for alert in alertas_caso:
                        f.write(json.dumps(alert) + "\n")
            except Exception as e:
                print(f"  [ERROR] No se pudo escribir alerta a disco: {e}")

    def consume_data(self, stop_event):
        """Inicia el bucle de escucha de eventos."""
        print("[*] Iniciando Consumidor en espera de eventos clínicos...")
        
        while not stop_event.is_set():
            if self.simulated:
                # Consumo de cola en memoria compartida
                try:
                    caso = self.shared_queue.get(timeout=1.0)
                    self.procesar_caso(caso)
                    self.shared_queue.task_done()
                except queue.Empty:
                    continue
            else:
                # Consumo real de Apache Kafka con poll no bloqueante
                try:
                    records = self.consumer.poll(timeout_ms=1000)
                    for tp, msgs in records.items():
                        for msg in msgs:
                            if stop_event.is_set():
                                break
                            self.procesar_caso(msg.value)
                except Exception as e:
                    print(f"\n[ERROR] En el bucle de consumo Kafka: {e}. Reintentando...")
                    time.sleep(2.0)
                    
        # Cerrar consumidor si existe
        if self.consumer:
            try:
                self.consumer.close()
            except Exception:
                pass
        print(f"\n[OK] Consumidor finalizado.")
