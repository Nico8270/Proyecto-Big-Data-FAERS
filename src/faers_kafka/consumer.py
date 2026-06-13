"""
src/kafka/consumer.py
=====================
CONSUMIDOR Y ANALIZADOR EN TIEMPO REAL FAERS

Consume eventos del stream (Kafka o cola compartida), carga el modelo ML entrenado
(modelo_severidad.joblib) y predice la severidad de cada caso clínico.
Genera alertas automáticas cuando la predicción indica severidad Grave (4) o Muy Grave (5).
"""

import json
import os
import time
import sys
import queue
from pathlib import Path

import pandas as pd
import joblib

from kafka.errors import NoBrokersAvailable
from kafka import KafkaConsumer

from faers_kafka.config import BOOTSTRAP_SERVERS, TOPIC_NAME, ALERTS_FILE

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

        # Estadísticas ACUMULADAS a lo largo de toda la sesión (no por batch)
        # Estas se actualizan con los deltas de cada micro-batch procesado.
        self.stats = {
            "total_consumidos":  0,
            "total_alertas":     0,
            "alertas_criticas":  0,
            "alertas_altas":     0,
            "alertas_conocidas": 0,
            "alertas_nuevas":    0,
        }

        # Cargar modelo entrenado y encoders
        self.modelo_ml = self._cargar_modelo_ml()
        self.encoders = self._cargar_encoders()

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

    def _cargar_modelo_ml(self):
        """Carga el modelo ML entrenado para predicción de severidad."""
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        model_path = PROJECT_ROOT / "outputs" / "model_assets" / "modelo_severidad.joblib"
        
        if not model_path.exists():
            print(f"{YELLOW}[AVISO] Modelo ML no encontrado en {model_path}.{RESET}")
            print(f"        Asegúrese de ejecutar la etapa 5 de entrenamiento primero.")
            return None
        
        try:
            clf = joblib.load(model_path)
            print(f"[OK] Modelo ML cargado: {model_path.name}")
            return clf
        except Exception as e:
            print(f"{RED}[ERROR] Al cargar el modelo ML: {e}{RESET}")
            return None
    
    def _cargar_encoders(self):
        """Carga los encoders de características categóricas."""
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        assets_dir = PROJECT_ROOT / "outputs" / "model_assets"
        
        encoders = {
            "drug": None,
            "pt": None
        }
        
        try:
            drug_enc_path = assets_dir / "drug_encoder.joblib"
            pt_enc_path = assets_dir / "pt_encoder.joblib"
            
            if drug_enc_path.exists():
                encoders["drug"] = joblib.load(drug_enc_path)
            if pt_enc_path.exists():
                encoders["pt"] = joblib.load(pt_enc_path)
            
            return encoders
        except Exception as e:
            print(f"{RED}[AVISO] Error cargando encoders: {e}{RESET}")
            return encoders

    def procesar_micro_batch(self, casos: list[dict]) -> list[dict]:
        """
        Evalúa casos usando el modelo ML entrenado.
        
        Para cada caso, extrae features (age, sex, drug, pt), aplica encoders,
        y predice severidad con el modelo RandomForest.
        Si severidad >= 4 (Grave o Muy Grave), genera una alerta.
        """
        
        if self.modelo_ml is None:
            return []
        
        alertas_batch: list[dict] = []
        batch_total_alertas = 0
        batch_alertas_criticas = 0
        batch_alertas_altas = 0
        
        for caso in casos:
            try:
                # Extraer features del caso
                age = pd.to_numeric(caso.get("age", 45.0), errors="coerce")
                if pd.isna(age):
                    age = 45.0
                
                sex = str(caso.get("sex", "U")).upper().strip()
                sex_map = {"M": 1.0, "F": 0.0, "U": 0.5}
                sex_encoded = sex_map.get(sex, 0.5)
                
                # Tomar el primer fármaco y reacción (en casos reales, iterar sobre todos)
                drugs = caso.get("drugs", [])
                reactions = caso.get("reactions", [])
                
                if not drugs or not reactions:
                    continue
                
                drug = str(drugs[0]).upper().strip()
                pt = str(reactions[0]).upper().strip()
                
                # Codificar usando los encoders
                drug_encoded = -1.0
                pt_encoded = -1.0
                
                if self.encoders["drug"] is not None:
                    try:
                        drug_encoded = float(
                            self.encoders["drug"].transform([[drug]]).ravel()[0]
                        )
                    except:
                        drug_encoded = -1.0
                
                if self.encoders["pt"] is not None:
                    try:
                        pt_encoded = float(
                            self.encoders["pt"].transform([[pt]]).ravel()[0]
                        )
                    except:
                        pt_encoded = -1.0
                
                # Armar feature vector
                X = pd.DataFrame([[drug_encoded, pt_encoded, sex_encoded, age]],
                                columns=["drug_encoded", "pt_encoded", "sex_encoded", "age"])
                
                # Predecir severidad
                pred_shifted = self.modelo_ml.predict(X)[0]
                
                # Revertir label shift (modelo predice en [0,1,2,3,4], necesitamos [1,2,3,4,5])
                label_shift = 1
                pred_severidad = pred_shifted + label_shift
                
                # Obtener probabilidades para confidence score
                pred_probs = self.modelo_ml.predict_proba(X)[0]
                confidence_score = float(pred_probs[pred_shifted])
                
                # Generar alerta si severidad >= 4
                if pred_severidad >= 4:
                    severity_label = "Grave" if pred_severidad == 4 else "Muy Grave"
                    alert_type = "PREDICCIÓN_ML_CRÍTICA"
                    
                    alert_item = {
                        "alert_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "primaryid": int(caso.get("primaryid", 0)),
                        "sex": sex,
                        "age": float(age),
                        "drugname": drug,
                        "pt": pt,
                        "predicted_severity": int(pred_severidad),
                        "severity_label": severity_label,
                        "confidence_score": round(confidence_score, 4),
                        "alert_type": alert_type,
                        "alert_reason": (
                            f"Modelo ML predice severidad={pred_severidad} ({severity_label}) "
                            f"con confianza={confidence_score:.2%}"
                        ),
                        "outcomes": caso.get("outcomes", [])
                    }
                    
                    alertas_batch.append(alert_item)
                    batch_total_alertas += 1
                    
                    if pred_severidad == 4:
                        batch_alertas_criticas += 1
                        color_tag = RED
                    else:
                        batch_alertas_altas += 1
                        color_tag = YELLOW
                    
                    log_info(
                        f"[ALERTA] {drug} + {pt} | "
                        f"Severidad={pred_severidad} ({severity_label}) | "
                        f"Conf={confidence_score:.2%}"
                    )
            
            except Exception as e:
                log_info(f"[ERROR] Procesando caso {caso.get('primaryid')}: {e}")
                continue
        
        # Actualizar estadísticas globales
        self.stats["total_consumidos"] += len(casos)
        self.stats["total_alertas"] += batch_total_alertas
        self.stats["alertas_criticas"] += batch_alertas_criticas
        self.stats["alertas_altas"] += batch_alertas_altas
        self.stats["alertas_conocidas"] += batch_total_alertas
        
        # Persistir alertas a disco
        if alertas_batch:
            try:
                with open(ALERTS_FILE, "a", encoding="utf-8") as f:
                    for alert in alertas_batch:
                        f.write(json.dumps(alert) + "\n")
            except Exception as e:
                print(f"  [ERROR] No se pudo escribir alerta a disco: {e}")
        
        return alertas_batch

    def consume_data(self, stop_event):
        """Inicia el bucle de escucha de eventos."""
        print("[*] Iniciando Consumidor en espera de eventos clínicos...")

        while not stop_event.is_set():
            if self.simulated:
                # Consumo de cola en memoria compartida
                try:
                    caso = self.shared_queue.get(timeout=1.0)
                    self.procesar_micro_batch([caso])
                    self.shared_queue.task_done()
                except queue.Empty:
                    continue
            else:
                # Consumo real de Apache Kafka con poll no bloqueante
                # Bug 1 corregido: se recoge todo el micro-batch del poll y se
                # evalúa en conjunto con procesar_micro_batch(), no fila a fila.
                try:
                    records = self.consumer.poll(timeout_ms=1000)
                    batch: list[dict] = []
                    for tp, msgs in records.items():
                        for msg in msgs:
                            if stop_event.is_set():
                                break
                            batch.append(msg.value)
                    if batch:
                        self.procesar_micro_batch(batch)
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
