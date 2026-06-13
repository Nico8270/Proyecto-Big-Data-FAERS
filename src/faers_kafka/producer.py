"""
src/kafka/producer.py
=====================
PRODUCTOR DE EVENTOS DE STREAMING FAERS

Lee los datos crudos desde la carpeta data/raw/faers/ (DEMO, DRUG, REAC, OUTC)
y ensambla los registros por `primaryid` para simular/ingestar reportes de
pacientes completos hacia el Topic de Kafka (o cola local).
"""

import json
import time
import queue
import random
from pathlib import Path
import pandas as pd

from kafka.errors import NoBrokersAvailable
from kafka import KafkaProducer

from faers_kafka.config import BOOTSTRAP_SERVERS, TOPIC_NAME, RAW_DATA_DIR, SIMULATED_QUEUE_FILE, LOGS_FILE

import logging
logging.basicConfig(
    filename=str(LOGS_FILE),
    filemode="a",
    format="%(asctime)s [%(levelname)s] (Productor) %(message)s",
    level=logging.INFO,
    encoding="utf-8"
)

def log_print(*args, **kwargs):
    text = " ".join(str(arg) for arg in args)
    # Strip any ANSI color codes if they exist in producer prints
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', text)
    logging.info(clean_text)

print = log_print

class FAERSProducer:
    def __init__(self, shared_queue: queue.Queue = None):
        self.shared_queue = shared_queue
        self.producer = None
        self.simulated = False
        
        # Intentar conectar con el cliente oficial de Kafka
        try:
            print(f"[*] Conectando al broker Kafka en {BOOTSTRAP_SERVERS}...")
            self.producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=5000,
                connection_max_idle_ms=10000
            )
            print("[OK] Conexión con Kafka establecida con éxito.")
        except (NoBrokersAvailable, Exception) as e:
            print(f"[!] AVISO: No se pudo conectar al Broker de Kafka ({e}).")
            print("[*] Iniciando Productor en MODO SIMULADO (Local Queue).")
            self.simulated = True

    def _cargar_casos_reales(self, nrows: int = 5000) -> list[dict]:
        """Carga y ensambla reportes de pacientes uniendo archivos crudos por primaryid."""
        print(f"[*] Leyendo archivos crudos de FAERS para streaming (límite: {nrows} filas)...")
        
        demo_path = RAW_DATA_DIR / "DEMO25Q4.txt"
        drug_path = RAW_DATA_DIR / "DRUG25Q4.txt"
        reac_path = RAW_DATA_DIR / "REAC25Q4.txt"
        outc_path = RAW_DATA_DIR / "OUTC25Q4.txt"
        
        # Verificar que existan al menos los archivos principales
        if not (demo_path.exists() and drug_path.exists() and reac_path.exists()):
            print("[AVISO] No se encontraron archivos crudos en data/raw/faers.")
            return self._generar_casos_sinteticos()
        
        try:
            # 1. Cargar DEMO
            df_demo = pd.read_csv(demo_path, sep="$", encoding="latin-1", nrows=nrows, on_bad_lines="skip", quoting=3)
            df_demo.columns = df_demo.columns.str.strip().str.lower()
            
            # 2. Cargar DRUG
            df_drug = pd.read_csv(drug_path, sep="$", encoding="latin-1", nrows=nrows*2, on_bad_lines="skip", quoting=3)
            df_drug.columns = df_drug.columns.str.strip().str.lower()
            
            # 3. Cargar REAC
            df_reac = pd.read_csv(reac_path, sep="$", encoding="latin-1", nrows=nrows*2, on_bad_lines="skip", quoting=3)
            df_reac.columns = df_reac.columns.str.strip().str.lower()
            
            # 4. Cargar OUTC (opcional)
            df_outc = None
            if outc_path.exists():
                df_outc = pd.read_csv(outc_path, sep="$", encoding="latin-1", nrows=nrows, on_bad_lines="skip", quoting=3)
                df_outc.columns = df_outc.columns.str.strip().str.lower()
                
            # Limpiar y agrupar fármacos
            df_drug = df_drug.dropna(subset=["primaryid", "drugname"])
            df_drug["drugname"] = df_drug["drugname"].astype(str).str.upper().str.strip()
            drug_dict = df_drug.groupby("primaryid")["drugname"].apply(list).to_dict()
            
            # Limpiar y agrupar reacciones
            df_reac = df_reac.dropna(subset=["primaryid", "pt"])
            df_reac["pt"] = df_reac["pt"].astype(str).str.upper().str.strip()
            reac_dict = df_reac.groupby("primaryid")["pt"].apply(list).to_dict()
            
            # Limpiar y agrupar outcomes
            outc_dict = {}
            if df_outc is not None and "outc_cod" in df_outc.columns:
                df_outc = df_outc.dropna(subset=["primaryid", "outc_cod"])
                df_outc["outc_cod"] = df_outc["outc_cod"].astype(str).str.upper().str.strip()
                outc_dict = df_outc.groupby("primaryid")["outc_cod"].apply(list).to_dict()
            
            # Ensamblar casos de pacientes
            casos = []
            for _, row in df_demo.iterrows():
                pid = row.get("primaryid")
                if pd.isna(pid):
                    continue
                pid = int(pid)
                
                # Omitir si no tiene fármacos o reacciones
                drugs = drug_dict.get(pid, [])
                reactions = reac_dict.get(pid, [])
                if not drugs or not reactions:
                    continue
                
                caso = {
                    "primaryid": pid,
                    "sex": str(row.get("sex", "U")).upper().strip(),
                    "age": float(row["age"]) if pd.notna(row.get("age")) else None,
                    "reporter_country": str(row.get("reporter_country", "UNKNOWN")).upper().strip(),
                    "drugs": drugs,
                    "reactions": reactions,
                    "outcomes": outc_dict.get(pid, [])
                }
                casos.append(caso)
                
            print(f"[OK] Cargados {len(casos)} reportes de pacientes unificados y listos para streaming.")
            return casos
            
        except Exception as e:
            print(f"[ERROR] Al procesar datos crudos: {e}. Pasando a generación sintética.")
            return self._generar_casos_sinteticos()

    def _generar_casos_sinteticos(self) -> list[dict]:
        """Genera casos de prueba aleatorios realistas si no hay datos crudos."""
        print("[*] Generando reportes clínicos sintéticos para pruebas de streaming...")
        farmacos = ["ASPIRIN", "METFORMIN", "IBUPROFEN", "ATORVASTATIN", "LISINOPRIL", "PENICILLIN", "OMEPRAZOLE", "WARFARIN"]
        reacciones = ["NAUSEA", "CEFALEA", "MAREO", "FATIGA", "DOLOR", "VOMITO", "EXANTEMA", "PRURITO", "DIARREA", "FALLO RENAL ACUDO", "HEMORRAGIA"]
        outcomes = ["DE", "LT", "HO", "DS", "CA", "RI", "OT"]
        paises = ["US", "ES", "CA", "GB", "DE", "FR", "JP", "MX"]
        
        casos = []
        for i in range(1000):
            pid = 9000000 + i
            caso = {
                "primaryid": pid,
                "sex": random.choice(["F", "M", "U"]),
                "age": round(random.uniform(5.0, 85.0), 1),
                "reporter_country": random.choice(paises),
                "drugs": random.sample(farmacos, random.randint(1, 3)),
                "reactions": random.sample(reacciones, random.randint(1, 2)),
                "outcomes": random.sample(outcomes, random.randint(0, 2))
            }
            casos.append(caso)
        return casos

    def stream_data(self, stop_event, delay: float = 1.5):
        """Inicia el envío de casos clínicos de forma continua."""
        casos = self._cargar_casos_reales()
        
        if not casos:
            print("[ERROR] No hay datos disponibles para el streaming.")
            return
        
        idx = 0
        total_enviados = 0
        
        # Limpiar el archivo de cola simulada para nueva ejecución
        if self.simulated:
            try:
                if SIMULATED_QUEUE_FILE.exists():
                    SIMULATED_QUEUE_FILE.unlink()
            except Exception:
                pass
        
        print(f"\n[*] INICIANDO TRANSMISIÓN en tiempo real (Retardo: {delay}s)...")
        while not stop_event.is_set():
            # Obtener el caso actual (en bucle circular)
            caso = casos[idx % len(casos)]
            idx += 1
            
            # Inyectar marca de tiempo del stream
            caso["stream_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            
            if self.simulated:
                # 1. Empujar a la cola en memoria compartida
                if self.shared_queue is not None:
                    self.shared_queue.put(caso)
                
                # 2. Escribir en el archivo JSONL simulado para persistencia
                try:
                    with open(SIMULATED_QUEUE_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(caso) + "\n")
                except Exception:
                    pass
            else:
                # Transmitir a Apache Kafka
                try:
                    self.producer.send(TOPIC_NAME, value=caso)
                except Exception as e:
                    # Si falla, guardar en la cola local
                    if self.shared_queue is not None:
                        self.shared_queue.put(caso)
            
            total_enviados += 1
            time.sleep(delay)
            
        # Cerrar producer de Kafka si existe
        if self.producer:
            try:
                self.producer.flush()
                self.producer.close()
            except Exception:
                pass
        print(f"\n[OK] Productor finalizado. Total reportes transmitidos: {total_enviados}")
