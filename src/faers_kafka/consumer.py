"""
src/kafka/consumer.py
=====================
CONSUMIDOR Y ANALIZADOR EN TIEMPO REAL FAERS

Consume los eventos del stream (Kafka o cola compartida), carga el modelo
entrenado (outputs/mapreduce_results/top_safety_signals.csv) y evalúa
cada combinación fármaco-reacción en base a gravedad clínica y significancia estadística.

CORRECCIÓN DE BUGS (inflación de alertas):
  Bug 1 — Evaluación por fila: la lógica de señal ahora corre UNA VEZ por par
           (drug, reaction) único por micro-batch, NO una vez por fila de entrada.
  Bug 2 — Deduplicación: si un par (drug, pt) ya disparó alerta en el batch
           actual, se omite completamente.
  Bug 3 — Umbral mínimo de exposición: se requieren >= MIN_CASES_THRESHOLD (3)
           co-ocurrencias del par en el batch antes de evaluar PRR/Chi2.
  Bug 4 — Umbral de Chi-square corregido: se evalúa p-value < 0.05 usando
           scipy.stats.chi2.sf en lugar de comparar el estadístico directamente.
  Bug 5 — Contadores de batch con scope local: las variables de conteo de alertas
           se crean dentro de cada invocación de procesar_micro_batch() y nunca
           se acumulan en estado global entre batches.
  Bug 6 — Alerta solo por gravedad clínica eliminada: un caso grave NO es
           suficiente para disparar una alerta de señal. Se requiere que el par
           (drug, pt) supere los umbrales PRR > 2.0 Y p-value < 0.05.
"""

import json
import os
import time
import sys
import queue
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import chi2 as chi2_dist

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

# ── Umbrales de detección de señal ────────────────────────────────────────────
# Estos valores se aplican CONJUNTAMENTE: ambos deben cumplirse para emitir alerta.
PRR_THRESHOLD       = 2.0   # Proportional Reporting Ratio mínimo
PVALUE_THRESHOLD    = 0.05  # p-value del test Chi-cuadrado (< umbral = significativo)
MIN_CASES_THRESHOLD = 3     # Mínimo de co-ocurrencias del par en el batch


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
                pt   = str(row["pt"]).upper().strip()
                senales[(drug, pt)] = {
                    "prr":     float(row["prr"]),
                    "chi2":    float(row["chi2"]),
                    "count_a": int(row["count_a"]),
                }
            print(f"[OK] Modelo de entrenamiento cargado: {len(senales):,} señales activas indexadas.")
        except Exception as e:
            print(f"{RED}[ERROR] Al cargar el modelo entrenado: {e}{RESET}")
        return senales

    def _evaluar_gravedad(self, outcomes: list) -> tuple[int, str, str]:
        """Calcula el nivel de gravedad del caso en base a los outcomes reportados."""
        outcomes_set = set(str(o).upper().strip() for o in outcomes if o)

        if "DE" in outcomes_set or "LT" in outcomes_set:
            return 4, "CRÍTICA", RED
        elif "HO" in outcomes_set or "DS" in outcomes_set or "CA" in outcomes_set:
            return 3, "ALTA", YELLOW
        elif "RI" in outcomes_set or "OT" in outcomes_set:
            return 2, "MEDIA", CYAN
        else:
            return 1, "BAJA", GREEN

    # ── Núcleo corregido: evaluación de señales por micro-batch ──────────────

    def procesar_micro_batch(self, casos: list[dict]) -> list[dict]:
        """
        Evalúa señales de seguridad para un micro-batch de N casos.

        Correcciones aplicadas:
          1. Agrega co-ocurrencias (drug, pt) a nivel de batch — no por fila.
          2. Aplica umbral mínimo MIN_CASES_THRESHOLD antes de evaluar estadísticos.
          3. Usa chi2_dist.sf(estadístico, df=1) para obtener el p-value real.
          4. Exige PRR > PRR_THRESHOLD AND p_value < PVALUE_THRESHOLD (condición conjunta).
          5. Deduplica: cada par (drug, pt) genera como máximo UNA alerta por batch.
          6. Los contadores son locales a esta función — nunca se acumulan globalmente
             entre llamadas a micro-batches distintos.

        Parámetros
        ----------
        casos : list[dict]
            Lista de registros del micro-batch. Cada registro tiene la misma
            estructura que los eventos producidos por FAERSProducer.

        Retorna
        -------
        list[dict]
            Lista de alertas generadas en este batch (puede ser vacía).
        """

        # ── Paso 1: Agregar co-ocurrencias (drug, pt) por batch ──────────────
        # Contamos cuántas veces cada par aparece en el batch.
        # Este contador es LOCAL a esta función — se resetea en cada llamada.
        pair_cases: dict[tuple, list[dict]] = defaultdict(list)

        for caso in casos:
            drugs     = caso.get("drugs", [])
            reactions = caso.get("reactions", [])
            outcomes  = caso.get("outcomes", [])

            # Normalizar a strings únicos por caso para evitar duplicados intra-caso
            drugs_unique     = list(dict.fromkeys(
                d.upper().strip() for d in drugs if d
            ))
            reactions_unique = list(dict.fromkeys(
                r.upper().strip() for r in reactions if r
            ))

            severity_level, severity_label, _ = self._evaluar_gravedad(outcomes)

            # Agregar este caso a cada par único (drug, pt) que expone
            for drug in drugs_unique:
                for pt in reactions_unique:
                    pair_cases[(drug, pt)].append({
                        "caso":           caso,
                        "severity_level": severity_level,
                        "severity_label": severity_label,
                        "outcomes":       outcomes,
                    })

        # ── Paso 2: Evaluar cada par (drug, pt) UNA SOLA VEZ ─────────────────
        alertas_batch: list[dict] = []

        # Contadores locales del batch (scope de esta función, NO globales)
        batch_total_alertas    = 0
        batch_alertas_criticas = 0
        batch_alertas_altas    = 0
        batch_alertas_conocidas = 0
        batch_alertas_nuevas   = 0

        # Conjunto de deduplicación: pares que ya generaron alerta en este batch
        # Bug 3 corregido: cada par puede emitir como máximo UNA alerta por batch
        alertas_emitidas: set[tuple] = set()

        for (drug, pt), case_list in pair_cases.items():

            # ── Bug 5 (deduplicación): saltar si ya fue procesado ─────────────
            if (drug, pt) in alertas_emitidas:
                log_info(f"[SKIP-DUP] {drug} + {pt}: ya procesado en este batch.")
                continue

            # ── Bug 3 (umbral mínimo): requerir >= MIN_CASES_THRESHOLD ────────
            n_cases = len(case_list)
            if n_cases < MIN_CASES_THRESHOLD:
                log_info(
                    f"[SKIP-N] {drug} + {pt}: solo {n_cases} caso(s) "
                    f"(mínimo requerido: {MIN_CASES_THRESHOLD})."
                )
                continue

            # ── Buscar en el modelo histórico ─────────────────────────────────
            match = self.modelo_senales.get((drug, pt))

            prr          = None
            chi2_stat    = None
            p_value      = None
            count_a_hist = 0
            is_known_signal = False

            if match:
                prr          = match["prr"]
                chi2_stat    = match["chi2"]
                count_a_hist = match["count_a"]

                # ── Bug 4 corregido: usar p-value real, NO el estadístico ─────
                # chi2.sf(x, df) = P(Chi2 > x) = p-value de cola derecha con 1 gl
                p_value = float(chi2_dist.sf(chi2_stat, df=1))

                # ── Bug 6 corregido: condición conjunta PRR Y p-value ─────────
                # Antes: PRR >= 2.0 AND chi2_stat >= 3.84 (estadístico, no p-value)
                # Ahora: PRR > PRR_THRESHOLD AND p_value < PVALUE_THRESHOLD
                is_known_signal = (prr > PRR_THRESHOLD) and (p_value < PVALUE_THRESHOLD)

            # ── Decisión de alerta basada en señal estadística ────────────────
            # Bug 6 corregido: ya NO se dispara alerta solo por gravedad clínica.
            # La gravedad del caso se registra en el payload de la alerta para
            # contexto clínico, pero NO es condición suficiente.
            if not is_known_signal:
                pv_str = f"{p_value:.4f}" if p_value is not None else "N/A"
                log_info(
                    f"[Control] {drug} + {pt}: Sin señal activa "
                    f"(PRR: {prr}, p-value: {pv_str}, "
                    f"n_batch: {n_cases})"
                )
                continue

            # ── Emitir UNA alerta representativa del par ──────────────────────
            # Usamos el caso de mayor gravedad como representante del par
            rep_case_info = max(case_list, key=lambda x: x["severity_level"])
            rep_caso      = rep_case_info["caso"]
            sev_level     = rep_case_info["severity_level"]
            sev_label     = rep_case_info["severity_label"]

            alert_type = "SEÑAL_SEGURIDAD_CONOCIDA"
            batch_alertas_conocidas += 1

            alert_item = {
                "alert_timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                "primaryid":        rep_caso.get("primaryid", "BATCH"),
                "sex":              rep_caso.get("sex", "U"),
                "age":              rep_caso.get("age", "N/A"),
                "reporter_country": rep_caso.get("reporter_country", "UNKNOWN"),
                "drugname":         drug,
                "pt":               pt,
                "n_cases_in_batch": n_cases,
                "severity_level":   sev_level,
                "severity_label":   sev_label,
                "is_known_signal":  True,
                "prr":              prr,
                "chi2_stat":        chi2_stat,
                "p_value":          round(p_value, 6) if p_value is not None else None,
                "count_a_historical": count_a_hist,
                "alert_type":       alert_type,
                "alert_reason":     (
                    f"PRR={prr:.2f} > {PRR_THRESHOLD} y "
                    f"p-value={p_value:.4f} < {PVALUE_THRESHOLD} "
                    f"({n_cases} casos en batch)"
                ),
                "outcomes":         rep_case_info["outcomes"],
            }

            alertas_batch.append(alert_item)
            alertas_emitidas.add((drug, pt))
            batch_total_alertas += 1

            if sev_level == 4:
                batch_alertas_criticas += 1
            elif sev_level == 3:
                batch_alertas_altas += 1

            log_info(
                f"[ALERTA] {drug} + {pt} | PRR={prr:.2f} | "
                f"p={p_value:.4f} | n_batch={n_cases} | {sev_label}"
            )

        # ── Actualizar estadísticas globales de sesión con delta del batch ────
        # Los deltas son locales; se suman a self.stats para el panel en tiempo real
        self.stats["total_consumidos"]  += len(casos)
        self.stats["total_alertas"]     += batch_total_alertas
        self.stats["alertas_criticas"]  += batch_alertas_criticas
        self.stats["alertas_altas"]     += batch_alertas_altas
        self.stats["alertas_conocidas"] += batch_alertas_conocidas
        self.stats["alertas_nuevas"]    += batch_alertas_nuevas

        # ── Persistir alertas a disco ─────────────────────────────────────────
        if alertas_batch:
            try:
                with open(ALERTS_FILE, "a", encoding="utf-8") as f:
                    for alert in alertas_batch:
                        f.write(json.dumps(alert) + "\n")
            except Exception as e:
                print(f"  [ERROR] No se pudo escribir alerta a disco: {e}")

        return alertas_batch

    # ── procesar_caso: wrapper por compatibilidad con consume_data ────────────

    def procesar_caso(self, caso: dict):
        """
        Wrapper de un único caso para el bucle de consumo individual.
        Internamente agrupa el caso en un micro-batch de 1 elemento.
        NOTA: para mayor eficiencia, el bucle de Kafka debería acumular
        registros y llamar a procesar_micro_batch() con N > 1.
        """
        self.procesar_micro_batch([caso])

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
