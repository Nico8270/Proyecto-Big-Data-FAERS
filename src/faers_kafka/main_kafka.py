"""
src/kafka/main_kafka.py
=======================
ORQUESTRADOR DEL PIPELINE DE STREAMING REAL-TIME CON KAFKA

Lanza de forma concurrente en hilos el Productor (simulado o real) y el Consumidor.
Muestra reportes en tiempo real y permite la finalización limpia del pipeline.
"""

import sys
import time
import queue
import threading
from pathlib import Path

# Determinar PROJECT_ROOT y resolverlo
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from faers_kafka.config import SIMULATION_DELAY, ALERTS_FILE
from faers_kafka.producer import FAERSProducer
from faers_kafka.consumer import FAERSConsumer

# Códigos ANSI para visualización
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
RED     = '\033[91m'
CYAN    = '\033[96m'
BOLD    = '\033[1m'
RESET   = '\033[0m'

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

def mostrar_cabecera():
    print(f"\n{CYAN}{BOLD}" + "=" * 80)
    print("   FAERS REAL-TIME PHARMACOVIGILANCE STREAMING PLATFORM WITH KAFKA")
    print("=" * 80 + f"{RESET}")
    print("  Este módulo ingesta reportes de reacciones adversas en tiempo real,")
    print("  las evalúa con el modelo MapReduce y genera alertas automáticas.\n")
    print(f"  * Retardo de simulación: {SIMULATION_DELAY}s")
    print(f"  * Archivo de Logs      : logs/streaming_pipeline.log")
    print(f"  * Salida de alertas    : {ALERTS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  * {BOLD}Presione Ctrl+C en cualquier momento para detener la ejecución.{RESET}")
    print("-" * 80 + "\n")

def main():
    mostrar_cabecera()
    
    # Cola compartida en memoria para el modo simulación
    shared_queue = queue.Queue()
    stop_event = threading.Event()
    
    # Instanciar Productor y Consumidor
    producer = FAERSProducer(shared_queue=shared_queue)
    consumer = FAERSConsumer(shared_queue=shared_queue)
    
    modo = f"{RED}{BOLD}MODO SIMULACIÓN LOCAL (Sin broker de Kafka){RESET}" if producer.simulated else f"{GREEN}{BOLD}MODO KAFKA ACTIVO (Conexión exitosa){RESET}"
    print(f"\n[*] Iniciando ejecución en: {modo}\n")
    time.sleep(2.0)
    
    # Crear e iniciar hilos concurrentes
    prod_thread = threading.Thread(
        target=producer.stream_data, 
        args=(stop_event, SIMULATION_DELAY), 
        daemon=True,
        name="ProducerThread"
    )
    
    cons_thread = threading.Thread(
        target=consumer.consume_data, 
        args=(stop_event,), 
        daemon=True,
        name="ConsumerThread"
    )
    
    prod_thread.start()
    cons_thread.start()
    
    # Registrar tiempo de inicio
    t_start = time.time()
    
    import os
    try:
        # Bucle principal para mostrar el panel de estadísticas acumuladas en tiempo real (actualizando en consola in-place)
        while not stop_event.is_set():
            time.sleep(0.5)
            
            stats = consumer.stats
            duracion = time.time() - t_start
            
            # Limpiar terminal para lograr efecto de refresco en tiempo real
            os.system("cls" if os.name == "nt" else "clear")
            
            mostrar_cabecera()
            
            modo = f"{RED}{BOLD}MODO SIMULACIÓN LOCAL (Sin broker de Kafka){RESET}" if producer.simulated else f"{GREEN}{BOLD}MODO KAFKA ACTIVO (Conexión exitosa){RESET}"
            print(f"  [*] Estado del Stream : {modo}")
            print(f"  [*] Archivo de Logs   : logs/streaming_pipeline.log")
            print(f"  [*] Salida de Alertas : {ALERTS_FILE.relative_to(PROJECT_ROOT)}")
            print(f"  [*] Control           : {BOLD}Presione Ctrl+C para detener y volver al menú.{RESET}")
            
            print(f"\n{CYAN}{BOLD}" + "─" * 80 + f"{RESET}")
            print(f"{CYAN}{BOLD}   PANEL DE CONTROL DE STREAMING EN VIVO (Activo: {duracion:.1f}s){RESET}")
            print(f"{CYAN}{BOLD}" + "─" * 80 + f"{RESET}")
            print(f"   • Casos Procesados : {stats['total_consumidos']:,}")
            print(f"   • Alertas Emitidas : {RED if stats['total_alertas'] > 0 else GREEN}{stats['total_alertas']:,}{RESET}")
            print(f"     ├── {RED}Críticas (Muerte / Riesgo Vital) : {stats['alertas_criticas']:,}{RESET}")
            print(f"     ├── {YELLOW}Altas (Hospitalización / Congénito) : {stats['alertas_altas']:,}{RESET}")
            print(f"     ├── {BOLD}Conocidas (Modelo MapReduce Activo) : {stats['alertas_conocidas']:,}{RESET}")
            print(f"     └── {BOLD}Nuevas Asociaciones Críticas       : {stats['alertas_nuevas']:,}{RESET}")
            print(f"{CYAN}{BOLD}" + "─" * 80 + f"{RESET}\n")
            
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}[*] Deteniendo pipeline de streaming... Por favor, espere.{RESET}")
        stop_event.set()
        
        # Dar un tiempo razonable para que los hilos terminen sus tareas
        prod_thread.join(timeout=3.0)
        cons_thread.join(timeout=3.0)
        
        # Resumen final de la ejecución
        duracion_final = time.time() - t_start
        stats_final = consumer.stats
        print(f"\n{GREEN}{BOLD}" + "=" * 80)
        print("   RESUMEN FINAL DE LA EJECUCIÓN DEL STREAMING")
        print("=" * 80 + f"{RESET}")
        print(f"  Duración total  : {duracion_final:.1f} segundos")
        print(f"  Casos leídos    : {stats_final['total_consumidos']:,}")
        print(f"  Alertas totales : {stats_final['total_alertas']:,}")
        print(f"    - Críticas   : {stats_final['alertas_criticas']:,}")
        print(f"    - Altas      : {stats_final['alertas_altas']:,}")
        print(f"    - Conocidas  : {stats_final['alertas_conocidas']:,}")
        print(f"    - Nuevas     : {stats_final['alertas_nuevas']:,}")
        print(f"{GREEN}{BOLD}" + "=" * 80 + f"{RESET}\n")
        
        sys.exit(0)

if __name__ == "__main__":
    main()
