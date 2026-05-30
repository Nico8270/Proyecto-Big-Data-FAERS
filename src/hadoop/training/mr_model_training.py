"""
src/hadoop/training/mr_model_training.py
===========================================
JOB DE MAPREDUCE: MRModelTraining

Este script implementa el Job de MapReduce utilizando la librería mrjob.
Realiza un escaneo distribuido masivo de las variables de interés (fármacos, reacciones y combinaciones).

Mapper:
  - Lee el archivo JSONL de features.
  - Para cada registro, emite las claves de conteo para:
    * Total de registros en la base de datos ("TOTAL:all")
    * Cada fármaco individual ("DRUG:nombre_farmaco")
    * Cada reacción individual ("REAC:nombre_reaccion")
    * Cada pareja fármaco-reacción ("PAIR:nombre_farmaco@nombre_reaccion")

Reducer:
  - Suma todos los valores (conteo total) para cada clave.
  - Emite una estructura JSON limpia conteniendo la clave y su respectiva frecuencia agregada
    gracias al protocolo JSONValueProtocol.
"""

import json
import sys
import shlex
import types

# Polyfill para compatibilidad con Python 3.13 / 3.14 (módulo 'pipes' eliminado en PEP 594)
if "pipes" not in sys.modules:
    pipes_module = types.ModuleType("pipes")
    pipes_module.quote = shlex.quote
    sys.modules["pipes"] = pipes_module

from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol

class MRModelTraining(MRJob):
    # Salida directa en formato JSON Lines para facilitar el postprocesamiento
    OUTPUT_PROTOCOL = JSONValueProtocol

    def mapper(self, _, line):
        try:
            line = line.strip()
            if not line:
                return
            record = json.loads(line)
            
            drugname = record.get("drugname")
            pt       = record.get("pt")

            if drugname and pt:
                # 1. Contador del total general de reportes (N_total)
                yield "TOTAL:all", 1
                
                # 2. Contador marginal del fármaco (N_drug)
                yield f"DRUG:{drugname}", 1
                
                # 3. Contador marginal de la reacción (N_reaction)
                yield f"REAC:{pt}", 1
                
                # 4. Contador de la co-ocurrencia del par fármaco-reacción (N_pair)
                yield f"PAIR:{drugname}@{pt}", 1
        except Exception:
            # Ignorar líneas corruptas de manera segura en Big Data
            pass

    def reducer(self, key, values):
        total_count = sum(values)
        # Yield estructurado para JSONValueProtocol
        yield None, {"key": key, "count": total_count}

if __name__ == "__main__":
    MRModelTraining.run()
