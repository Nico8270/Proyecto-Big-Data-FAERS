"""
src/hadoop/s02_mapreduce_join.py
==================================
Implementación de un Reduce-Side Join usando mrjob.

Mapper:
  Lee cada línea del JSONL de entrada, la decodifica y emite (str(primaryid), registro).

Reducer:
  Agrupa los registros por su tabla origen ("table").
  Verifica si están presentes las 7 tablas esperadas (DEMO, DRUG, REAC, OUTC, INDI, THER, RPSR).
  Realiza el producto cartesiano de los registros de todas las tablas para ese primaryid.
  Fusiona los campos, añadiendo sufijos en caso de conflicto de nombres de columnas.
  Emite el registro consolidado final.
"""

import json
import itertools
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

# Lista de tablas requeridas para el Inner Join
ACTIVE_TABLES = ["DEMO", "DRUG", "REAC", "OUTC", "INDI", "THER", "RPSR"]

class MRJoinJob(MRJob):
    # Usar JSONValueProtocol para que la salida sea directamente JSON Lines sin llaves de MapReduce
    OUTPUT_PROTOCOL = JSONValueProtocol

    def mapper(self, _, line):
        try:
            line = line.strip()
            if not line:
                return
            record = json.loads(line)
            primaryid = record.get("primaryid")
            if primaryid is not None:
                yield str(primaryid), record
        except Exception:
            # Ignorar líneas corruptas de forma segura
            pass

    def reducer(self, primaryid, records):
        # Agrupar registros por la tabla de origen
        grouped = {}
        for record in records:
            table = record.get("table")
            if table:
                grouped.setdefault(table, []).append(record)

        # Si falta alguna de las tablas requeridas, omitir (Inner Join)
        if any(table not in grouped for table in ACTIVE_TABLES):
            return

        # Obtener las listas de registros en el orden especificado en ACTIVE_TABLES
        lists_to_product = [grouped[table] for table in ACTIVE_TABLES]

        # Producto cartesiano de los registros
        for combo in itertools.product(*lists_to_product):
            # combo[0] es el registro de DEMO. Iniciamos el merge desde ahí.
            merged = {}
            
            # Copiar campos de DEMO
            demo_rec = combo[0]
            for k, v in demo_rec.items():
                if k != "table":
                    merged[k] = v

            # Fusionar secuencialmente las demás tablas
            for i in range(1, len(ACTIVE_TABLES)):
                table_name = ACTIVE_TABLES[i]
                table_rec = combo[i]
                table_suffix = f"_{table_name.lower()}"

                for k, v in table_rec.items():
                    # No mezclar el tag 'table' y mantener 'primaryid' sin sufijos
                    if k in ("table", "primaryid"):
                        continue
                    
                    if k in merged:
                        merged[f"{k}{table_suffix}"] = v
                    else:
                        merged[k] = v

            # Emitir el registro combinado final
            yield None, merged

if __name__ == "__main__":
    MRJoinJob.run()
