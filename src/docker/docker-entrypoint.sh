#!/bin/bash
set -e

echo "========================================================================"
echo "  FAERS Hadoop Pipeline – Docker Entrypoint"
echo "========================================================================"

# 1. Configurar SSH
echo "[1/5] Configurando SSH..."
mkdir -p /root/.ssh
chmod 700 /root/.ssh
ssh-keygen -t rsa -P "" -f /root/.ssh/id_rsa -q
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 2. Formatear HDFS (solo la primera vez)
echo "[2/5] Inicializando HDFS..."
if [ ! -d /tmp/hadoop/dfs/name/current ]; then
    echo "  Formateando namenode..."
    hdfs namenode -format -noninteractive -force
else
    echo "  HDFS ya formateado, skip."
fi

# 3. Iniciar demonios Hadoop
echo "[3/5] Iniciando demonios Hadoop..."
start-dfs.sh
start-yarn.sh

# Esperar a que los servicios estén listos
echo "  Esperando 10 segundos para que los servicios inicien..."
sleep 10

# 4. Crear directorios en HDFS
echo "[4/5] Creando directorios en HDFS..."
hdfs dfs -mkdir -p /input/faers
hdfs dfs -mkdir -p /output

# 5. Copiar datos a HDFS
echo "  Subiendo archivos FAERS a HDFS..."
hdfs dfs -put -f /workspace/data/*.txt /input/faers/ 2>/dev/null || true

# Listar archivos en HDFS
echo "  Archivos en HDFS:"
hdfs dfs -ls /input/faers/

# 6. Ejecutar pipeline si se pasa argumento --run
if [ "$1" = "--run" ] || [ "$1" = "-r" ]; then
    echo "[5/5] Ejecutando pipeline MapReduce..."
    cd /workspace
    python3 hadoop/run_faers_pipeline.py --hadoop
else
    echo ""
    echo "[INFO] Para ejecutar el pipeline, usa:"
    echo "  docker exec -it <container_id> python3 hadoop/run_faers_pipeline.py --hadoop"
    echo ""
    echo "  O ejecuta jobs individuales:"
    echo "  python3 hadoop/run_count_by_country.py --hadoop"
    echo "  python3 hadoop/run_top_drugs.py --hadoop"
    echo "  python3 hadoop/run_top_reactions.py --hadoop"
    echo ""
fi

# Mantener contenedor vivo
echo "========================================================================"
echo "  Hadoop listo. Services: HDFS (9000), YARN (8088)"
echo "========================================================================"
echo ""

# Ejecutar tail para mantener contenedor corriendo
if [ "$1" != "--run" ]; then
    tail -f /dev/null
fi
