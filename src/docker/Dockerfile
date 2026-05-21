# Dockerfile: Entorno Hadoop + Python para FAERS MapReduce
# Basado en la imagen oficial de Hadoop

FROM ubuntu:22.04

ENV HADOOP_VERSION=3.3.6
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

# Evitar prompts durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# 1. Instalar Java y dependencias
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    python3 \
    python3-pip \
    curl \
    wget \
    ssh \
    rsync \
    net-tools \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 2. Descargar e instalar Hadoop
WORKDIR /opt
RUN wget https://downloads.apache.org/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz \
    && tar -xzf hadoop-${HADOOP_VERSION}.tar.gz \
    && mv hadoop-${HADOOP_VERSION} hadoop \
    && rm hadoop-${HADOOP_VERSION}.tar.gz

# 3. Configurar Hadoop (modo pseudo-distribuido)
COPY hadoop-config/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml
COPY hadoop-config/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml
COPY hadoop-config/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml
COPY hadoop-config/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml

# 4. Instalar dependencias Python
COPY requirements_hadoop.txt /tmp/
RUN pip3 install -r /tmp/requirements_hadoop.txt

# 5. Configurar SSH local (para Hadoop)
RUN ssh-keygen -t rsa -P "" -f /root/.ssh/id_rsa \
    && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys \
    && chmod 0600 /root/.ssh/authorized_keys

# 6. Crear directorios de trabajo
WORKDIR /workspace
RUN mkdir -p /workspace/data /workspace/scripts /workspace/outputs

# 7. Copiar scripts del proyecto
COPY scripts/ /workspace/scripts/
COPY data/ /workspace/data/
COPY hadoop/ /workspace/hadoop/

# 8. Puerto expuestos
EXPOSE 9870 8088 50070 50075

# 9. Script de entrada
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
