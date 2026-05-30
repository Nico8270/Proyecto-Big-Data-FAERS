# FAERS Pharmacovigilance & Big Data Analytics Platform

Una plataforma avanzada e integrada de Big Data y Machine Learning para procesar, limpiar, analizar e inferir señales de seguridad clínica a gran escala a partir del **FDA Adverse Event Reporting System (FAERS)**.

---

## 1. Descripción del Proyecto

Este proyecto implementa un pipeline robusto de punta a punta (End-to-End) diseñado para procesar millones de reportes de reacciones adversas a medicamentos proporcionados oficialmente por la FDA (Food and Drug Administration) de EE.UU. 

La plataforma unifica tecnologías clave de la ingeniería de datos moderna y la ciencia de datos en una única interfaz interactiva:

*   **Ingesta y Análisis Exploratorio (EDA):** Scripts optimizados en español con barras de progreso detalladas en consola que analizan la demografía de los pacientes, medicamentos más consumidos, reacciones adversas y tendencias temporales, generando reportes estadísticos y gráficos de alta calidad.
*   **Limpieza y Consolidación de Datos:** Procesamiento modular de las distintas tablas relacionales de FAERS (Demografía, Drogas, Reacciones, Indicaciones, Terapias, Outcomes, etc.) aplicando reglas estrictas de imputación y normalización.
*   **Big Data con MapReduce (Hadoop):** Un sistema que ejecuta un Join distribuido localmente a través de la librería `mrjob` y un JDK portable de Java (autodescargable), procesando y cruzando los conjuntos de datos masivos para calcular señales de alerta de seguridad en fármacos.
*   **Balanceo de Clases:** Módulos avanzados para mitigar el desbalance severo en datos clínicos mediante técnicas de submuestreo y sobremuestreo (SMOTE, etc.).
*   **Machine Learning a Gran Escala:** Entrenamiento de clasificadores basados en ensamblados de árboles de decisión (`RandomForest`) para predecir la severidad y el desenlace (Outcome) de una reacción adversa según el perfil clínico del paciente.
*   **Streaming en Vivo (Kafka):** Un panel de streaming en vivo interactivo en consola que simula la llegada en tiempo real de nuevos reportes de pacientes y procesa de forma concurrente telemetría clínica.
*   **Motor de Inferencia Clínica Avanzada:** Un consultor interactivo que permite ingresar perfiles de pacientes en tiempo real y evalúa la probabilidad de severidad del caso clínico cargando los encoders y assets generados por el modelo de IA.

---

## 2. Manual de Uso (Desarrollado para Windows)

El proyecto ha sido optimizado específicamente para ejecutarse de forma portátil y nativa en entornos **Windows 10/11** sin requerir complejas instalaciones globales de Hadoop, Spark o Java en el sistema operativo.

### Paso 1: Requisitos Previos en Windows
1.  **Python 3.10 o superior** instalado en Windows.
    > [!IMPORTANT]
    > Al instalar Python en Windows, asegúrate de marcar la casilla **"Add Python to PATH"** en el instalador.
2.  Un terminal de Windows (PowerShell, CMD o Windows Terminal).

---

### Paso 2: Obtención e Instalación del Dataset (FAERS)
Los datos de origen provienen directamente del sistema oficial de la FDA de Estados Unidos (**FAERS**). Debido al tamaño de los archivos (cientos de megabytes), el dataset de origen **no** se sube al repositorio de Git y debe ser provisto manualmente:

1.  **Descargar los datos:** Visita el portal oficial de descarga de datos de la FDA:
    👉 [FDA FAERS Electronic Submissions and Raw Data](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files)
2.  Descarga el paquete de datos en formato **ASCII** correspondiente al periodo que desees utilizar (este proyecto está configurado por defecto para el cuarto trimestre de 2025, **2025Q4**).
3.  **Colocar los archivos:** Extrae el archivo ZIP descargado y coloca los archivos `.txt` planos directamente dentro de la siguiente estructura en el proyecto:
    `data/raw/faers/`
    
    Dentro de esta carpeta deben quedar los siguientes archivos clave extraídos:
    *   `DEMO25Q4.txt` (Demografía)
    *   `DRUG25Q4.txt` (Medicamentos)
    *   `REAC25Q4.txt` (Reacciones)
    *   `INDI25Q4.txt` (Indicaciones médicas)
    *   `OUTC25Q4.txt` (Resultados/Outcomes clínicos)
    *   `THER25Q4.txt` (Terapias)
    *   `RPSR25Q4.txt` (Datos del reportador)

---

### Paso 3: Ejecución de la Plataforma
El proyecto cuenta con un orquestador en Windows que ejecuta todo de forma controlada y automatizada.

1.  Abre un terminal en la carpeta raíz del proyecto.
2.  Ejecuta el archivo por lotes de Windows:
    ```cmd
    run.bat
    ```
    *(Este comando iniciará el menú interactivo estableciendo el encoding UTF-8 en consola para mostrar barras de progreso y cajas dinámicas sin problemas de codificación CP1252).*

---

### Paso 4: Guía del Menú Interactivo en Consola

Al iniciar `run.bat`, verás la interfaz de control principal con 9 opciones. Sigue este orden recomendado para la primera ejecución:

#### 🔹 [0] Configurar entorno y descargar tecnologías (¡Ejecutar Primero!)
Instala automáticamente todas las librerías necesarias especificadas en `requirements.txt` a través de `pip`.
> [!TIP]
> **Hadoop Local Portable en Windows:** Esta opción detectará si tienes Java instalado. Si no lo tienes, **descargará de forma 100% automatizada e instalará un JDK 17 portable de Eclipse Adoptium** en la carpeta `vendor/jdk/`. El software configura las variables de entorno `JAVA_HOME` y `PATH` en memoria, permitiendo correr MapReduce y Hadoop localmente en Windows sin configurar nada de forma manual en tu PC.

#### 🔹 [1] Analisis exploratorio (EDA)
Ejecuta secuencialmente los 9 scripts de análisis exploratorio (`s01` a `s09`). Genera gráficos interactivos en PDF/PNG y reportes analíticos consolidados dentro de la carpeta `outputs/eda_results/`.

#### 🔹 [2] Limpiar datos
Procesa y remueve duplicados, imputa campos vacíos y limpia anomalías en las tablas crudas de la FDA, depositando los archivos optimizados listos para Big Data en `data/clean_data/`.

#### 🔹 [3] Join con Hadoop y MapReduce
Ejecuta la unión distribuida en tu máquina local. Utiliza el framework de MapReduce a través de `mrjob` y el JDK portable para procesar de forma masiva millones de registros y extraer las señales de seguridad de fármacos más críticas en `outputs/mapreduce_results/top_safety_signals.csv`.

#### 🔹 [4] Balancear datos
Genera conjuntos de entrenamiento equilibrados en `data/balanced_data/` para mitigar el sesgo hacia desenlaces leves o raros, lo cual es fundamental para obtener una buena precisión en los modelos de Machine Learning.

#### 🔹 [5] Entrenar modelo con Hadoop MapReduce
Entrena un clasificador `RandomForest` robusto a partir de los datos procesados y balanceados. Al finalizar, exporta los encoders de las variables clínicas y el binario serializado del modelo final entrenado dentro de `outputs/model_assets/`.

#### 🔹 [6] Correr modelo con Kafka (Streaming en Vivo)
Inicia un panel dinámico de visualización de eventos de farmacovigilancia por streaming que refresca estadísticas en tiempo real y calcula métricas clave en la terminal en intervalos de 500ms, enviando diagnósticos detallados en paralelo al archivo de logs `/logs/streaming_pipeline.log`.

#### 🔹 [7] Consultar modelo con caso clínico (Predicción ML)
Un motor predictivo en tiempo real donde puedes ingresar perfiles de pacientes en formato de lista (ejemplo: `[65, F, LISINOPRIL 5MG, HYPERKALAEMIA]`). El programa carga los assets previamente entrenados de `outputs/model_assets/`, decodifica la información clínica e infiere instantáneamente la severidad del reporte en pantalla con métricas de probabilidad y una explicación detallada del caso.

#### 🔹 [8] Salir
Cierra de forma segura el entorno y finaliza los procesos.