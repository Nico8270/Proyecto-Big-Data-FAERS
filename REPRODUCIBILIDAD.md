# REPRODUCIBILIDAD Y ARTEFACTOS

Este documento describe los artefactos clave generados por el pipeline, permitiendo reproducibilidad completa de los resultados descritos en el paper.

## Archivos de Configuración (Versionados en Git)

### Dependencias
- **`requirements.txt`**: Todas las librerías Python necesarias
  - pandas, numpy, scikit-learn, imbalanced-learn
  - xgboost, lightgbm (modelos de comparación)
  - joblib (serialización)
  - kafka-python, mrjob (ingesta y big data)
  - matplotlib, seaborn (visualización)

### Configuración del Proyecto
- **`src/config/`**: Configuración de Hadoop (hdfs-site.xml, etc.)
- **`.gitignore`**: Especifica qué NO se sube a Git (datos crudos, outputs grandes)

---

## Datos

### Datos de Entrada (No Versionados por Tamaño)

**Fuente Primaria:** FDA FAERS Q4 2025
- Disponible en: https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers/
- Ubicación local: `data/raw/faers/`
- Archivos requeridos:
  - DEMO25Q4.txt (Demografía - 29 MB)
  - DRUG25Q4.txt (Medicamentos - 174 MB)
  - REAC25Q4.txt (Reacciones - 43 MB)
  - OUTC25Q4.txt (Outcomes - 9 MB)
  - INDI25Q4.txt (Indicaciones - 18 MB)
  - THER25Q4.txt (Terapias - 3 MB)
  - RPSR25Q4.txt (Reportador - 65 MB)

**Alternativa: Datos de Ejemplo Coherentes**
- Ubicación: `data/sample/`
- Genera fácilmente: `python crear_sample.py`
- Contiene ~1,000 primaryid comunes a todas las tablas
- Suficiente para ejecutar el pipeline completo (~2 minutos vs ~10 minutos con datos completos)

---

## Artefactos del Pipeline (Generados)

Cada estapa del pipeline produce artefactos específicos. Estos se describen en orden de ejecución.

### Etapa 0: Configuración de Entorno
**Entrada:** `requirements.txt`
**Salida:**
- Entorno Python con todas las dependencias instaladas
- JDK 17 portable en `vendor/jdk/` (si no existe Java)

---

### Etapa 1: Análisis Exploratorio (EDA)
**Script:** `src/eda/eda_main.py`
**Entrada:** `data/raw/faers/`
**Salida:** `outputs/eda_results/`
- `s01_demo_columnas.csv` - Estadísticas de demografía
- `s02_estadisticas_edad_por_genero.csv` - Análisis por edad/sexo
- `s03_estadisticas_roles.csv`, `s03_top25_farmacos.csv` - Top medicamentos
- `s04_top25_reacciones.csv` - Reacciones adversas más comunes
- `s05_conteos_outcomes.csv` - Distribución de outcomes
- `s06_serie_mensual_reportes.csv` - Tendencia temporal
- `s07_top20_pares_farmaco_reaccion.csv` - Pares más frecuentes
- `s08_duplicados_por_tabla.csv` - QA: duplicados
- `s09_balanceo.csv`, `s09_resumen_completo.csv` - Resumen consolidado

**Importancia:** Validación de calidad de datos y distribuciones baseline.

---

### Etapa 2: Limpieza de Datos
**Script:** `src/cleaning/cleaning_main.py`
**Entrada:** `data/raw/faers/`
**Salida:** `data/clean_data/`
- `dataset_consolidado.parquet` - Tabla única con todas las variables normalizadas

**Reglas Aplicadas:**
- Deduplicación exacta por primaryid
- Imputación de edad (media: 45 años)
- Normalización de texto (UPPER, strip)
- Validación de tipos

---

### Etapa 3: MapReduce (Hadoop) - Señales de Seguridad
**Script:** `src/hadoop/hadoop_main.py`
**Entrada:** `data/clean_data/dataset_consolidado.parquet`
**Salida:** `outputs/mapreduce_results/`
- **`top_safety_signals.csv`** ← ARTEFACTO CRÍTICO (paper §3.4)
  - Columnas: drugname, pt, prr, chi2, chi2_pvalue, support_count, count_a, count_b, count_c, count_d, excess_ratio, ror
  - **Filtros aplicados:** PRR ≥ 2.0, χ² ≥ 3.84 (con corrección de Yates), soporte ≥ 3
  - **Fórmula PRR:** (a/(a+b)) / (c/(c+d))
  - **Fórmula χ²:** ((|a·d - b·c| - 0.5)² · N) / ((a+b)·(c+d)·(a+c)·(b+d))
  - Registra >13,800 pares únicos de fármaco-reacción

**Importancia:** Base de datos de señales de seguridad conocidas para comparación en streaming.

---

### Etapa 4: Balanceo de Clases
**Script:** `src/balance/balance_main.py`
**Entrada:** `data/clean_data/dataset_consolidado.parquet`
**Salida:** `data/balanced_data/{técnica}/`
- `dataset.parquet` - Datos balanceados
- Técnicas disponibles:
  - `random_undersampling/` - Submuestreo aleatorio
  - `smote/` - Sobremuestreo sintético
  - `smote_nc/` - SMOTE para features mixtas
  - `smoteenn/` - SMOTE + ENN

**Distribución Original:** 
- Clase 1 (Leve):       4,486 (22.4%)
- Clase 2 (Moderado):   3,202 (16.0%)
- Clase 3 (Importante): 3,411 (17.0%)
- Clase 4 (Grave):        901 (4.5%)
- Clase 5 (Muy Grave):  8,123 (40.5%)

**Distribución Balanceada (Random Undersampling):** ~3,000 por clase

---

### Etapa 5: Feature Engineering (Sin Data Leakage)
**Script:** `src/hadoop/training/02_feature_engineering.py`
**Entrada:** `data/balanced_data/random_undersampling/dataset.parquet`
**Salida:** `data/staging/training_features/`
- `features_train.jsonl` (12,030 casos)
- `features_test.jsonl` (3,008 casos)

**Key Methodological Artifacts:**
- **`leaking_pt_terms.json`** ← ARTEFACTO (paper §3.2, control de leakage)
  - 145 términos de resultado (PT terms) identificados como outcome-proxies
  - Criterio: Entropía H(Y|X) < 0.5 en conjunto de entrenamiento
  - Método: Split 80/20 ANTES de cualquier encoding
  - Reemplazo: Centinela "OUTCOME_PROXY" durante encoding
  - Validación adicional: Información Mutua ≥ 0.7 marcada para auditoría

**Features Generados:**
- `drug_encoded` - Ordinales [0, 1, 2, ..., K-1] de farmácos únicos
- `pt_encoded` - Ordinales de reacciones (con máscara de proxies)
- `sex_encoded` - {0.0: F, 1.0: M, 0.5: U}
- `age` - Numérico, imputado a 45 si falta

**Split Strategy:**
- Temporal: Todos los casos en Q4 2025 (no hay información de fecha para stratify temporal)
- Estratificado: Por `severity_level` (proporción de clases preservada)
- Proporción: 80% train / 20% test

---

### Etapa 6: Model Training (Comparación de 4 Modelos)
**Script:** `src/hadoop/training/03_model_training.py`
**Entrada:** `data/staging/training_features/`
**Salida:** `outputs/model_assets/`

#### Artefactos Principales

**1. `modelo_severidad.joblib`** ← ARTEFACTO CRÍTICO
- Modelo seleccionado: Random Forest (mejor CV f1_macro)
- Hiperparámetros:
  ```
  RandomForestClassifier(
    n_estimators=100,
    max_depth=16,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
  )
  ```
- Serialización: joblib con compress=3
- Test Accuracy: 68.92%
- Test F1-macro: 0.5173

**2. `model_comparison.json`** ← ARTEFACTO (paper Reproducibility)
- Tabla comparativa de 4 modelos:
  - Logistic Regression (baseline)
  - Random Forest ← MEJOR
  - XGBoost
  - LightGBM
- Métricas por modelo: CV mean/std, Test f1_macro, Test accuracy
- Metadata:
  ```json
  {
    "models": [...],
    "best_model": "Random Forest",
    "feature_cols": ["drug_encoded", "pt_encoded", "sex_encoded", "age"],
    "n_train": 12030,
    "n_test": 3008,
    "class_labels": [1, 2, 3, 4, 5],
    "smote_applied": true,
    "cv_folds": 5,
    "data_leakage_free": true
  }
  ```

**3. Encoders Entrenados**
- `drug_encoder.joblib` - OrdinalEncoder([drugnames])
- `pt_encoder.joblib` - OrdinalEncoder([pt_encoded con máscara])

**4. Visualizaciones**
- `cv_fold_scores.png` - Barras CV por modelo
- `confusion_matrices.png` - Matrices de confusión (4 modelos)
- `roc_curves.png` - Curvas ROC One-vs-Rest (macro)

#### Metodología

**Cross-Validation Pipeline:**
```python
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("model", RandomForestClassifier(...))
])

scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=StratifiedKFold(n_splits=5),
    scoring=["f1_macro", "accuracy"],
    return_train_score=True
)
```

**Key Property:** SMOTE applica SOLO dentro de cada fold de entrenamiento (previene data leakage hacia test set).

**Label Shifting:**
- Datos: classes [1, 2, 3, 4, 5]
- XGBoost requiere: [0, 1, 2, 3, 4]
- Solución: `y_shifted = y - 1` dentro de pipeline
- Reversión: Al exportar resultados, `original = predicted + 1`

---

### Etapa 7: Evaluación del Modelo
**Script:** `src/hadoop/training/04_evaluation.py`
**Entrada:** `outputs/model_assets/modelo_severidad.joblib` + test features
**Salida:** `outputs/balance_reports/model_evaluation_metrics.json`

**Métricas por Clase (Test Set Limpio):**
```
Clase 1 (Leve):       Precision 50.0%, Recall 47.5%, F1 48.7%
Clase 2 (Moderado):   Precision 47.1%, Recall 43.0%, F1 44.9%
Clase 3 (Importante): Precision 48.2%, Recall 40.0%, F1 43.8%
Clase 4 (Grave):      Precision 22.7%, Recall 32.8%, F1 26.9%
Clase 5 (Muy Grave):  Precision 92.8%, Recall 96.1%, F1 94.4% ← Mejor rendimiento
```

**Matriz de Confusión:** 5×5 con alineación correcta de clases.

---

### Etapa 8: Inferencia Batch
**Script:** `src/hadoop/training/05_inference.py`
**Entrada:**
- `outputs/model_assets/modelo_severidad.joblib`
- `outputs/model_assets/{drug,pt}_encoder.joblib`
- `outputs/model_assets/leaking_pt_terms.json`
- `data/clean_data/dataset_consolidado.parquet` (203,791 casos)
**Salida:** `data/alerts/high_risk_alerts.json`

**Criterio de Alerta:**
- Si predicted_severity ≥ 4 (Grave o Muy Grave): genera alerta
- Probabilidad de confianza calculada como max(predict_proba())

**Resultados Batch:**
- 62,621 alertas de 203,791 casos (30.7%)
- Breakdown:
  - Severidad 4 (Grave): 8,976 (14.3%)
  - Severidad 5 (Muy Grave): 53,645 (85.7%)

---

### Etapa 9: Streaming en Vivo (Kafka)
**Script:** `src/faers_kafka/main_kafka.py`
**Entrada:** Simulación o broker Kafka real
**Salida:** `data/alerts/realtime_alerts.jsonl`

**Arquitectura:**
- **Productor:** Lee casos de `simulated_queue.jsonl` (68 casos), emite con delay 1.5s
- **Consumidor:** Carga modelo ML, predice severidad por caso, genera alertas online

**Resultados Streaming:**
- 67 casos procesados → 16 alertas (23.9%)
- Consistencia con batch: ✓ (30.7% vs 23.9% con muestra pequeña)

---

### Etapa 10: Consulta Interactiva (Predicción en Vivo)
**Script:** `src/consultor_modelo.py`
**Entrada:** Perfil del paciente (edad, sexo, fármaco, reacción)
**Output:** Predicción de severidad con confianza

---

## Reproducción Garantizada

### Paso 1: Configurar Entorno
```bash
pip install -r requirements.txt
```

### Paso 2: Preparar Datos (Opción A: Rápido)
```bash
python crear_sample.py  # Genera ~1,000 casos coherentes
# O usa data/sample/ preexistente
```

### Paso 3: Ejecutar Pipeline
```bash
python src/hadoop/training/training_main.py
```

### Paso 4: Verificar Artefactos
```bash
# Todos estos archivos deben existir:
outputs/model_assets/modelo_severidad.joblib
outputs/model_assets/model_comparison.json
outputs/model_assets/drug_encoder.joblib
outputs/model_assets/pt_encoder.joblib
outputs/model_assets/leaking_pt_terms.json
outputs/model_assets/cv_fold_scores.png
outputs/model_assets/confusion_matrices.png
outputs/model_assets/roc_curves.png
outputs/mapreduce_results/top_safety_signals.csv
outputs/balance_reports/model_evaluation_metrics.json
data/alerts/high_risk_alerts.json
```

---

## Validaciones de Fidelidad al Paper

### ✅ Control de Data Leakage
- [ ] Split train/test ANTES de encoding
- [ ] 145 PT terms enmascarados (entropy < 0.5)
- [ ] SMOTE solo dentro de CV folds
- [ ] leaking_pt_terms.json serializado

### ✅ Comparación de 4 Modelos
- [ ] Logistic Regression, Random Forest, XGBoost, LightGBM
- [ ] Cross-validation 5-fold estratificado
- [ ] Métricas: CV f1_macro, Test accuracy
- [ ] model_comparison.json con metadata

### ✅ Signal Detection (MapReduce)
- [ ] PRR ≥ 2.0, χ² ≥ 3.84 (Yates), soporte ≥ 3
- [ ] >13,800 pares en top_safety_signals.csv
- [ ] Fórmulas exactas del paper

### ✅ Reproducibilidad
- [ ] Todos los artefactos versionados o regenerables
- [ ] requirements.txt completo
- [ ] Datos de ejemplo coherentes
- [ ] Scripts de regeneración documentados

---

## Referencias

- **Paper:** [Incluir DOI/arXiv cuando se publique]
- **Datos:** FDA FAERS Q4 2025
- **Metodología:**
  - Feature engineering: sklearn.preprocessing.OrdinalEncoder
  - Balanceo: imblearn.ImbPipeline, SMOTE
  - Modelos: sklearn, xgboost, lightgbm
  - Señales: PRR + χ² con corrección de Yates
  - Streaming: Kafka producer/consumer con predicción en tiempo real

