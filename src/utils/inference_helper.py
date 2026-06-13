"""
src/utils/inference_helper.py
=============================
AYUDANTE DE INFERENCIA EN VIVO DEL MODELO IA

Carga el clasificador RandomForest, codifica las características clínicas
y evalúa la predicción de severidad del paciente en tiempo real. También
se conecta con el archivo de señales de seguridad de Hadoop MapReduce.
"""

from pathlib import Path
import pandas as pd
import joblib

# Definir rutas
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH   = PROJECT_ROOT / "outputs" / "model_assets" / "modelo_severidad.joblib"
DRUG_ENC_PATH = PROJECT_ROOT / "outputs" / "model_assets" / "drug_encoder.joblib"
PT_ENC_PATH   = PROJECT_ROOT / "outputs" / "model_assets" / "pt_encoder.joblib"
SIGNALS_FILE  = PROJECT_ROOT / "outputs" / "mapreduce_results" / "top_safety_signals.csv"

class FAERSInferenceHelper:
    def __init__(self):
        self.model = None
        self.drug_encoder = None
        self.pt_encoder = None
        self.signals = {}
        self.loaded = False
        self._load_assets()

    def _load_assets(self):
        """Carga el clasificador, codificadores y CSV de MapReduce."""
        try:
            if not MODEL_PATH.exists() or not DRUG_ENC_PATH.exists() or not PT_ENC_PATH.exists():
                return
            
            self.model = joblib.load(MODEL_PATH)
            self.drug_encoder = joblib.load(DRUG_ENC_PATH)
            self.pt_encoder = joblib.load(PT_ENC_PATH)
            self._cargar_senales_mapreduce()
            self.loaded = True
        except Exception as e:
            # Si hay algún problema cargando los assets, mantener loaded = False
            pass

    def _cargar_senales_mapreduce(self):
        """Carga en memoria las señales estadísticas de MapReduce para cruce O(1)."""
        if not SIGNALS_FILE.exists():
            return
        try:
            df = pd.read_csv(SIGNALS_FILE)
            for _, row in df.iterrows():
                drug = str(row["drugname"]).upper().strip()
                pt = str(row["pt"]).upper().strip()
                self.signals[(drug, pt)] = {
                    "prr": float(row["prr"]),
                    "chi2": float(row["chi2"]),
                    "count": int(row["count_a"])
                }
        except Exception:
            pass

    def predecir_caso(self, age: float, sex: str, drug: str, pt: str) -> dict:
        """Codifica los campos y ejecuta la predicción real en el clasificador RandomForest."""
        if not self.loaded:
            return {
                "success": False,
                "error": "Los assets del modelo no están disponibles. Asegúrate de ejecutar el balanceo y entrenamiento (Opción 4 y 5)."
            }

        try:
            # 1. Limpiar y Normalizar entradas
            drug_up = str(drug).upper().strip()
            pt_up = str(pt).upper().strip()
            sex_up = str(sex).upper().strip()

            # 2. Codificar Sexo
            sex_map = {"M": 1.0, "F": 0.0, "U": 0.5}
            sex_encoded = sex_map.get(sex_up, 0.5)

            # 3. Codificar Fármaco (drugname)
            try:
                # El codificador OrdinalEncoder espera un array 2D
                drug_encoded = float(self.drug_encoder.transform([[drug_up]])[0][0])
            except Exception:
                # Si es un fármaco desconocido que el modelo no vio en entrenamiento
                drug_encoded = -1.0

            # 4. Codificar Reacción (pt)
            try:
                pt_encoded = float(self.pt_encoder.transform([[pt_up]])[0][0])
            except Exception:
                pt_encoded = -1.0

            # 5. Crear vector de características: ["drug_encoded", "pt_encoded", "sex_encoded", "age"]
            feature_vector = pd.DataFrame([{
                "drug_encoded": drug_encoded,
                "pt_encoded": pt_encoded,
                "sex_encoded": sex_encoded,
                "age": age
            }])

            # 6. Ejecutar inferencia en el RandomForest
            pred_class = int(self.model.predict(feature_vector)[0])
            probs = self.model.predict_proba(feature_vector)[0].tolist()

            # 7. Buscar si hay señal histórica de MapReduce
            key = (drug_up, pt_up)
            historical_match = self.signals.get(key)

            # Definir etiquetas explicativas para cada nivel de severidad
            labels_map = {
                1: "Leve (No requiere hospitalización)",
                2: "Moderada (Requiere intervención médica menor)",
                3: "Severa / Hospitalización (Requiere atención médica inmediata)",
                4: "Grave / Discapacidad Permanente (Produce secuelas o condiciones severas)",
                5: "Crítica / Mortalidad (Presenta riesgo vital o muerte del paciente)"
            }

            return {
                "success": True,
                "age": age,
                "sex": sex_up,
                "drug": drug_up,
                "pt": pt_up,
                "predicted_level": pred_class,
                "predicted_label": labels_map.get(pred_class, "Desconocida"),
                "probabilities": probs,
                "historical_signal": historical_match
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Fallo al procesar el caso en el modelo: {e}"
            }
