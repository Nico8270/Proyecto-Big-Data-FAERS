"""
src/utils/interactive_menu.py
=============================
MENÚ INTERACTIVO DE CONSULTA Y PREDICCIÓN CLÍNICA EN VIVO

Muestra una interfaz interactiva de consulta de severidad de casos
para demostrar la inteligencia artificial (RandomForest) del proyecto.
"""

import sys
import time
import re
from pathlib import Path

# Registrar la carpeta 'src' en sys.path para resolver las importaciones correctamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.inference_helper import FAERSInferenceHelper

# Códigos ANSI para dar estética premium en consola
RED     = '\033[91m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
BLUE    = '\033[94m'
CYAN    = '\033[96m'
BOLD    = '\033[1m'
RESET   = '\033[0m'

def safe_print(*args, **kwargs):
    text = " ".join(str(arg) for arg in args)
    try:
        sys.stdout.write(text + kwargs.get("end", "\n"))
        sys.stdout.flush()
    except UnicodeEncodeError:
        text_safe = text.replace("🚨", "[CRITICO]").replace("⚠️", "[ALERTA]").replace("─", "-").replace("•", "*").replace("🔴", "*")
        try:
            sys.stdout.write(text_safe + kwargs.get("end", "\n"))
            sys.stdout.flush()
        except UnicodeEncodeError:
            sys.stdout.write(text_safe.encode('ascii', errors='ignore').decode('ascii') + kwargs.get("end", "\n"))
            sys.stdout.flush()

def mostrar_cabecera_consulta():
    safe_print(f"\n{CYAN}{BOLD}" + "=" * 80)
    safe_print("   FAERS CLINICAL INFERENCE PLATFORM (Interactive ML Model Explorer)")
    safe_print("=" * 80 + f"{RESET}")
    safe_print("  Esta herramienta evalúa en tiempo real el nivel de gravedad clínica (1 al 5)")
    safe_print("  de un paciente usando el RandomForest entrenado sobre datos balanceados.\n")

def parsear_entrada_caso(entrada: str) -> tuple:
    """Intenta extraer [edad, sexo, farmaco, reaccion] desde una entrada tipo array o cadena."""
    # Intentar limpiar la cadena de corchetes, comillas y espacios externos
    limpio = entrada.strip()
    if limpio.startswith("[") and limpio.endswith("]"):
        limpio = limpio[1:-1]
    
    # Separar por comas
    partes = [p.strip() for p in limpio.split(",")]
    if len(partes) != 4:
        return None
    
    try:
        age = float(partes[0])
        sex = partes[1].upper().replace("'", "").replace('"', '').strip()
        drug = partes[2].upper().replace("'", "").replace('"', '').strip()
        pt = partes[3].upper().replace("'", "").replace('"', '').strip()
        return age, sex, drug, pt
    except Exception:
        return None

def solicitar_caso_manualmente() -> tuple:
    """Solicita los datos del paciente de forma guiada uno por uno."""
    safe_print(f"\n{YELLOW}[*] Entrada estructurada no detectada. Iniciando formulario guiado...{RESET}\n")
    
    # 1. Solicitar Edad
    while True:
        try:
            edad_in = input("  • Ingrese la Edad del paciente (0 a 100): ").strip()
            age = float(edad_in)
            if 0 <= age <= 120:
                break
            safe_print(f"    {RED}[ERROR] Por favor ingrese una edad realista (0-120).{RESET}")
        except ValueError:
            safe_print(f"    {RED}[ERROR] Ingrese un número válido para la edad.{RESET}")
            
    # 2. Solicitar Sexo
    while True:
        sex_in = input("  • Ingrese el Sexo (F = Femenino, M = Masculino, U = Desconocido): ").strip().upper()
        if sex_in in ["M", "F", "U"]:
            sex = sex_in
            break
        safe_print(f"    {RED}[ERROR] Opción no válida. Ingrese F, M o U.{RESET}")

    # 3. Solicitar Fármaco
    while True:
        drug = input("  • Ingrese el Nombre del Fármaco sospechoso: ").strip().upper()
        if drug:
            break
        safe_print(f"    {RED}[ERROR] El nombre del fármaco no puede estar vacío.{RESET}")

    # 4. Solicitar Reacción
    while True:
        pt = input("  • Ingrese el Término de Reacción Adversa (PT): ").strip().upper()
        if pt:
            break
        safe_print(f"    {RED}[ERROR] El término de la reacción no puede estar vacío.{RESET}")

    return age, sex, drug, pt

def run_interactive_inference():
    """Ejecuta el bucle de interacción de consulta del modelo."""
    mostrar_cabecera_consulta()
    
    # 1. Cargar el helper de inferencia
    safe_print(f"{YELLOW}[*] Inicializando assets del modelo... Por favor, espera.{RESET}")
    helper = FAERSInferenceHelper()
    
    if not helper.loaded:
        safe_print(f"\n{RED}[ERROR] No se pudieron cargar los assets del modelo de Machine Learning.{RESET}")
        safe_print(f"        Asegúrate de haber corrido las opciones anteriores del menú:")
        safe_print(f"        -> Opción 4: Balancear datos (SMOTE+ENN)")
        safe_print(f"        -> Opción 5: Entrenar modelo con Hadoop MapReduce")
        safe_print(f"\n  Presione Enter para volver al menú principal...")
        input()
        return

    safe_print(f"{GREEN}[OK] Assets de Machine Learning y MapReduce cargados correctamente.{RESET}\n")
    
    # 2. Mostrar ejemplos
    safe_print(f"{CYAN}{BOLD}📋 CASOS CLÍNICOS DE EJEMPLO (Copia y pega la línea completa):{RESET}")
    safe_print(f"  {BOLD}Caso A (Alta Severidad)  →{RESET}  [65, F, LISINOPRIL 5MG PO QD, HYPERKALAEMIA]")
    safe_print(f"  {BOLD}Caso B (Media Severidad) →{RESET}  [48, M, EMPAGLIFLOZIN, BLOOD CREATININE INCREASED]")
    safe_print(f"  {BOLD}Caso C (Baja Severidad)  →{RESET}  [32, U, METFORMIN, NAUSEA]")
    safe_print(f"{CYAN}" + "─" * 80 + f"{RESET}")
    
    # 3. Solicitar entrada
    safe_print(f"\n{CYAN}{BOLD}✍️ ENTRADA DE DATOS:{RESET}")
    safe_print("  Ingrese el caso en formato [Edad, Sexo(F/M/U), Fármaco, Reacción]")
    safe_print("  o presione ENTER para usar el Caso A de ejemplo por defecto:")
    
    entrada = input(f"\n  {BOLD}> Ingrese su entrada:{RESET} ").strip()
    
    if not entrada:
        # Usar caso A por defecto
        entrada = "[65, F, LISINOPRIL 5MG PO QD, HYPERKALAEMIA]"
        safe_print(f"  [*] Usando Caso A por defecto: {entrada}")

    parsed = parsear_entrada_caso(entrada)
    if parsed:
        age, sex, drug, pt = parsed
    else:
        # Fallback a entrada guiada
        age, sex, drug, pt = solicitar_caso_manualmente()

    # 4. Procesar consulta
    safe_print(f"\n{YELLOW}[*] Procesando consulta en el RandomForestClassifier...{RESET}")
    time.sleep(1.0) # Retardo estético de cálculo
    
    res = helper.predecir_caso(age, sex, drug, pt)
    
    if not res["success"]:
        safe_print(f"\n{RED}[ERROR] {res['error']}{RESET}")
        safe_print(f"\n  Presione Enter para volver al menú principal...")
        input()
        return

    # 5. Imprimir reporte premium consolidado
    predicted_level = res["predicted_level"]
    predicted_label = res["predicted_label"]
    probs = res["probabilities"]
    historical_signal = res["historical_signal"]

    # Determinar color según severidad predicha
    if predicted_level == 5:
        color_sev = RED
        simbolo_sev = "🔴"
        rec_label = "[CRÍTICO] Perfil de Altísima Severidad / Riesgo Vital Detectado."
        rec_text = "El modelo asocia fuertemente este perfil demográfico y clínico con un desenlace crítico. Se recomienda alertar de inmediato al departamento de farmacovigilancia y suspender/evaluar la dosis terapéutica de forma inmediata."
    elif predicted_level in [3, 4]:
        color_sev = YELLOW
        simbolo_sev = "⚠️"
        rec_label = "[ALERTA ALTA] Perfil con Alta Gravedad Clínica (Hospitalización)."
        rec_text = "El modelo prevé un desenlace con alta probabilidad de hospitalización prolongada o intervención clínica crítica. Requiere supervisión médica estrecha y revisión del tratamiento."
    else:
        color_sev = GREEN
        simbolo_sev = "🟢"
        rec_label = "[INFORMACIÓN] Perfil Clínico de Severidad Moderada o Leve."
        rec_text = "El modelo clasifica la gravedad del paciente como de baja a moderada. Se sugiere continuar con monitorización habitual y reportar cualquier cambio en el estado del paciente."

    safe_print(f"\n{CYAN}{BOLD}" + "─" * 80 + f"{RESET}")
    safe_print(f"{CYAN}{BOLD}   REPORTE DE INFERENCIA PREDICTIVA DEL CASO CLINICO (ML Engine){RESET}")
    safe_print(f"{CYAN}{BOLD}" + "─" * 80 + f"{RESET}")
    
    safe_print(f"  {BOLD}📊 Datos Clínicos Ingresados:{RESET}")
    safe_print(f"   • Edad: {age} años    • Sexo: {sex} ({'Femenino' if sex=='F' else 'Masculino' if sex=='M' else 'Desconocido'})")
    safe_print(f"   • Fármaco Sospechoso: {BOLD}{drug}{RESET}")
    safe_print(f"   • Reacción Adversa: {BOLD}{pt}{RESET}")
    safe_print()
    
    safe_print(f"  {BOLD}⚡ RESULTADO DE GRAVEDAD DEL MODELO (Clasificación Multiclase):{RESET}")
    safe_print(f"   • Nivel de Severidad Predicho : {color_sev}{BOLD}{simbolo_sev} Nivel {predicted_level} ({predicted_label}){RESET}")
    safe_print(f"   • Confianza de la predicción  : {color_sev}{BOLD}{probs[predicted_level-1]*100:.2f}%{RESET}")
    safe_print()

    safe_print(f"  {BOLD}📈 Distribución de Probabilidades de Severidad (Bosque de Decisión):{RESET}")
    safe_print(f"   ├── Nivel 1 (Leve)             : {probs[0]*100:5.2f}%" + ("  <--" if predicted_level==1 else ""))
    safe_print(f"   ├── Nivel 2 (Moderada)         : {probs[1]*100:5.2f}%" + ("  <--" if predicted_level==2 else ""))
    safe_print(f"   ├── Nivel 3 (Hospitalización)  : {probs[2]*100:5.2f}%" + ("  <--" if predicted_level==3 else ""))
    safe_print(f"   ├── Nivel 4 (Grave/Secuela)    : {probs[3]*100:5.2f}%" + ("  <--" if predicted_level==4 else ""))
    safe_print(f"   └── Nivel 5 (Crítica/Muerte)   : {probs[4]*100:5.2f}%" + ("  <--" if predicted_level==5 else ""))
    safe_print()

    safe_print(f"  {BOLD}🛡️ RECOMENDACIÓN DE SEGURIDAD AUTOMÁTICA:{RESET}")
    safe_print(f"   {color_sev}{BOLD}{rec_label}{RESET}")
    safe_print(f"   {rec_text}")
    safe_print()

    # Enriquecer con MapReduce si existe señal
    safe_print(f"  {BOLD}⚙️ HISTORIAL MAPREDUCE (Señales de Seguridad Detectadas por Hadoop):{RESET}")
    if historical_signal:
        safe_print(f"   {GREEN}[SEÑAL DETECTADA]{RESET} Esta combinación fármaco-reacción posee registros estadísticos históricos:")
        safe_print(f"    • Frecuencia Histórica : {historical_signal['count']} reportes de co-ocurrencia.")
        safe_print(f"    • Métrica PRR          : {historical_signal['prr']:.2f} (Umbral de alerta >= 2.0)")
        safe_print(f"    • Métrica Chi-cuadrado : {historical_signal['chi2']:.2f} (Umbral de alerta >= 3.84)")
        safe_print(f"    • Estado de Alerta     : {RED if historical_signal['prr']>=2.0 and historical_signal['chi2']>=3.84 else YELLOW}{BOLD}{'SEÑAL ACTIVA Y CONFIRMADA' if historical_signal['prr']>=2.0 and historical_signal['chi2']>=3.84 else 'BAJO RIESGO ESTADISTICO'}{RESET}")
    else:
        safe_print(f"   {YELLOW}[INFO]{RESET} No se han registrado señales de seguridad estadísticas en el histórico MapReduce")
        safe_print(f"          para la pareja exacta '{drug}' + '{pt}'.")
    
    safe_print(f"{CYAN}" + "─" * 80 + f"{RESET}\n")
    safe_print("  Presione Enter para volver al menú principal...")
    input()

if __name__ == "__main__":
    run_interactive_inference()
