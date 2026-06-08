"""
src/hadoop/training/03_model_training.py
============================================
ETAPA 03: ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING
        + Comparación de modelos: LR · RandomForest · XGBoost · LightGBM

Carga features de entrenamiento desde data/staging/training_features/features_train.jsonl
(generado en la etapa 02, que ya realizó el train/test split correctamente).

CORRECCIÓN DE DATA LEAKAGE — orden garantizado:
  1. X_train / y_train ← features_train.jsonl   (ya particionado antes de SMOTE)
  2. X_test  / y_test  ← features_test.jsonl    (intactos, sin sintéticos)
  3. SMOTE vive DENTRO de ImbPipeline → nunca ve datos de validación en CV
  4. Modelo final: SMOTE sobre X_train completo → fit → serializado

COMPARACIÓN DE MODELOS (nuevo):
  Registro de modelos:
    - "Baseline (LR)"  : LogisticRegression(max_iter=1000, random_state=42)
    - "Random Forest"  : RandomForestClassifier (modelo actual del pipeline)
    - "XGBoost"        : XGBClassifier(eval_metric='mlogloss', random_state=42)
    - "LightGBM"       : LGBMClassifier(random_state=42, verbosity=-1)

  Por cada modelo:
    a) 5-fold StratifiedKFold CV con ImbPipeline([SMOTE, classifier]) en X_train
    b) Entrenamiento sobre X_train_resampled (SMOTE full-train)
    c) Evaluación en X_test (limpio, sin sintéticos)

  Salidas gráficas:
    outputs/model_assets/cv_fold_scores.png          ← barras CV por fold
    outputs/model_assets/confusion_matrices.png      ← grilla 2×2 de matrices de conf.
    outputs/model_assets/roc_curves.png              ← curvas ROC OvR macro por modelo

  Salidas de datos:
    outputs/model_assets/modelo_severidad.joblib     ← mejor modelo (más alto test f1_macro)
    outputs/balance_reports/model_comparison.json    ← tabla comparativa completa
"""

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")           # backend sin GUI — seguro en pipelines headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize

import xgboost as xgb
import lightgbm as lgb

# ── Suprimir advertencias no críticas de versión/convergencia ─────────────────
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data"    / "staging" / "training_features"
ASSETS_DIR   = PROJECT_ROOT / "outputs" / "model_assets"
REPORTS_DIR  = PROJECT_ROOT / "outputs" / "balance_reports"

FEATURE_COLS = ["drug_encoded", "pt_encoded", "sex_encoded", "age"]

# ── Paleta de colores por modelo (consistente en todas las figuras) ───────────
MODEL_COLORS = {
    "Baseline (LR)": "#7B8CDE",
    "Random Forest": "#4CAF82",
    "XGBoost":       "#E87C4C",
    "LightGBM":      "#C44CE8",
}


# ─────────────────────────────────────────────────────────────────────────────
# Registro de modelos
# ─────────────────────────────────────────────────────────────────────────────

def build_model_registry() -> dict:
    """
    Construye el registro de modelos con hiperparámetros por defecto.

    XGBoost ≥ 2.0 eliminó el parámetro `use_label_encoder`; lo omitimos
    para compatibilidad hacia adelante. El parámetro `eval_metric='mlogloss'`
    sigue siendo válido en todas las versiones recientes.
    """
    xgb_kwargs = dict(eval_metric="mlogloss", random_state=42, n_jobs=-1)
    # `use_label_encoder` eliminado en XGB >= 2.0: no lo pasamos.

    return {
        "Baseline (LR)": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=16,
            class_weight="balanced",
            random_state=42, n_jobs=1,   # n_jobs=1: CV paraleliza en folds
        ),
        "XGBoost": xgb.XGBClassifier(**xgb_kwargs),
        "LightGBM": lgb.LGBMClassifier(
            random_state=42, verbosity=-1, n_jobs=-1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de formato
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def _bar(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cv_for_model(
    name: str,
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Corre StratifiedKFold CV usando ImbPipeline([SMOTE, clf]).
    SMOTE se ajusta SOLO en el fold de entrenamiento de cada iteración.
    Retorna array de shape (n_splits,) con f1_macro por fold.
    """
    pipeline = ImbPipeline([
        ("smote",      SMOTE(random_state=42)),
        ("classifier", clf),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento final por modelo
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(clf, X_train_res: np.ndarray, y_train_res: np.ndarray):
    """
    Ajusta el clasificador sobre el conjunto de entrenamiento
    ya remuestreado con SMOTE. Retorna el clasificador ajustado.
    """
    clf.fit(X_train_res, y_train_res)
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1: Barras CV fold-a-fold
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_bars(
    cv_results: dict[str, np.ndarray],
    output_path: Path,
):
    """
    Gráfico de barras agrupadas: f1_macro por fold para cada modelo.
    Una línea horizontal por modelo indica su media CV.
    """
    model_names = list(cv_results.keys())
    n_models = len(model_names)
    n_folds  = len(next(iter(cv_results.values())))

    x      = np.arange(n_folds)
    width  = 0.8 / n_models
    offset = (np.arange(n_models) - n_models / 2 + 0.5) * width

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, name in enumerate(model_names):
        scores = cv_results[name]
        color  = MODEL_COLORS[name]
        bars   = ax.bar(
            x + offset[i], scores,
            width=width * 0.92,
            color=color, alpha=0.82, label=name, zorder=3,
        )
        ax.axhline(
            scores.mean(), color=color, linewidth=1.5,
            linestyle="--", alpha=0.7, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)], fontsize=11)
    ax.set_xlabel("Fold de validación cruzada", fontsize=12)
    ax.set_ylabel("f1_macro", fontsize=12)
    ax.set_title(
        "Validación Cruzada Estratificada (5-Fold) — f1_macro por fold\n"
        "ImbPipeline: SMOTE → Clasificador (SMOTE solo en fold de entrenamiento)",
        fontsize=12, fontweight="bold",
    )
    ymin = max(0.0, min(s.min() for s in cv_results.values()) - 0.04)
    ymax = min(1.0, max(s.max() for s in cv_results.values()) + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico CV guardado : {output_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2: Matrices de confusión 2×2
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: list,
    output_path: Path,
):
    """
    Grilla 2×2 de matrices de confusión (una por modelo) evaluadas en X_test.
    Las etiquetas de clase son los niveles de severidad (1-5).
    """
    n_models = len(fitted_models)
    ncols = 2
    nrows = (n_models + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5.5))
    axes_flat = axes.flatten()

    for idx, (name, clf) in enumerate(fitted_models.items()):
        ax    = axes_flat[idx]
        y_pred = clf.predict(X_test)
        cm    = confusion_matrix(y_test, y_pred, labels=class_labels)

        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")

        acc   = accuracy_score(y_test, y_pred)
        f1_m  = f1_score(y_test, y_pred, average="macro")
        ax.set_title(
            f"{name}\nAcc={acc:.3f}  |  f1_macro={f1_m:.4f}",
            fontsize=11, fontweight="bold",
            color=MODEL_COLORS[name],
        )
        ax.set_xlabel("Predicho", fontsize=9)
        ax.set_ylabel("Real", fontsize=9)

    # Ocultar subplots vacíos si el número de modelos es impar
    for idx in range(len(fitted_models), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Matrices de Confusión — Evaluación sobre X_test (datos reales, sin SMOTE)\n"
        "Filas = clases reales · Columnas = clases predichas",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrices guardadas  : {output_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 3: Curvas ROC OvR macro
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: list,
    output_path: Path,
):
    """
    Curvas ROC One-vs-Rest con media macro para cada modelo, en los mismos ejes.
    Solo los modelos que implementan predict_proba son graficados.
    """
    y_bin = label_binarize(y_test, classes=class_labels)
    n_classes = y_bin.shape[1]

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, clf in fitted_models.items():
        if not hasattr(clf, "predict_proba"):
            print(f"  [SKIP ROC] {name} no tiene predict_proba — omitido en ROC.")
            continue

        try:
            y_score = clf.predict_proba(X_test)    # shape (n_samples, n_classes)
        except Exception as exc:
            print(f"  [SKIP ROC] {name}: predict_proba falló ({exc})")
            continue

        # Calcular curva ROC macro: promedio de AUC por clase
        fprs, tprs, aucs_per_class = [], [], []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
            fprs.append(fpr)
            tprs.append(tpr)
            aucs_per_class.append(auc(fpr, tpr))

        # Interpolar sobre base FPR común para promedio macro
        fpr_grid = np.linspace(0, 1, 500)
        tprs_interp = [np.interp(fpr_grid, fpr, tpr)
                       for fpr, tpr in zip(fprs, tprs)]
        mean_tpr  = np.mean(tprs_interp, axis=0)
        macro_auc = float(np.mean(aucs_per_class))

        ax.plot(
            fpr_grid, mean_tpr,
            color=MODEL_COLORS[name],
            linewidth=2.2,
            label=f"{name}  (AUC macro = {macro_auc:.4f})",
        )

    # Línea de referencia aleatoria
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Azar")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    ax.set_title(
        "Curvas ROC — One-vs-Rest (OvR) · Promedio Macro\n"
        "Evaluación sobre X_test (datos reales, sin SMOTE)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curvas ROC guardadas: {output_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Tabla comparativa
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]):
    """
    Imprime una tabla ASCII de comparación y devuelve el nombre del mejor modelo
    según test f1_macro.
    """
    _section("TABLA COMPARATIVA — CV f1_macro · Test f1_macro · Test Accuracy")
    header = f"  {'Modelo':<20} {'CV mean':>10} {'CV std':>8} {'Test f1':>10} {'Test acc':>10}"
    print(header)
    print("  " + "─" * 62)
    best_name  = ""
    best_score = -1.0
    for r in results:
        marker = "◀ MEJOR" if r["test_f1_macro"] == max(x["test_f1_macro"] for x in results) else ""
        print(
            f"  {r['model']:<20} "
            f"{r['cv_mean']:>10.4f} "
            f"{r['cv_std']:>8.4f} "
            f"{r['test_f1_macro']:>10.4f} "
            f"{r['test_accuracy']:>10.4f}  {marker}"
        )
        if r["test_f1_macro"] > best_score:
            best_score = r["test_f1_macro"]
            best_name  = r["model"]
    print("  " + "─" * 62)
    return best_name


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 64)
    print("  ETAPA 03: COMPARACIÓN DE MODELOS  (LR · RF · XGBoost · LightGBM)")
    print("  Pipeline: ImbPipeline(SMOTE → clf) · 5-Fold StratifiedKFold")
    print("=" * 64)

    # --- Step 1: Cargar splits ---
    train_file = FEATURES_DIR / "features_train.jsonl"
    test_file  = FEATURES_DIR / "features_test.jsonl"

    if not train_file.exists():
        train_file = FEATURES_DIR / "features.jsonl"
        if not train_file.exists():
            print(f"\n[ERROR] No se encontró features_train.jsonl ni features.jsonl.")
            sys.exit(1)
        print("  [AVISO] Usando features.jsonl (pipeline legado).")

    if not test_file.exists():
        print(f"\n[ERROR] features_test.jsonl no encontrado: {test_file}")
        print("        Ejecute la etapa 02 para generar los splits.")
        sys.exit(1)

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    try:
        # --- Step 2: Cargar datos ---
        df_train = pd.read_json(train_file, orient="records", lines=True)
        df_test  = pd.read_json(test_file,  orient="records", lines=True)

        X_train = df_train[FEATURE_COLS].copy()
        y_train = df_train["severity_level"].astype(int)
        X_test  = df_test[FEATURE_COLS].copy()
        y_test  = df_test["severity_level"].astype(int)

        class_labels = sorted(y_train.unique().tolist())
        n_classes    = len(class_labels)

        print(f"\n  Train : {len(X_train):,} muestras | Test : {len(X_test):,} muestras")
        print(f"  Clases: {class_labels}  ({n_classes} niveles de severidad)")
        print(f"  Features: {FEATURE_COLS}")

        # --- Step 3: Distribución de clases pre-SMOTE ---
        print(f"\n  Distribución y_train (antes de SMOTE):")
        for cls, cnt in sorted(y_train.value_counts().items()):
            pct = cnt / len(y_train) * 100
            print(f"    Clase {cls}: {cnt:>6,}  ({pct:.1f}%)")

        # --- Step 4: Registro de modelos ---
        models = build_model_registry()

        # --- Step 5: SMOTE sobre X_train completo para entrenamiento final ---
        # CV usa su propio SMOTE dentro del pipeline (no contamina folds de val).
        # El fit final de cada modelo usa este X_train_resampled.
        print("\n  Aplicando SMOTE sobre X_train completo (para entrenamientos finales)...")
        smote_global = SMOTE(random_state=42)
        X_train_res, y_train_res = smote_global.fit_resample(X_train, y_train)
        print(f"    -> {len(X_train):,} originales + {len(X_train_res)-len(X_train):,} "
              f"sintéticas = {len(X_train_res):,} total")

        # --- Step 6: Loop principal: CV + entrenamiento + evaluación ---
        cv_results:     dict[str, np.ndarray] = {}
        fitted_models:  dict[str, object]     = {}
        comparison_rows: list[dict]           = []

        for name, clf in models.items():
            _section(f"Modelo: {name}")

            # --- Step 6a: Cross-validation (SMOTE DENTRO del pipeline) ---
            print(f"  Ejecutando CV 5-fold (ImbPipeline[SMOTE → {clf.__class__.__name__}])...")
            t0 = time.time()
            scores = run_cv_for_model(name, clf, X_train, y_train, n_splits=5)
            t_cv = time.time() - t0
            cv_results[name] = scores

            print(f"  CV completado en {t_cv:.1f}s")
            print(f"  {' ':>8} f1_macro por fold:")
            for i, s in enumerate(scores, 1):
                print(f"    Fold {i}: {_bar(str(i), s)}")
            print(f"  Media : {scores.mean():.4f}  |  Std : {scores.std():.4f}")

            # --- Step 6b: Entrenamiento final (SMOTE ya aplicado) ---
            print(f"\n  Entrenando {name} sobre X_train_resampled ({len(X_train_res):,} muestras)...")
            t0 = time.time()
            # Necesitamos una instancia fresca (CV puede haber clonado la anterior)
            clf_final = train_final_model(clf, X_train_res.values, y_train_res)
            t_fit = time.time() - t0
            print(f"  Ajuste completado en {t_fit:.1f}s")
            fitted_models[name] = clf_final

            # --- Step 6c: Evaluación en X_test ---
            y_pred    = clf_final.predict(X_test.values)
            test_f1   = f1_score(y_test, y_pred, average="macro")
            test_acc  = accuracy_score(y_test, y_pred)

            print(f"\n  [Test] f1_macro  = {test_f1:.4f}")
            print(f"  [Test] accuracy  = {test_acc:.4f}")
            print(f"\n  classification_report (X_test):")
            print(classification_report(
                y_test, y_pred,
                labels=class_labels,
                target_names=[f"Sev {c}" for c in class_labels],
                digits=4,
            ))

            comparison_rows.append({
                "model":        name,
                "cv_mean":      float(scores.mean()),
                "cv_std":       float(scores.std()),
                "test_f1_macro": float(test_f1),
                "test_accuracy": float(test_acc),
            })

        # --- Step 7: Tabla comparativa ---
        best_model_name = print_comparison_table(comparison_rows)

        # --- Step 8: Guardar mejor modelo ---
        _section(f"Serializando mejor modelo: {best_model_name}")
        model_path = ASSETS_DIR / "modelo_severidad.joblib"
        joblib.dump(fitted_models[best_model_name], model_path, compress=3)
        print(f"  Guardado en: {model_path.relative_to(PROJECT_ROOT)}")

        # --- Step 9: Guardar tabla comparativa JSON ---
        comparison_json = REPORTS_DIR / "model_comparison.json"
        with open(comparison_json, "w", encoding="utf-8") as f:
            json.dump({
                "models": comparison_rows,
                "best_model": best_model_name,
                "feature_cols": FEATURE_COLS,
                "n_train": int(len(X_train)),
                "n_test":  int(len(X_test)),
                "class_labels": class_labels,
                "smote_applied": True,
                "cv_folds": 5,
                "data_leakage_free": True,
            }, f, indent=2)
        print(f"  Comparación JSON: {comparison_json.relative_to(PROJECT_ROOT)}")

        # ── Gráfico 1: Barras CV ──────────────────────────────────────────────
        _section("Generando gráficos")
        plot_cv_bars(cv_results, ASSETS_DIR / "cv_fold_scores.png")

        # ── Gráfico 2: Matrices de confusión 2×2 ──────────────────────────────
        plot_confusion_matrices(
            fitted_models, X_test, y_test, class_labels,
            ASSETS_DIR / "confusion_matrices.png",
        )

        # ── Gráfico 3: Curvas ROC ─────────────────────────────────────────────
        plot_roc_curves(
            fitted_models, X_test, y_test, class_labels,
            ASSETS_DIR / "roc_curves.png",
        )

        # ── Resumen final ─────────────────────────────────────────────────────
        dur_total = time.time() - t_total
        print()
        print("=" * 64)
        print("  RESUMEN ETAPA 03")
        print("=" * 64)
        for r in sorted(comparison_rows, key=lambda x: x["test_f1_macro"], reverse=True):
            marker = " ← MEJOR" if r["model"] == best_model_name else ""
            print(f"  {r['model']:<20}  CV={r['cv_mean']:.4f}±{r['cv_std']:.4f}"
                  f"  Test-f1={r['test_f1_macro']:.4f}  Acc={r['test_accuracy']:.4f}{marker}")
        print(f"\n  Tiempo total : {dur_total:.1f}s")
        print(f"  Modelo final : {model_path.relative_to(PROJECT_ROOT)}")
        print(f"  Tabla JSON   : {comparison_json.relative_to(PROJECT_ROOT)}")

    except Exception as exc:
        print(f"\n[ERROR] Excepción en etapa 03: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 64)
    print("  ETAPA 03 COMPLETADA")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
