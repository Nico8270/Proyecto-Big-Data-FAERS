"""
Script 05: Análisis de Outcomes (OUTC)
Objetivo: Distribución de outcomes, combinación con DEMO para severidad
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = FIGURE_DPI


def load_outc() -> pd.DataFrame:
    """Carga la tabla OUTC."""
    filepath = DATA_DIR / FAERS_FILES["OUTC"]
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def plot_outcome_distribution(outc: pd.DataFrame):
    """Distribución de outcomes."""
    if "outc_cod" not in outc.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    outc_counts = outc["outc_cod"].fillna("").value_counts()

    labels = [OUTCOME_MAP.get(code, code) for code in outc_counts.index]
    colors = sns.color_palette("Reds_r", len(outc_counts))

    bars = outc_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Distribución de Outcomes (Resultados Clínicos)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Tipo de Outcome")
    ax.set_ylabel("Número de Reportes")
    ax.tick_params(axis="x", rotation=45)

    total = outc_counts.sum()
    for bar, count in zip(bars.patches, outc_counts.values):
        height = bar.get_height()
        pct = height / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, height + max(outc_counts)*0.02,
                f"{count:,}\n({pct:.1f}%)", ha="center", fontsize=9)

    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_outcomes_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_outcome_by_gender(demo: pd.DataFrame, outc: pd.DataFrame):
    """Outcomes por género."""
    if "sex" not in demo.columns or "outc_cod" not in outc.columns:
        return

    # Unir DEMO y OUTC
    merged = pd.merge(
        demo[["primaryid", "sex"]],
        outc[["primaryid", "outc_cod"]],
        on="primaryid",
        how="inner"
    )

    if len(merged) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    cross_tab = pd.crosstab(merged["sex"], merged["outc_cod"], normalize="index") * 100
    cross_tab.columns = [OUTCOME_MAP.get(col, col) for col in cross_tab.columns]

    cross_tab.plot(kind="bar", ax=ax, colormap=PALETTE_DIVERG, edgecolor="white")
    ax.set_title("Distribución de Outcomes por Género (%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Género")
    ax.set_ylabel("Porcentaje de Reportes")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_outcomes_by_gender.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_serious_vs_nonserious(demo: pd.DataFrame, outc: pd.DataFrame):
    """Comparación de casos serios vs no serios."""
    # Definir outcomes considerados graves
    serious_codes = ["DE", "HO", "LT", "DS", "CA", "RI"]

    merged = pd.merge(
        demo[["primaryid", "sex", "age"]],
        outc[["primaryid", "outc_cod"]],
        on="primaryid",
        how="inner"
    )

    if len(merged) == 0:
        return

    merged["serio"] = merged["outc_cod"].isin(serious_codes).map({True: "Sí", False: "No"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Conteo
    serious_counts = merged["serio"].value_counts()
    axes[0].pie(serious_counts, labels=serious_counts.index, autopct="%1.1f%%",
                colors=[sns.color_palette("Set2")[1], sns.color_palette("Set2")[0]],
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("Proporción de Casos Serios vs No Serios", fontsize=13, fontweight="bold")
    axes[0].axis("equal")

    # Edad por severidad
    merged["age_num"] = pd.to_numeric(merged["age"], errors="coerce")
    valid = merged[(merged["age_num"] >= 0) & (merged["age_num"] <= 120)]

    if len(valid) > 0:
        sns.boxplot(data=valid, x="serio", y="age_num", ax=axes[1],
                    palette={"Sí": "firebrick", "No": "forestgreen"})
        axes[1].set_title("Edad por Severidad del Outcome", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("¿Caso serio?")
        axes[1].set_ylabel("Edad (años)")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_seriousness_analysis.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 05: ANÁLISIS DE OUTCOMES – OUTC")
    print("=" * 70)

    print("\n[1/5] Cargando tabla OUTC...")
    try:
        outc = load_outc()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[2/5] Dimensiones: {outc.shape[0]:,} filas × {outc.shape[1]} columnas")

    # Cargar DEMO
    print("\n[3/5] Cargando tabla DEMO para análisis combinado...")
    try:
        demo = pd.read_csv(
            DATA_DIR / FAERS_FILES["DEMO"],
            sep=FAERS_SEP,
            encoding=FAERS_ENCODING,
            low_memory=False,
            usecols=["primaryid", "sex", "age", "outc_cod"]
        )
        demo.columns = demo.columns.str.strip().str.lower()
    except:
        demo = None
        print("  [ADVERTENCIA] No se pudo cargar DEMO completamente")

    print("\n[4/5] Generando visualizaciones de outcomes...")
    plot_outcome_distribution(outc)

    if demo is not None:
        plot_outcome_by_gender(demo, outc)
        plot_serious_vs_nonserious(demo, outc)

    # Estadísticas
    print("\n[5/5] Guardando estadísticas de outcomes...")
    outc_stats = outc["outc_cod"].fillna("").value_counts().reset_index()
    outc_stats.columns = ["outc_cod", "count"]
    outc_stats["descripcion"] = outc_stats["outc_cod"].map(OUTCOME_MAP)
    outc_stats["porcentaje"] = (outc_stats["count"] / len(outc) * 100).round(2)
    outc_stats.to_csv(OUTPUT_DIR / "05_outcome_statistics.csv", index=False)
    print(f"  [OK] Estadísticas guardadas en CSV")

    print("\n[5/5] Análisis de outcomes completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
