"""
Script 02: Análisis Demográfico (DEMO)
Objetivo: Distribución de edad, género, países, y outcomes
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


def load_demo() -> pd.DataFrame:
    """Carga la tabla DEMO."""
    filepath = DATA_DIR / FAERS_FILES["DEMO"]
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def plot_gender_distribution(demo: pd.DataFrame):
    """Gráfico de distribución por género."""
    if "sex" not in demo.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sex_counts = demo["sex"].fillna("U").value_counts()
    colors = sns.color_palette(PALETTE_MAIN, len(sex_counts))
    sex_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Distribución por Género", fontsize=14, fontweight="bold")
    ax.set_xlabel("Género (F=Femenino, M=Masculino, U=Desconocido)")
    ax.set_ylabel("Número de Reportes")
    ax.tick_params(axis="x", rotation=0)

    for i, v in enumerate(sex_counts):
        ax.text(i, v + max(sex_counts)*0.01, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_gender_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_age_distribution(demo: pd.DataFrame):
    """Histograma de edad por unidad de medida."""
    if "age" not in demo.columns or "age_cod" not in demo.columns:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    codes = demo["age_cod"].fillna("UNK").unique()[:6]
    colors = sns.color_palette(PALETTE_MAIN, len(codes))

    for idx, code in enumerate(codes):
        ax = axes[idx]
        subset = demo[demo["age_cod"] == code]
        ages = pd.to_numeric(subset["age"], errors="coerce")
        ages = ages[(ages >= 0) & (ages <= 120)]

        if len(ages) > 0:
            ax.hist(ages.dropna(), bins=30, color=colors[idx], edgecolor="white", alpha=0.85)
            ax.set_title(f"Edad en {AGE_CODE_MAP.get(code, code)} (código: {code})")
            ax.set_xlabel("Edad")
            ax.set_ylabel("Frecuencia")

    # Ajustar ejes vacíos
    for idx in range(len(codes), 6):
        axes[idx].axis("off")

    plt.suptitle("Distribución de Edad por Unidad de Medida", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_age_distribution_by_unit.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_top_countries(demo: pd.DataFrame, top_n: int = 15):
    """Top países reportantes."""
    if "reporter_country" not in demo.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    top_countries = demo["reporter_country"].fillna("UNK").value_counts().head(top_n)

    colors = sns.color_palette(PALETTE_SEQUEN, len(top_countries))
    top_countries.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Países Reportantes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Reportes")
    ax.set_ylabel("Código de País (ISO)")
    ax.invert_yaxis()

    for i, v in enumerate(top_countries):
        ax.text(v + max(top_countries)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_top_countries.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_outcomes(demo: pd.DataFrame):
    """Distribución de outcomes clínicos."""
    if "outc_cod" not in demo.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    outc_counts = demo["outc_cod"].fillna("").value_counts()

    labels = [OUTCOME_MAP.get(code, code) for code in outc_counts.index]
    colors = sns.color_palette("Reds_r", len(outc_counts))

    bars = outc_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.set_title("Distribución de Resultados Clínicos (Outcomes)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tipo de Outcome")
    ax.set_ylabel("Número de Reportes")
    ax.tick_params(axis="x", rotation=45)

    # Etiquetas de porcentaje
    total = outc_counts.sum()
    for bar, label in zip(bars.patches, labels):
        height = bar.get_height()
        pct = height / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, height + max(outc_counts)*0.01,
                f"{height:,}\n({pct:.1f}%)", ha="center", fontsize=9)

    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_outcomes_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_age_gender_boxplot(demo: pd.DataFrame):
    """Boxplot de edad por género."""
    if "age" not in demo.columns or "sex" not in demo.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    valid_ages = demo.copy()
    valid_ages["age_num"] = pd.to_numeric(valid_ages["age"], errors="coerce")
    valid_ages = valid_ages[(valid_ages["age_num"] >= 0) & (valid_ages["age_num"] <= 120)]

    if len(valid_ages) == 0:
        return

    sns.boxplot(data=valid_ages, x="sex", y="age_num", ax=ax, palette=PALETTE_CATEG)
    ax.set_title("Distribución de Edad por Género", fontsize=14, fontweight="bold")
    ax.set_xlabel("Género")
    ax.set_ylabel("Edad (años)")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_age_gender_boxplot.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 02: ANÁLISIS DEMOGRÁFICO – DEMO")
    print("=" * 70)

    print("\n[1/5] Cargando tabla DEMO...")
    try:
        demo = load_demo()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("\n[2/5] Cargando tabla OUTC para análisis de outcomes...")
    try:
        outc = pd.read_csv(
            DATA_DIR / FAERS_FILES["OUTC"],
            sep=FAERS_SEP,
            encoding=FAERS_ENCODING,
            low_memory=False,
            usecols=["primaryid", "outc_cod"]
        )
        outc.columns = outc.columns.str.strip().str.lower()
        print(f"  OUTC cargada: {len(outc):,} filas")
    except Exception as e:
        print(f"  [ADVERTENCIA] No se pudo cargar OUTC: {e}")
        outc = None

    print(f"\n[2/5] Dimensiones: {demo.shape[0]:,} filas × {demo.shape[1]} columnas")

    print("\n[3/5] Generando gráficos demográficos...")
    plot_gender_distribution(demo)
    plot_age_distribution(demo)
    plot_top_countries(demo)
    plot_age_gender_boxplot(demo)

    # Guardar estadísticas numéricas
    print("\n[4/5] Calculando estadísticas de edad...")
    valid_ages = pd.to_numeric(demo["age"], errors="coerce")
    valid_ages = valid_ages[(valid_ages >= 0) & (valid_ages <= 120)]

    stats_df = pd.DataFrame({
        "estadística": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "valor": valid_ages.describe().round(2).values
    })
    stats_path = OUTPUT_DIR / "02_age_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  [OK] Estadísticas guardadas: {stats_path.name}")

    print("\n[5/5] Análisis demográfico completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
