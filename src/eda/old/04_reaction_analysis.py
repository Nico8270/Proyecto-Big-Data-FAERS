"""
Script 04: Análisis de Reacciones Adversas (REAC)
Objetivo: Reacciones más frecuentes, número de reacciones por caso, estadísticas
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


def load_reac() -> pd.DataFrame:
    """Carga la tabla REAC."""
    filepath = DATA_DIR / FAERS_FILES["REAC"]
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def plot_top_reactions(reac: pd.DataFrame, top_n: int = 30):
    """Top reacciones adversas más reportadas."""
    if "pt" not in reac.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    top_reac = (reac["pt"]
                .str.strip()
                .str.title()
                .value_counts()
                .head(top_n))

    colors = sns.color_palette(PALETTE_MAIN, len(top_reac))
    top_reac.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Reacciones Adversas Más Reportadas (MedDRA PT)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Reportes")
    ax.set_ylabel("Término Preferido (PT)")
    ax.invert_yaxis()

    for i, v in enumerate(top_reac):
        ax.text(v + max(top_reac)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"04_top{top_n}_reactions.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")

    # Guardar CSV
    top_reac.to_csv(OUTPUT_DIR / "04_top_reactions.csv", header=["reportes"])
    print(f"  [OK] Top reacciones guardado en CSV")


def plot_reactions_per_case(reac: pd.DataFrame):
    """Distribución del número de reacciones por caso."""
    if "primaryid" not in reac.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    reac_per_case = reac.groupby("primaryid").size()

    # Limitar a 20 para visualización
    reac_clipped = reac_per_case.clip(upper=20)

    ax.hist(reac_clipped, bins=range(1, 22), color="coral", edgecolor="white", alpha=0.85)
    ax.set_title("Número de Reacciones Adversas por Caso", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cantidad de Reacciones (máx. 20)")
    ax.set_ylabel("Número de Casos")
    ax.set_xticks(range(1, 21))

    # Estadísticas
    stats_text = f"Media: {reac_per_case.mean():.2f} | Mediana: {reac_per_case.median():.0f} | Máx: {reac_per_case.max()}"
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_reactions_per_case.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_seriousness_analysis(demo: pd.DataFrame, reac: pd.DataFrame):
    """Análisis de seriedad: reacciones más asociadas con muerte/hospitalización."""
    if "outc_cod" not in demo.columns or "pt" not in reac.columns:
        return

    # Unir DEMO y REAC
    merged = pd.merge(demo[["primaryid", "outc_cod"]], reac[["primaryid", "pt"]], on="primaryid", how="inner")

    if len(merged) == 0:
        return

    # Análisis para outcomes graves: DE (muerte) y HO (hospitalización)
    serious_outcomes = ["DE", "HO"]
    fig, axes = plt.subplots(1, len(serious_outcomes), figsize=(16, 6))

    for idx, outcome in enumerate(serious_outcomes):
        subset = merged[merged["outc_cod"] == outcome]
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, f"No hay datos para {outcome}",
                           ha="center", va="center", transform=axes[idx].transAxes)
            axes[idx].set_title(f"Outcome: {OUTCOME_MAP.get(outcome, outcome)}")
            continue

        top_reactions = subset["pt"].value_counts().head(10)
        colors = sns.color_palette("Reds_r", len(top_reactions))
        top_reactions.plot(kind="barh", ax=axes[idx], color=colors, edgecolor="white")
        axes[idx].set_title(f"Top 10 Reacciones – {OUTCOME_MAP.get(outcome, outcome)}",
                            fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Número de Reportes")
        axes[idx].invert_yaxis()

    plt.suptitle("Reacciones Adversas por Outcome Grave", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_serious_outcomes_reactions.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_reaction_term_cloud_stats(reac: pd.DataFrame):
    """Estadísticas de longitud de términos de reacción."""
    if "pt" not in reac.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Longitud de los términos
    reac["pt_length"] = reac["pt"].astype(str).apply(len)
    reac_clean = reac[reac["pt_length"] > 0]

    if len(reac_clean) > 0:
        axes[0].hist(reac_clean["pt_length"], bins=30, color="purple", edgecolor="white", alpha=0.85)
        axes[0].set_title("Distribución de Longitud de Términos de Reacción", fontsize=13)
        axes[0].set_xlabel("Número de Caracteres")
        axes[0].set_ylabel("Frecuencia")

    # Palabras más comunes (top 15)
    from collections import Counter
    all_words = " ".join(reac_clean["pt"].astype(str)).lower().split()
    word_counts = Counter(all_words)
    top_words = pd.DataFrame(word_counts.most_common(15), columns=["palabra", "frecuencia"])

    axes[1].barh(range(len(top_words)), top_words["frecuencia"],
                 color=sns.color_palette("magma", len(top_words)), edgecolor="white")
    axes[1].set_yticks(range(len(top_words)))
    axes[1].set_yticklabels(top_words["palabra"])
    axes[1].set_title("Top 15 Palabras Más Frecuentes en Términos de Reacción", fontsize=13)
    axes[1].set_xlabel("Frecuencia")
    axes[1].invert_yaxis()

    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_reaction_text_stats.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 04: ANÁLISIS DE REACCIONES ADVERSAS – REAC")
    print("=" * 70)

    print("\n[1/5] Cargando tabla REAC...")
    try:
        reac = load_reac()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[2/5] Dimensiones: {reac.shape[0]:,} filas × {reac.shape[1]} columnas")

    # Cargar DEMO para análisis de seriedad
    print("\n[3/5] Cargando tabla DEMO para análisis cruzado...")
    try:
        demo = pd.read_csv(
            DATA_DIR / FAERS_FILES["DEMO"],
            sep=FAERS_SEP,
            encoding=FAERS_ENCODING,
            low_memory=False,
            usecols=["primaryid", "outc_cod"]
        )
        demo.columns = demo.columns.str.strip().str.lower()
    except:
        demo = None
        print("  [ADVERTENCIA] No se pudo cargar DEMO para análisis cruzado")

    print("\n[4/5] Generando análisis de reacciones...")
    plot_top_reactions(reac, top_n=30)
    plot_reactions_per_case(reac)
    plot_reaction_term_cloud_stats(reac)
    if demo is not None:
        plot_seriousness_analysis(demo, reac)

    # Estadísticas resumen
    print("\n[5/5] Guardando estadísticas detalladas...")
    if "pt" in reac.columns:
        reac_stats = (reac["pt"]
                      .str.strip()
                      .str.title()
                      .value_counts()
                      .reset_index()
                      .rename(columns={"index": "reaccion_pt", "pt": "frecuencia"}))
        reac_stats["porcentaje"] = (reac_stats["frecuencia"] / len(reac) * 100).round(3)
        reac_stats.to_csv(OUTPUT_DIR / "04_reaction_statistics.csv", index=False)
        print(f"  [OK] Estadísticas guardadas en CSV")

    print("\n[5/5] Análisis de reacciones adversas completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
