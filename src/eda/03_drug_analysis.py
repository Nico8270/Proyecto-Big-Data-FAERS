"""
Script 03: Análisis de Medicamentos (DRUG)
Objetivo: Fármacos más reportados, roles, vías de administración, dosificación
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


def load_drug() -> pd.DataFrame:
    """Carga la tabla DRUG."""
    filepath = DATA_DIR / FAERS_FILES["DRUG"]
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df


def plot_top_drugs(drug: pd.DataFrame, top_n: int = 25):
    """Top fármacos más reportados."""
    if "drugname" not in drug.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    top_drugs = (drug["drugname"]
                 .str.upper()
                 .str.strip()
                 .value_counts()
                 .head(top_n))

    colors = sns.color_palette(PALETTE_MAIN, len(top_drugs))
    top_drugs.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Fármacos Más Reportados", fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Registros")
    ax.set_ylabel("Nombre del Fármaco")
    ax.invert_yaxis()

    for i, v in enumerate(top_drugs):
        ax.text(v + max(top_drugs)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"03_top{top_n}_drugs.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")

    # Guardar tabla CSV
    top_drugs.to_csv(OUTPUT_DIR / "03_top_drugs.csv", header=["reportes"])
    print(f"  [OK] Top fármacos guardado en CSV")


def plot_drug_roles(drug: pd.DataFrame):
    """Distribución de roles de medicamentos."""
    if "role_cod" not in drug.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    role_counts = drug["role_cod"].fillna("??").value_counts()
    labels = [DRUG_ROLE_MAP.get(code, code) for code in role_counts.index]

    # Gráfico de barras
    bars = role_counts.plot(kind="bar", ax=ax1, color=sns.color_palette(PALETTE_CATEG, len(role_counts)), edgecolor="white")
    ax1.set_title("Distribución por Rol del Fármaco", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Rol")
    ax1.set_ylabel("Número de Registros")
    ax1.tick_params(axis="x", rotation=0)

    for bar, label in zip(bars.patches, labels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + max(role_counts)*0.02,
                f"{height:,}\n({height/len(drug)*100:.1f}%)", ha="center", fontsize=9)

    ax1.set_xticklabels(labels, rotation=0)

    # Gráfico de torta
    ax2.pie(role_counts, labels=labels, autopct="%1.1f%%",
            colors=sns.color_palette(PALETTE_CATEG, len(role_counts)),
            startangle=90, wedgeprops={"edgecolor": "white"})
    ax2.set_title("Proporción por Rol", fontsize=13, fontweight="bold")
    ax2.axis("equal")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_drug_roles.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_administration_routes(drug: pd.DataFrame, top_n: int = 15):
    """Vías de administración más frecuentes."""
    if "route" not in drug.columns:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    top_routes = drug["route"].fillna("UNK").value_counts().head(top_n)

    colors = sns.color_palette(PALETTE_SEQUEN, len(top_routes))
    top_routes.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Vías de Administración", fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Registros")
    ax.set_ylabel("Vía de Administración")
    ax.invert_yaxis()

    for i, v in enumerate(top_routes):
        ax.text(v + max(top_routes)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_administration_routes.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_dosage_distribution(drug: pd.DataFrame):
    """Distribución de dosis reportadas."""
    if "dose_amt" not in drug.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma general
    doses = pd.to_numeric(drug["dose_amt"], errors="coerce")
    doses = doses[(doses > 0) & (doses < 10000)]  # Filtrar outliers extremos

    if len(doses) > 0:
        axes[0].hist(doses.dropna(), bins=50, color="teal", edgecolor="white", alpha=0.85)
        axes[0].set_title("Distribución de Dosis Reportadas", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Dosis (unidades)")
        axes[0].set_ylabel("Frecuencia")

    # Top 10 unidades de dosis
    if "dose_unit" in drug.columns:
        top_units = drug["dose_unit"].fillna("UNK").value_counts().head(10)
        axes[1].barh(range(len(top_units)), top_units.values,
                     color=sns.color_palette(PALETTE_SEQUEN, len(top_units)), edgecolor="white")
        axes[1].set_yticks(range(len(top_units)))
        axes[1].set_yticklabels(top_units.index)
        axes[1].set_title("Top 10 Unidades de Dosis", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Número de Reportes")
        axes[1].invert_yaxis()

        for i, v in enumerate(top_units):
            axes[1].text(v + max(top_units)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_dosage_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_dechal_rechal(drug: pd.DataFrame):
    """Indicadores de suspensión/reinstauración de medicamento."""
    cols = ["dechal", "rechal"]
    if not all(c in drug.columns for c in cols):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # DE (Drug discontinued/withdrawn)
    dechal_counts = drug["dechal"].fillna("U").value_counts()
    dechal_labels = {"Y": "Sí", "N": "No", "U": "Desconocido"}
    dechal_index = [dechal_labels.get(c, c) for c in dechal_counts.index]

    ax1.bar(dechal_index, dechal_counts.values, color=sns.color_palette(PALETTE_CATEG, 3), edgecolor="white")
    ax1.set_title("¿Se Suspendió el Medicamento? (DE)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Número de Reportes")

    # RE (Drug restarted)
    rechal_counts = drug["rechal"].fillna("U").value_counts()
    rechal_labels = {"Y": "Sí", "N": "No", "U": "Desconocido"}
    rechal_index = [rechal_labels.get(c, c) for c in rechal_counts.index]

    ax2.bar(rechal_index, rechal_counts.values, color=sns.color_palette(PALETTE_CATEG, 3), edgecolor="white")
    ax2.set_title("¿Se Reinició el Medicamento? (RE)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Número de Reportes")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_dechal_rechal.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 03: ANÁLISIS DE MEDICAMENTOS – DRUG")
    print("=" * 70)

    print("\n[1/5] Cargando tabla DRUG...")
    try:
        drug = load_drug()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[2/5] Dimensiones: {drug.shape[0]:,} filas × {drug.shape[1]} columnas")
    print(f"      PrimaryIDs únicos: {drug['primaryid'].nunique():,}" if "primaryid" in drug.columns else "")

    print("\n[3/5] Generando análisis de fármacos...")
    plot_top_drugs(drug, top_n=25)
    plot_drug_roles(drug)
    plot_administration_routes(drug)
    plot_dosage_distribution(drug)
    plot_dechal_rechal(drug)

    # Guardar estadísticas de roles
    print("\n[4/5] Guardando estadísticas de roles...")
    if "role_cod" in drug.columns:
        role_stats = drug["role_cod"].fillna("??").value_counts().reset_index()
        role_stats.columns = ["role_cod", "count"]
        role_stats["descripcion"] = role_stats["role_cod"].map(DRUG_ROLE_MAP)
        role_stats["porcentaje"] = (role_stats["count"] / len(drug) * 100).round(2)
        role_stats.to_csv(OUTPUT_DIR / "03_drug_role_statistics.csv", index=False)
        print(f"  [OK] Estadísticas de roles guardadas")

    print("\n[5/5] Análisis de medicamentos completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
