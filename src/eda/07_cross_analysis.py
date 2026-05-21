"""
Script 07: Análisis Cruzado y Combinado – DRUG + REAC + OUTC
Objetivo: Heatmaps de fármaco-reacción, correlaciones, pares frecuentes
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


def load_tables() -> dict:
    """Carga las tablas necesarias para análisis cruzado."""
    tables = {}
    needed = ["DEMO", "DRUG", "REAC", "OUTC", "INDI"]

    for name in needed:
        filepath = DATA_DIR / FAERS_FILES[name]
        if not filepath.exists():
            print(f"  [ADVERTENCIA] No se encontró: {filepath.name}")
            continue

        print(f"  Cargando {name}...", end=" ", flush=True)
        df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        print(f"{len(df):,} filas")
        tables[name] = df

    return tables


def plot_drug_reaction_heatmap(drug: pd.DataFrame, reac: pd.DataFrame, top_n: int = 10):
    """Heatmap de fármaco (sospechoso) × reacción."""
    if "drugname" not in drug.columns or "pt" not in reac.columns:
        return

    # Filtrar solo fármacos sospechosos primarios
    drug_ps = drug[drug["role_cod"] == "PS"].copy()
    drug_ps["drugname_clean"] = drug_ps["drugname"].str.upper().str.strip()

    merged = pd.merge(
        drug_ps[["primaryid", "drugname_clean"]],
        reac[["primaryid", "pt"]],
        on="primaryid",
        how="inner"
    )

    if len(merged) == 0:
        return

    # Top N fármacos y reacciones
    top_drugs = merged["drugname_clean"].value_counts().head(top_n).index
    top_reac = merged["pt"].value_counts().head(top_n).index

    filtered = merged[
        merged["drugname_clean"].isin(top_drugs) &
        merged["pt"].isin(top_reac)
    ]

    pivot = filtered.groupby(["drugname_clean", "pt"]).size().unstack(fill_value=0)

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "Número de Reportes"})
    ax.set_title(f"Heatmap: Top {top_n} Fármacos (PS) × Top {top_n} Reacciones",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Reacción Adversa (MedDRA PT)")
    ax.set_ylabel("Fármaco Sospechoso")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"07_heatmap_drug_reac_top{top_n}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Heatmap guardado: {output_path.name}")

    # Guardar pares en CSV
    top_pairs = (merged.groupby(["drugname_clean", "pt"])
                  .size()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False)
                  .head(50))
    top_pairs.to_csv(OUTPUT_DIR / "07_top50_drug_reaction_pairs.csv", index=False)
    print(f"  [OK] Top 50 pares fármaco-reacción guardados en CSV")


def plot_drug_outcome_association(drug: pd.DataFrame, outc: pd.DataFrame, top_n: int = 10):
    """Asociación entre fármacos y outcomes graves."""
    drug_ps = drug[drug["role_cod"] == "PS"].copy()
    drug_ps["drugname_clean"] = drug_ps["drugname"].str.upper().str.strip()

    merged = pd.merge(
        drug_ps[["primaryid", "drugname_clean"]],
        outc[["primaryid", "outc_cod"]],
        on="primaryid",
        how="inner"
    )

    if len(merged) == 0:
        return

    # Top fármacos
    top_drugs = merged["drugname_clean"].value_counts().head(top_n).index
    filtered = merged[merged["drugname_clean"].isin(top_drugs)]

    # Calcular proporción de outcomes por fármaco
    cross_tab = pd.crosstab(filtered["drugname_clean"], filtered["outc_cod"], normalize="index") * 100
    cross_tab.columns = [OUTCOME_MAP.get(col, col) for col in cross_tab.columns]

    fig, ax = plt.subplots(figsize=(14, 8))
    cross_tab.plot(kind="bar", ax=ax, colormap=PALETTE_DIVERG, edgecolor="white", width=0.8)
    ax.set_title(f"Proporción de Outcomes por Fármaco (Top {top_n})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fármaco Sospechoso")
    ax.set_ylabel("Porcentaje de Reportes (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"07_drug_outcome_association_top{top_n}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_multidrug_cases(drug: pd.DataFrame):
    """Distribución del número de medicamentos por caso."""
    if "primaryid" not in drug.columns:
        return

    drugs_per_case = drug.groupby("primaryid")["drug_seq"].nunique()

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.hist(drugs_per_case.clip(upper=10), bins=range(1, 11),
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title("Número de Medicamentos Diferentes por Caso", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cantidad de Medicamentos")
    ax.set_ylabel("Número de Casos")
    ax.set_xticks(range(1, 11))

    stats_text = f"Media: {drugs_per_case.mean():.2f} | Mediana: {drugs_per_case.median():.0f}"
    ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    output_path = OUTPUT_DIR / "07_drugs_per_case.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_indication_distribution(indi: pd.DataFrame, top_n: int = 20):
    """Indicaciones terapéuticas más frecuentes."""
    if "indi_pt" not in indi.columns:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    top_indi = indi["indi_pt"].fillna("UNK").value_counts().head(top_n)

    colors = sns.color_palette("magma", len(top_indi))
    top_indi.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Top {top_n} Indicaciones Terapéuticas", fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Reportes")
    ax.set_ylabel("Indicación")
    ax.invert_yaxis()

    for i, v in enumerate(top_indi):
        ax.text(v + max(top_indi)*0.01, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / f"07_top{top_n}_indications.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 07: ANÁLISIS CRUZADO Y COMBINADO")
    print("=" * 70)

    print("\n[1/5] Cargando tablas necesarias...")
    tables = load_tables()

    drug = tables.get("DRUG")
    reac = tables.get("REAC")
    outc = tables.get("OUTC")
    indi = tables.get("INDI")
    demo = tables.get("DEMO")

    if drug is None or reac is None:
        print("[ERROR] No se pueden continuar sin DRUG y REAC")
        sys.exit(1)

    print("\n[2/5] Generando análisis cruzado fármaco-reacción...")
    plot_drug_reaction_heatmap(drug, reac, top_n=10)

    print("\n[3/5] Analizando asociación fármaco-outcome...")
    if outc is not None:
        plot_drug_outcome_association(drug, outc, top_n=10)

    print("\n[4/5] Analizando polimedicación...")
    plot_multidrug_cases(drug)

    if indi is not None:
        print("\n[4b/5] Analizando indicaciones terapéuticas...")
        plot_indication_distribution(indi, top_n=20)

    # Guardar resumen
    print("\n[5/5] Guardando resumen de pares frecuentes...")
    if drug is not None and reac is not None:
        drug_ps = drug[drug["role_cod"] == "PS"].copy()
        drug_ps["drugname_clean"] = drug_ps["drugname"].str.upper().str.strip()

        merged = pd.merge(
            drug_ps[["primaryid", "drugname_clean"]],
            reac[["primaryid", "pt"]],
            on="primaryid",
            how="inner"
        )

        top_pairs = (merged.groupby(["drugname_clean", "pt"])
                      .size()
                      .reset_index(name="count")
                      .sort_values("count", ascending=False))

        top_pairs.to_csv(OUTPUT_DIR / "07_all_drug_reaction_pairs.csv", index=False)
        print(f"  [OK] Todos los pares guardados en CSV ({len(top_pairs):,} combinaciones)")

    print("\n[5/5] Análisis cruzado completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
