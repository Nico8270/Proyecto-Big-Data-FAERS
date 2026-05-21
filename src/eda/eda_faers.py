"""
=============================================================================
EDA - Sistema FAERS (FDA Adverse Event Reporting System)
=============================================================================
Proyecto: Farmacovigilancia a escala con Hadoop/MapReduce
Tablas principales: DEMO, DRUG, REAC
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────
# Ajusta esta ruta a la carpeta donde tienes los archivos .txt / .csv de FAERS
DATA_DIR = Path("./faers_data")

# Subcarpeta de salida para los gráficos
OUTPUT_DIR = Path("./eda_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta de colores corporativa para gráficos
PALETTE = "viridis"
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120

# Columnas mínimas esperadas por tabla
COLS_DEMO = ["primaryid", "caseid", "age", "age_cod", "sex", "wt", "wt_cod",
             "reporter_country", "event_dt", "rept_dt", "mfr_dt", "fda_dt",
             "occp_cod", "init_fda_dt", "outc_cod", "to_mfr", "quarter"]
COLS_DRUG = ["primaryid", "caseid", "drug_seq", "role_cod", "drugname",
             "val_vbm", "route", "dose_vbm", "cum_dose_chr", "cum_dose_unit",
             "dechal", "rechal", "lot_num", "exp_dt", "nda_num", "dose_amt",
             "dose_unit", "dose_form", "dose_freq"]
COLS_REAC = ["primaryid", "caseid", "pt", "drug_rec_act"]


# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
def load_table(filename: str, usecols: list = None, nrows: int = None) -> pd.DataFrame:
    """
    Carga una tabla FAERS desde un archivo delimitado por '$'.
    Ajusta sep=',' si tus archivos usan comas.
    """
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  [AVISO] Archivo no encontrado: {path}. Se omitirá.")
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        sep="$",           # delimitador FAERS ASCII estándar
        encoding="latin-1",
        low_memory=False,
        nrows=nrows,
        usecols=usecols if usecols else None,
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


print("=" * 60)
print("  CARGA DE DATOS FAERS")
print("=" * 60)

# Ajusta los nombres de archivos según el trimestre que uses, ej. DEMO24Q1.txt
demo = load_table("DEMO.txt")
drug = load_table("DRUG.txt")
reac = load_table("REAC.txt")

for name, df in [("DEMO", demo), ("DRUG", drug), ("REAC", reac)]:
    if not df.empty:
        print(f"  {name:6s} → {df.shape[0]:>10,} filas  |  {df.shape[1]:>3} columnas")


# ─────────────────────────────────────────────
# 2. RESUMEN ESTADÍSTICO GENERAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN ESTADÍSTICO")
print("=" * 60)

def describe_table(name: str, df: pd.DataFrame):
    if df.empty:
        return
    print(f"\n── {name} ──")
    print(f"  Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
    print(f"  Tipos de datos:\n{df.dtypes.value_counts().to_string()}")
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        print("\n  Estadísticas numéricas:")
        print(num_df.describe().T.to_string())

for name, df in [("DEMO", demo), ("DRUG", drug), ("REAC", reac)]:
    describe_table(name, df)


# ─────────────────────────────────────────────
# 3. ANÁLISIS DE VALORES FALTANTES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  VALORES FALTANTES")
print("=" * 60)

def missing_report(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    miss = df.isnull().sum()
    pct  = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": miss, "pct_missing": pct})
    report = report[report["missing"] > 0].sort_values("pct_missing", ascending=False)
    print(f"\n── {name} ── ({len(report)} columnas con nulos)")
    if not report.empty:
        print(report.head(15).to_string())
    return report

miss_demo = missing_report("DEMO", demo)
miss_drug = missing_report("DRUG", drug)
miss_reac = missing_report("REAC", reac)

# Gráfico: heatmap de nulos para DEMO
if not demo.empty:
    fig, ax = plt.subplots(figsize=(14, 4))
    miss_pct = demo.isnull().mean().sort_values(ascending=False).head(20)
    miss_pct.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("DEMO – % de valores nulos por columna (top 20)", fontsize=13)
    ax.set_ylabel("Proporción de nulos")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_nulos_demo.png")
    plt.close()
    print("\n  [OK] Gráfico guardado: 01_nulos_demo.png")


# ─────────────────────────────────────────────
# 4. DISTRIBUCIÓN DEMOGRÁFICA (DEMO)
# ─────────────────────────────────────────────
if not demo.empty:
    print("\n" + "=" * 60)
    print("  DISTRIBUCIÓN DEMOGRÁFICA (DEMO)")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 4a. Distribución por sexo
    if "sex" in demo.columns:
        sex_counts = demo["sex"].value_counts(dropna=False)
        sex_counts.plot(kind="bar", ax=axes[0], color=sns.color_palette(PALETTE, len(sex_counts)),
                        edgecolor="white")
        axes[0].set_title("Distribución por Sexo")
        axes[0].set_xlabel("Sexo")
        axes[0].set_ylabel("Número de reportes")
        axes[0].tick_params(axis="x", rotation=0)

    # 4b. Distribución de edad
    if "age" in demo.columns:
        age = pd.to_numeric(demo["age"], errors="coerce")
        age = age[(age >= 0) & (age <= 120)]
        axes[1].hist(age.dropna(), bins=40, color="teal", edgecolor="white", alpha=0.85)
        axes[1].set_title("Distribución de Edad de los Pacientes")
        axes[1].set_xlabel("Edad (años)")
        axes[1].set_ylabel("Frecuencia")

    # 4c. Top 10 países reportantes
    if "reporter_country" in demo.columns:
        top_countries = demo["reporter_country"].value_counts().head(10)
        top_countries.plot(kind="barh", ax=axes[2],
                           color=sns.color_palette(PALETTE, 10), edgecolor="white")
        axes[2].set_title("Top 10 Países Reportantes")
        axes[2].set_xlabel("Número de reportes")
        axes[2].invert_yaxis()

    plt.suptitle("Perfil Demográfico – FAERS DEMO", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_demograficos.png")
    plt.close()
    print("  [OK] Gráfico guardado: 02_demograficos.png")

    # 4d. Distribución de outcomes
    if "outc_cod" in demo.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        outc_map = {
            "DE": "Muerte", "LT": "Amenaza vida", "HO": "Hospitalización",
            "DS": "Discapacidad", "CA": "Anomalía congénita",
            "RI": "Intervención requerida", "OT": "Otro"
        }
        outc = demo["outc_cod"].map(outc_map).fillna("Desconocido").value_counts()
        outc.plot(kind="bar", ax=ax, color=sns.color_palette("Reds_r", len(outc)), edgecolor="white")
        ax.set_title("Distribución de Outcomes Clínicos", fontsize=13)
        ax.set_xlabel("Tipo de outcome")
        ax.set_ylabel("Número de reportes")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_outcomes.png")
        plt.close()
        print("  [OK] Gráfico guardado: 03_outcomes.png")


# ─────────────────────────────────────────────
# 5. ANÁLISIS DE MEDICAMENTOS (DRUG)
# ─────────────────────────────────────────────
if not drug.empty:
    print("\n" + "=" * 60)
    print("  ANÁLISIS DE MEDICAMENTOS (DRUG)")
    print("=" * 60)

    # 5a. Top 20 fármacos más reportados
    if "drugname" in drug.columns:
        top_drugs = (drug["drugname"].str.upper().str.strip()
                     .value_counts().head(20))
        print(f"\n  Top 5 fármacos:\n{top_drugs.head().to_string()}")

        fig, ax = plt.subplots(figsize=(12, 6))
        top_drugs.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("Top 20 Fármacos más Reportados", fontsize=13)
        ax.set_xlabel("Número de registros")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_top20_farmacos.png")
        plt.close()
        print("  [OK] Gráfico guardado: 04_top20_farmacos.png")

    # 5b. Distribución por rol del fármaco (PS/SS/C/I)
    if "role_cod" in drug.columns:
        role_map = {"PS": "Sospechoso primario", "SS": "Sospechoso secundario",
                    "C": "Concomitante", "I": "Interactuante"}
        roles = drug["role_cod"].map(role_map).fillna("Otro").value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        roles.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(roles)), edgecolor="white")
        ax.set_title("Rol del Fármaco en el Reporte")
        ax.set_ylabel("Registros")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_rol_farmaco.png")
        plt.close()
        print("  [OK] Gráfico guardado: 05_rol_farmaco.png")

    # 5c. Vías de administración más frecuentes
    if "route" in drug.columns:
        top_routes = drug["route"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        top_routes.plot(kind="barh", ax=ax,
                        color=sns.color_palette("coolwarm", len(top_routes)), edgecolor="white")
        ax.set_title("Top 15 Vías de Administración", fontsize=13)
        ax.set_xlabel("Número de registros")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_vias_admin.png")
        plt.close()
        print("  [OK] Gráfico guardado: 06_vias_admin.png")


# ─────────────────────────────────────────────
# 6. ANÁLISIS DE REACCIONES ADVERSAS (REAC)
# ─────────────────────────────────────────────
if not reac.empty:
    print("\n" + "=" * 60)
    print("  ANÁLISIS DE REACCIONES ADVERSAS (REAC)")
    print("=" * 60)

    if "pt" in reac.columns:
        top_reac = reac["pt"].str.strip().str.title().value_counts().head(25)
        print(f"\n  Top 5 reacciones:\n{top_reac.head().to_string()}")

        fig, ax = plt.subplots(figsize=(12, 8))
        top_reac.plot(kind="barh", ax=ax, color="darkorange", edgecolor="white")
        ax.set_title("Top 25 Reacciones Adversas más Reportadas (MedDRA PT)", fontsize=13)
        ax.set_xlabel("Número de reportes")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "07_top25_reacciones.png")
        plt.close()
        print("  [OK] Gráfico guardado: 07_top25_reacciones.png")

    # Número de reacciones por caso
    if "primaryid" in reac.columns:
        reac_per_case = reac.groupby("primaryid")["pt"].count()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(reac_per_case.clip(upper=20), bins=20, color="orchid", edgecolor="white", alpha=0.85)
        ax.set_title("Número de Reacciones Adversas por Caso", fontsize=13)
        ax.set_xlabel("Cantidad de reacciones (limitado a 20)")
        ax.set_ylabel("Número de casos")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "08_reacciones_por_caso.png")
        plt.close()
        print("  [OK] Gráfico guardado: 08_reacciones_por_caso.png")


# ─────────────────────────────────────────────
# 7. ANÁLISIS TEMPORAL (DEMO)
# ─────────────────────────────────────────────
if not demo.empty and "fda_dt" in demo.columns:
    print("\n" + "=" * 60)
    print("  TENDENCIAS TEMPORALES")
    print("=" * 60)

    demo["fda_dt_parsed"] = pd.to_datetime(
        demo["fda_dt"].astype(str).str[:6], format="%Y%m", errors="coerce"
    )
    monthly = demo.dropna(subset=["fda_dt_parsed"]).groupby(
        demo["fda_dt_parsed"].dt.to_period("M")
    ).size()

    if len(monthly) > 1:
        fig, ax = plt.subplots(figsize=(14, 4))
        monthly.plot(ax=ax, color="teal", linewidth=1.5)
        ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color="teal")
        ax.set_title("Evolución Temporal de Reportes FAERS (por mes)", fontsize=13)
        ax.set_xlabel("Período")
        ax.set_ylabel("Número de reportes")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "09_tendencia_temporal.png")
        plt.close()
        print("  [OK] Gráfico guardado: 09_tendencia_temporal.png")


# ─────────────────────────────────────────────
# 8. ANÁLISIS CRUZADO: DRUG + REAC (JOIN)
# ─────────────────────────────────────────────
if not drug.empty and not reac.empty:
    print("\n" + "=" * 60)
    print("  ANÁLISIS CRUZADO DRUG × REAC")
    print("=" * 60)

    # JOIN por primaryid
    merged = pd.merge(
        drug[["primaryid", "drugname", "role_cod"]].query("role_cod == 'PS'"),
        reac[["primaryid", "pt"]],
        on="primaryid",
        how="inner"
    )
    merged["drugname"] = merged["drugname"].str.upper().str.strip()
    merged["pt"] = merged["pt"].str.strip().str.title()

    print(f"  Registros tras JOIN (fármacos sospechosos × reacciones): {len(merged):,}")

    # Heatmap top 10 fármacos × top 10 reacciones
    top10_drugs = merged["drugname"].value_counts().head(10).index
    top10_reac  = merged["pt"].value_counts().head(10).index

    pivot = (merged
             .loc[merged["drugname"].isin(top10_drugs) & merged["pt"].isin(top10_reac)]
             .groupby(["drugname", "pt"])
             .size()
             .unstack(fill_value=0))

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=0.4, linecolor="white", ax=ax)
        ax.set_title("Heatmap: Fármaco (PS) × Reacción Adversa (Top 10 × Top 10)", fontsize=13)
        ax.set_xlabel("Reacción Adversa (MedDRA PT)")
        ax.set_ylabel("Fármaco")
        ax.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "10_heatmap_drug_reac.png")
        plt.close()
        print("  [OK] Gráfico guardado: 10_heatmap_drug_reac.png")

    # Top 10 pares (fármaco, reacción)
    top_pairs = (merged.groupby(["drugname", "pt"])
                 .size()
                 .reset_index(name="count")
                 .sort_values("count", ascending=False)
                 .head(10))
    print(f"\n  Top 10 pares Fármaco–Reacción:\n{top_pairs.to_string(index=False)}")


# ─────────────────────────────────────────────
# 9. DETECCIÓN DE DUPLICADOS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DETECCIÓN DE DUPLICADOS")
print("=" * 60)

for name, df in [("DEMO", demo), ("DRUG", drug), ("REAC", reac)]:
    if df.empty:
        continue
    n_dup = df.duplicated().sum()
    print(f"  {name}: {n_dup:,} filas duplicadas ({n_dup / len(df) * 100:.2f}%)")
    if "primaryid" in df.columns:
        n_dup_id = df["primaryid"].duplicated().sum()
        print(f"         {n_dup_id:,} primaryid duplicados")


# ─────────────────────────────────────────────
# 10. REPORTE FINAL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN EDA – FAERS")
print("=" * 60)
for name, df in [("DEMO", demo), ("DRUG", drug), ("REAC", reac)]:
    if not df.empty:
        print(f"  {name:6s} | Filas: {df.shape[0]:>10,} | Columnas: {df.shape[1]:>3} "
              f"| Nulos totales: {df.isnull().sum().sum():>10,}")

print(f"\n  Gráficos guardados en: {OUTPUT_DIR.resolve()}/")
print("=" * 60)
print("  EDA completado exitosamente.")
print("=" * 60)
