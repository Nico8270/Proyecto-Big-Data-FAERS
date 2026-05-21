"""
Script 08: Resumen Ejecutivo y Estadísticas Clave
Objetivo: Consolidar hallazgos principales en un solo reporte
"""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *


def load_all_tables() -> dict:
    """Carga todas las tablas FAERS."""
    tables = {}
    for name, filename in FAERS_FILES.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            continue

        print(f"  Cargando {name}...", end=" ", flush=True)
        try:
            df = pd.read_csv(filepath, sep=FAERS_SEP, encoding=FAERS_ENCODING, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()
            print(f"{len(df):,} filas × {df.shape[1]} cols")
            tables[name] = df
        except Exception as e:
            print(f"[ERROR] {e}")

    return tables


def generate_insights(tables: dict):
    """Genera insights principales."""
    insights = []

    # ── DEMO ──
    if "DEMO" in tables:
        demo = tables["DEMO"]
        insights.append("=" * 70)
        insights.append("  DEMO – Datos Demográficos")
        insights.append("=" * 70)
        insights.append(f"  • Total de reportes únicos: {demo['primaryid'].nunique():,}")
        insights.append(f"  • Período cubierto: {demo.get('rept_dt', pd.Series()).astype(str).min()} – {demo.get('rept_dt', pd.Series()).astype(str).max()}")
        if "age" in demo.columns:
            ages = pd.to_numeric(demo["age"], errors="coerce")
            ages = ages[(ages >= 0) & (ages <= 120)]
            insights.append(f"  • Edad promedio: {ages.mean():.1f} años")
            insights.append(f"  • Rango de edad: {ages.min():.0f} – {ages.max():.0f} años")
        if "sex" in demo.columns:
            sex_dist = demo["sex"].value_counts(dropna=False)
            for sex, count in sex_dist.items():
                insights.append(f"  • Género {sex}: {count:,} reportes ({count/len(demo)*100:.1f}%)")
        insights.append("")

    # ── DRUG ──
    if "DRUG" in tables:
        drug = tables["DRUG"]
        insights.append("=" * 70)
        insights.append("  DRUG – Medicamentos")
        insights.append("=" * 70)
        insights.append(f"  • Total de registros de medicamentos: {len(drug):,}")
        insights.append(f"  • Casos únicos con medicamentos: {drug['primaryid'].nunique():,}")
        insights.append(f"  • Fármacos únicos: {drug['drugname'].nunique():,}")
        if "role_cod" in drug.columns:
            role_counts = drug["role_cod"].value_counts()
            for role, count in role_counts.items():
                pct = count / len(drug) * 100
                insights.append(f"  • Rol {role} ({DRUG_ROLE_MAP.get(role, role)}): {count:,} ({pct:.1f}%)")
        insights.append("")

    # ── REAC ──
    if "REAC" in tables:
        reac = tables["REAC"]
        insights.append("=" * 70)
        insights.append("  REAC – Reacciones Adversas")
        insights.append("=" * 70)
        insights.append(f"  • Total de reacciones reportadas: {len(reac):,}")
        insights.append(f"  • Casos con al menos 1 reacción: {reac['primaryid'].nunique():,}")
        insights.append(f"  • Reacciones únicas (PT): {reac['pt'].nunique():,}")

        # Top 5 reacciones
        top5 = reac["pt"].value_counts().head(5)
        insights.append("  • Top 5 reacciones:")
        for i, (react, count) in enumerate(top5.items(), 1):
            insights.append(f"    {i}. {react}: {count:,}")
        insights.append("")

    # ── OUTC ──
    if "OUTC" in tables:
        outc = tables["OUTC"]
        insights.append("=" * 70)
        insights.append("  OUTC – Resultados Clínicos")
        insights.append("=" * 70)
        outc_counts = outc["outc_cod"].value_counts()
        for code, count in outc_counts.items():
            pct = count / len(outc) * 100
            desc = OUTCOME_MAP.get(code, code)
            insights.append(f"  • {desc} ({code}): {count:,} ({pct:.1f}%)")
        insights.append("")

    # ── INDI ──
    if "INDI" in tables:
        indi = tables["INDI"]
        insights.append("=" * 70)
        insights.append("  INDI – Indicaciones Terapéuticas")
        insights.append("=" * 70)
        insights.append(f"  • Total de indicaciones reportadas: {len(indi):,}")
        insights.append(f"  • Indicaciones únicas: {indi['indi_pt'].nunique():,}")
        top5_indi = indi["indi_pt"].value_counts().head(5)
        insights.append("  • Top 5 indicaciones:")
        for i, (indi_pt, count) in enumerate(top5_indi.items(), 1):
            insights.append(f"    {i}. {indi_pt}: {count:,}")
        insights.append("")

    # ── Análisis de relación ──
    if "DRUG" in tables and "REAC" in tables:
        drug_ps = tables["DRUG"][tables["DRUG"]["role_cod"] == "PS"]
        merged = pd.merge(drug_ps[["primaryid", "drugname"]],
                          tables["REAC"][["primaryid", "pt"]],
                          on="primaryid", how="inner")
        insights.append("=" * 70)
        insights.append("  CRUCE DRUG × REAC (solo fármacos sospechosos primarios)")
        insights.append("=" * 70)
        insights.append(f"  • Pares únicos fármaco–reacción: {merged.groupby(['drugname','pt']).size().shape[0]:,}")

        top_pair = merged.groupby(["drugname", "pt"]).size().sort_values(ascending=False).head(1)
        if len(top_pair) > 0:
            (drug_name, reac_name), count = next(top_pair.items())
            insights.append(f"  • Par más frecuente: {drug_name} → {reac_name} ({count:,} reportes)")
        insights.append("")

    return insights


def main():
    print("=" * 70)
    print("  SCRIPT 08: RESUMEN EJECUTIVO – FAERS Q4 2025")
    print("=" * 70)

    print("\n[1/3] Cargando todas las tablas FAERS...")
    tables = load_all_tables()

    print("\n[2/3] Generando insights clave...")
    insights = generate_insights(tables)

    # Guardar insights
    insights_path = OUTPUT_DIR / "08_executive_summary.txt"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write("\n".join(insights))

    print(f"  [OK] Resumen guardado: {insights_path.name}")

    # También imprimir en consola
    print("\n" + "=" * 70)
    print("  INSIGHTS PRINCIPALES (también guardados en archivo)")
    print("=" * 70)
    for line in insights:
        print(line)

    print("\n[3/3] Resumen ejecutivo completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
