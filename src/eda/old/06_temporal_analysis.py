"""
Script 06: Análisis Temporal
Objetivo: Tendencias de reportes a lo largo del tiempo, estacionalidad
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


def load_demo_dates() -> pd.DataFrame:
    """Carga fechas relevantes de la tabla DEMO."""
    filepath = DATA_DIR / FAERS_FILES["DEMO"]
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró: {filepath}")

    # Cargar solo columnas de fecha
    date_cols = ["primaryid", "event_dt", "rept_dt", "mfr_dt", "init_fda_dt", "fda_dt"]
    df = pd.read_csv(
        filepath,
        sep=FAERS_SEP,
        encoding=FAERS_ENCODING,
        low_memory=False,
        usecols=lambda x: x in date_cols
    )
    df.columns = df.columns.str.strip().str.lower()
    return df


def parse_faers_date(series: pd.Series) -> pd.Series:
    """
    Parsea fechas FAERS que vienen como enteros YYYYMMDD o strings.
    """
    s = series.astype(str).str.strip()
    # Intentar parsear como fecha completa YYYYMMDD
    mask_full = s.str.len() >= 8
    result = pd.Series(pd.NaT, index=series.index)

    if mask_full.any():
        parsed = pd.to_datetime(s[mask_full], format="%Y%m%d", errors="coerce")
        result[mask_full] = parsed

    # Para los que no tienen 8 dígitos, intentar YYYYMM
    mask_partial = (~mask_full) & (s.str.len() >= 6)
    if mask_partial.any():
        parsed = pd.to_datetime(s[mask_partial] + "01", format="%Y%m%d", errors="coerce")
        result[mask_partial] = parsed

    return result


def plot_monthly_trends(demo: pd.DataFrame):
    """Tendencia mensual de reportes por fecha de FDA."""
    if "fda_dt" not in demo.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 5))

    # Parsear fechas
    demo["fda_dt_parsed"] = parse_faers_date(demo["fda_dt"])
    demo["fda_month"] = demo["fda_dt_parsed"].dt.to_period("M")

    monthly_counts = demo.dropna(subset=["fda_month"]).groupby("fda_month").size()

    if len(monthly_counts) < 2:
        print("  [ADVERTENCIA] Datos insuficientes para tendencia temporal")
        return

    monthly_counts.index = monthly_counts.index.to_timestamp()
    ax.plot(monthly_counts.index, monthly_counts.values,
            marker="o", linewidth=2.5, markersize=6, color="teal")
    ax.fill_between(monthly_counts.index, monthly_counts.values, alpha=0.2, color="teal")

    ax.set_title("Evolución Temporal de Reportes FAERS (por mes)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de Reportes")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_monthly_trend.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")

    return monthly_counts


def plot_quarterly_comparison(demo: pd.DataFrame):
    """Comparación entre trimestres."""
    if "fda_dt" not in demo.columns:
        return

    demo["fda_dt_parsed"] = parse_faers_date(demo["fda_dt"])
    demo["quarter"] = demo["fda_dt_parsed"].dt.quarter
    demo["year"] = demo["fda_dt_parsed"].dt.year

    # Filtrar datos válidos
    valid = demo.dropna(subset=["fda_dt_parsed"])

    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    quarter_year = valid.groupby(["year", "quarter"]).size().unstack(fill_value=0)

    if len(quarter_year) > 0:
        quarter_year.plot(kind="bar", ax=ax, colormap=PALETTE_MAIN, edgecolor="white")
        ax.set_title("Reportes por Trimestre", fontsize=14, fontweight="bold")
        ax.set_xlabel("Año - Trimestre")
        ax.set_ylabel("Número de Reportes")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "06_quarterly_comparison.png"
        plt.savefig(output_path)
        plt.close()
        print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_time_lags(demo: pd.DataFrame):
    """Análisis de retrasos entre fechas clave."""
    date_cols = ["event_dt", "rept_dt", "mfr_dt", "init_fda_dt", "fda_dt"]
    if not all(c in demo.columns for c in date_cols[:3]):
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    pairs = [
        ("event_dt", "rept_dt", "Evento → Reporte"),
        ("rept_dt", "mfr_dt", "Reporte → Fabricante"),
        ("mfr_dt", "init_fda_dt", "Fabricante → FDA Inicial"),
        ("init_fda_dt", "fda_dt", "FDA Inicial → FDA Final")
    ]

    for idx, (col1, col2, title) in enumerate(pairs):
        ax = axes[idx]
        d1 = parse_faers_date(demo[col1])
        d2 = parse_faers_date(demo[col2])

        lag = (d2 - d1).dt.days
        lag = lag[(lag >= 0) & (lag <= 365)]  # Filtrar retrasos razonables

        if len(lag) > 0:
            ax.hist(lag.dropna(), bins=30, color="purple", edgecolor="white", alpha=0.85)
            ax.set_title(f"Retraso: {title}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Días")
            ax.set_ylabel("Frecuencia")
            ax.axvline(lag.median(), color="red", linestyle="--", linewidth=2,
                       label=f"Mediana: {lag.median():.0f} días")
            ax.legend()

    plt.suptitle("Distribución de Retrasos entre Fechas Clave", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_time_lags.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def plot_weekday_distribution(demo: pd.DataFrame):
    """Distribución de reportes por día de la semana."""
    if "rept_dt" not in demo.columns:
        return

    demo["rept_dt_parsed"] = parse_faers_date(demo["rept_dt"])
    valid = demo.dropna(subset=["rept_dt_parsed"])

    if len(valid) == 0:
        return

    valid["weekday"] = valid["rept_dt_parsed"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

    weekday_counts = valid["weekday"].value_counts().reindex(weekday_order, fill_value=0)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.bar(range(len(weekday_order)), weekday_counts.values,
                  color=sns.color_palette("crest", 7), edgecolor="white")
    ax.set_title("Reportes por Día de la Semana", fontsize=14, fontweight="bold")
    ax.set_xlabel("Día de la Semana")
    ax.set_ylabel("Número de Reportes")
    ax.set_xticks(range(len(weekday_order)))
    ax.set_xticklabels(weekday_es, rotation=45)

    for bar, val in zip(bars, weekday_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(weekday_counts)*0.01,
                f"{val:,}", ha="center", fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_weekday_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"  [OK] Gráfico guardado: {output_path.name}")


def main():
    print("=" * 70)
    print("  SCRIPT 06: ANÁLISIS TEMPORAL – DEMO")
    print("=" * 70)

    print("\n[1/5] Cargando fechas de tabla DEMO...")
    try:
        demo = load_demo_dates()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\n[2/5] Registros con fechas: {len(demo):,}")

    print("\n[3/5] Generando análisis temporal...")
    plot_monthly_trends(demo)
    plot_quarterly_comparison(demo)
    plot_time_lags(demo)
    plot_weekday_distribution(demo)

    # Estadísticas de temporales
    print("\n[4/5] Calculando estadísticas temporales...")
    demo["fda_dt_parsed"] = parse_faers_date(demo["fda_dt"])
    valid = demo.dropna(subset=["fda_dt_parsed"])

    if len(valid) > 0:
        stats_df = pd.DataFrame({
            "estadística": ["Primera fecha", "Última fecha", "Período (días)",
                            "Reportes/mes (prom)", "Reportes/mes (mediana)"],
            "valor": [
                valid["fda_dt_parsed"].min().strftime("%Y-%m-%d"),
                valid["fda_dt_parsed"].max().strftime("%Y-%m-%d"),
                (valid["fda_dt_parsed"].max() - valid["fda_dt_parsed"].min()).days,
                round(valid.groupby(valid["fda_dt_parsed"].dt.to_period("M")).size().mean(), 2),
                round(valid.groupby(valid["fda_dt_parsed"].dt.to_period("M")).size().median(), 2)
            ]
        })
        stats_path = OUTPUT_DIR / "06_temporal_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        print(f"  [OK] Estadísticas temporales guardadas")

    print("\n[5/5] Análisis temporal completado.")
    print("=" * 70)


if __name__ == "__main__":
    main()
