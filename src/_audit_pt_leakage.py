"""
_audit_pt_leakage.py  (temporary — delete after audit)
========================================================
Runs the full target-leakage audit for pt_encoded vs severity_level.
Steps:
  1. MI table BEFORE fix (on X_train only)
  2. Identify leaking PT terms (entropy < 0.5 measured on X_train)
  3. Mask leaking PTs with sentinel, re-encode
  4. MI table AFTER fix
  5. Print comparison
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
PROJECT_ROOT = Path(__file__).resolve().parent
df = pd.read_parquet(PROJECT_ROOT / "data" / "clean_data" / "dataset_consolidado.parquet")
THRESHOLD = 0.7   # MI leakage threshold (user-specified)
ENT_CUTOFF = 0.5  # entropy below which a PT term is flagged as an outcome proxy
# ── Build feature matrix ──────────────────────────────────────────────────────
age_col  = pd.to_numeric(df["age"], errors="coerce").fillna(45.0)
sex_col  = df["sex"].fillna("U").astype(str).str.upper().str.strip()
sex_col  = sex_col.map({"M": 1.0, "F": 0.0, "U": 0.5}).fillna(0.5)
drug_col = df["drug_encoded"].fillna(-1)
pt_col   = df["pt_encoded"].fillna(-1)
X = pd.DataFrame({
    "age":          age_col.values,
    "sex_encoded":  sex_col.values,
    "drug_encoded": drug_col.values,
    "pt_encoded":   pt_col.values,
})
y = df["severity_level"].astype(int).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
df_reset = df.reset_index(drop=True)
# ── STEP 1: Mutual Information BEFORE any fix ─────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 1 — Mutual Information BEFORE fix (computed on X_train)")
print("=" * 60)
mi_before = mutual_info_classif(
    X_train, y_train, random_state=42,
    discrete_features=[False, False, True, True],
)
mi_before_df = pd.DataFrame({
    "feature":   list(X_train.columns),
    "MI_before": mi_before,
}).sort_values("MI_before", ascending=False).reset_index(drop=True)
print()
print(f"  {'Feature':<20} {'MI Score':>10}  {'Exceeds 0.7?':>14}")
print("  " + "-" * 48)
for _, row in mi_before_df.iterrows():
    flag = "*** LEAKING ***" if row["MI_before"] > THRESHOLD else ""
    print(f"  {row['feature']:<20} {row['MI_before']:>10.4f}  {flag}")
print()
leaking_features = mi_before_df[mi_before_df["MI_before"] > THRESHOLD]["feature"].tolist()
if leaking_features:
    print(f"  [FLAG] Features exceeding MI threshold ({THRESHOLD}): {leaking_features}")
else:
    print(f"  [OK] No feature exceeds MI threshold of {THRESHOLD}.")
# ── STEP 2: Identify leaking PT terms by per-term entropy ────────────────────
print("\n" + "=" * 60)
print("  STEP 2 — Per-PT-term entropy analysis on X_train")
print("=" * 60)
print(f"  Entropy cutoff for leakage flag: < {ENT_CUTOFF}")
print()
pt_raw_train = df_reset.loc[X_train.index, "pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
sev_train_vals = y_train.values
pt_entropy_map = {}
for pt, grp in pd.DataFrame({"pt": pt_raw_train.values, "sev": sev_train_vals}).groupby("pt")["sev"]:
    vc = grp.value_counts(normalize=True)
    h = float(-(vc * np.log2(vc + 1e-10)).sum())
    pt_entropy_map[pt] = h
pt_entropy_s = pd.Series(pt_entropy_map).sort_values()
leaking_pts = pt_entropy_s[pt_entropy_s < ENT_CUTOFF].index.tolist()
print(f"  Total unique PT terms in X_train : {len(pt_entropy_s)}")
print(f"  Zero-entropy (perfect proxy)     : {(pt_entropy_s < 0.001).sum()}")
print(f"  Near-zero entropy (< 0.5)        : {(pt_entropy_s < ENT_CUTOFF).sum()}")
print(f"  High entropy (>= 1.5) — safe     : {(pt_entropy_s >= 1.5).sum()}")
print()
print(f"  [FLAG] {len(leaking_pts)} PT terms flagged as outcome-proxy leakers.")
print()
print("  Top leaking PT terms (lowest entropy = most deterministic):")
for pt, h in pt_entropy_s[pt_entropy_s < ENT_CUTOFF].head(25).items():
    n = (pt_raw_train == pt).sum()
    dom_sev = pd.DataFrame({"pt": pt_raw_train.values, "sev": sev_train_vals})
    dom_sev = dom_sev[dom_sev["pt"] == pt]["sev"].value_counts().idxmax()
    print(f"    {pt:<50s}  H={h:.3f}  n={n:5d}  dominant_sev={dom_sev}")
print()
print("  Root cause: 'pt' is a MedDRA Preferred Term from REAC table.")
print("  'severity_level' is derived DETERMINISTICALLY from 'outc_cod' (OUTC table).")
print("  When FAERS records share primaryid, the join leaks outc_cod signal into 'pt'")
print("  because certain clinical outcomes (e.g., DE=Death) co-occur with specific PTs")
print("  (e.g., 'DEATH', 'CARDIAC ARREST') at near-100% rates.")
# ── STEP 3: Apply fix — mask leaking PT category levels ──────────────────────
print("\n" + "=" * 60)
print("  STEP 3 — Fix: mask leaking PT terms with sentinel 'OUTCOME_PROXY'")
print("=" * 60)
print()
print("  Strategy chosen: Option (a) — remove only leaking category levels.")
print("  Non-leaking PT terms retain their identity (preserving granularity).")
print()
print("  Why NOT SOC mapping (option b):")
print("  The FAERS pipeline uses custom synonym lists (rules_cleaning.py), not")
print("  a full MedDRA hierarchy. SOC codes are unavailable in the dataset.")
print("  Masking leakers is safer and still clinically meaningful.")
print()
# Mask leaking PTs in the full df (train + test) — mask derived only from train
SENTINEL = "OUTCOME_PROXY"
pt_full = df_reset["pt"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
pt_masked = pt_full.copy()
pt_masked[pt_masked.isin(leaking_pts)] = SENTINEL
# Fit OrdinalEncoder ONLY on masked train set
pt_train_masked = np.array(pt_masked.iloc[X_train.index.tolist()].tolist(), dtype=str).reshape(-1, 1)
pt_test_masked  = np.array(pt_masked.iloc[X_test.index.tolist()].tolist(),  dtype=str).reshape(-1, 1)
enc_fixed = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc_fixed.fit(pt_train_masked)   # fit ONLY on train
X_train_fixed = X_train.copy()
X_test_fixed  = X_test.copy()
X_train_fixed["pt_encoded"] = enc_fixed.transform(pt_train_masked).ravel()
X_test_fixed["pt_encoded"]  = enc_fixed.transform(pt_test_masked).ravel()
n_masked_train = (pt_masked.iloc[X_train.index.tolist()] == SENTINEL).sum()
n_masked_test  = (pt_masked.iloc[X_test.index.tolist()] == SENTINEL).sum()
print(f"  Rows with PT masked in X_train : {n_masked_train:,}")
print(f"  Rows with PT masked in X_test  : {n_masked_test:,}")
print(f"  Non-leaking PT terms preserved : {len(enc_fixed.categories_[0]) - 1}")
# ── STEP 4: Mutual Information AFTER fix ─────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4 — Mutual Information AFTER fix (on X_train_fixed)")
print("=" * 60)
mi_after = mutual_info_classif(
    X_train_fixed, y_train, random_state=42,
    discrete_features=[False, False, True, True],
)
mi_after_df = pd.DataFrame({
    "feature":  list(X_train_fixed.columns),
    "MI_after": mi_after,
}).sort_values("MI_after", ascending=False).reset_index(drop=True)
print()
print(f"  {'Feature':<20} {'MI Score':>10}  {'Status':>20}")
print("  " + "-" * 55)
for _, row in mi_after_df.iterrows():
    status = "OK (below 0.7)" if row["MI_after"] <= THRESHOLD else "STILL LEAKING"
    print(f"  {row['feature']:<20} {row['MI_after']:>10.4f}  {status}")
# ── STEP 5: Comparison table ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5 — Before / After comparison")
print("=" * 60)
cmp = mi_before_df.merge(mi_after_df, on="feature")
cmp["delta"] = cmp["MI_after"] - cmp["MI_before"]
cmp["delta_pct"] = cmp["delta"] / (cmp["MI_before"] + 1e-10) * 100
print()
print(f"  {'Feature':<20} {'MI Before':>10} {'MI After':>10} {'Delta':>10} {'Change':>10}")
print("  " + "-" * 65)
for _, row in cmp.sort_values("MI_before", ascending=False).iterrows():
    print(f"  {row['feature']:<20} {row['MI_before']:>10.4f} {row['MI_after']:>10.4f} "
          f"{row['delta']:>+10.4f} {row['delta_pct']:>+9.1f}%")
pt_before = mi_before_df[mi_before_df["feature"] == "pt_encoded"]["MI_before"].values[0]
pt_after  = mi_after_df[mi_after_df["feature"]  == "pt_encoded"]["MI_after"].values[0]
print()
print(f"  pt_encoded MI before fix : {pt_before:.4f}")
print(f"  pt_encoded MI after fix  : {pt_after:.4f}")
print(f"  Reduction                : {(pt_before-pt_after)/pt_before*100:.1f}%")
print()
if pt_after <= THRESHOLD:
    print("  [CONFIRMED] pt_encoded is now below the 0.7 leakage threshold.")
else:
    print("  [WARNING] pt_encoded still exceeds threshold — additional masking needed.")
# ── Save leaking PT list for use in stage 02 ─────────────────────────────────
out_path = PROJECT_ROOT / "outputs" / "model_assets" / "leaking_pt_terms.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(sorted(leaking_pts), f, indent=2)
print()
print(f"  Leaking PT list saved to: {out_path.relative_to(PROJECT_ROOT)}")
print("  (Used by 02_feature_engineering.py to mask at fit time)")
print()
print("=" * 60)
print("  AUDIT COMPLETE")
print("=" * 60)