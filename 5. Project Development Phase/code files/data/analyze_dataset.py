"""
Deep Statistical Analysis of the Original Flood Dataset
========================================================
Analyzes distributions, correlations, skewness, outliers, and class imbalance.
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_excel('data/flood_dataset.xlsx')

print("=" * 70)
print("1. COLUMN TYPES")
print("=" * 70)
print(df.dtypes)
print(f"\nNumerical columns: {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical columns: {df.select_dtypes(include=['object']).columns.tolist()}")

print("\n" + "=" * 70)
print("2. DESCRIPTIVE STATISTICS (Full)")
print("=" * 70)
print(df.describe().to_string())

print("\n" + "=" * 70)
print("3. CLASS DISTRIBUTION (Target: flood)")
print("=" * 70)
counts = df['flood'].value_counts()
print(counts)
print(f"\nImbalance ratio (majority/minority): {counts.max() / counts.min():.2f}")
print(f"Minority class %: {counts.min() / len(df) * 100:.1f}%")

print("\n" + "=" * 70)
print("4. CLASS-WISE STATISTICS")
print("=" * 70)
for cls in [0, 1]:
    subset = df[df['flood'] == cls]
    print(f"\n--- flood = {cls} (n={len(subset)}) ---")
    print(subset.describe().to_string())

print("\n" + "=" * 70)
print("5. SKEWNESS")
print("=" * 70)
skew = df.skew(numeric_only=True)
print(skew.to_string())
print("\nHighly skewed (|skew| > 1):", skew[skew.abs() > 1].index.tolist())

print("\n" + "=" * 70)
print("6. PEARSON CORRELATION MATRIX")
print("=" * 70)
corr_pearson = df.corr(method='pearson')
print(corr_pearson.round(3).to_string())

print("\n" + "=" * 70)
print("7. SPEARMAN CORRELATION MATRIX")
print("=" * 70)
corr_spearman = df.corr(method='spearman')
print(corr_spearman.round(3).to_string())

print("\n" + "=" * 70)
print("8. FEATURE CORRELATIONS WITH TARGET (flood)")
print("=" * 70)
for col in df.columns:
    if col != 'flood':
        p_corr, p_pval = stats.pearsonr(df[col], df['flood'])
        s_corr, s_pval = stats.spearmanr(df[col], df['flood'])
        print(f"  {col:15s}  Pearson={p_corr:+.4f} (p={p_pval:.4f})  Spearman={s_corr:+.4f} (p={s_pval:.4f})")

print("\n" + "=" * 70)
print("9. OUTLIER DETECTION (IQR Method)")
print("=" * 70)
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    if len(outliers) > 0:
        print(f"  {col:15s}: {len(outliers)} outliers (range [{lower:.1f}, {upper:.1f}])")
    else:
        print(f"  {col:15s}: No outliers")

print("\n" + "=" * 70)
print("10. DISTRIBUTION SHAPE TESTS (Shapiro-Wilk Normality)")
print("=" * 70)
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'flood':
        stat, pval = stats.shapiro(df[col])
        normal = "YES" if pval > 0.05 else "NO"
        print(f"  {col:15s}  W={stat:.4f}  p={pval:.6f}  Normal? {normal}")

print("\n" + "=" * 70)
print("11. SEASONAL RAINFALL RELATIONSHIPS")
print("=" * 70)
seasonal_cols = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
df['seasonal_sum'] = df[seasonal_cols].sum(axis=1)
diff = (df['ANNUAL'] - df['seasonal_sum']).describe()
print("ANNUAL vs sum(Jan-Feb + Mar-May + Jun-Sep + Oct-Dec):")
print(diff.to_string())
print(f"\nMean difference: {diff['mean']:.2f}")
print(f"This means ANNUAL ≈ sum_of_seasons + {diff['mean']:.2f}")

print("\n" + "=" * 70)
print("12. MIN/MAX RANGES PER COLUMN")
print("=" * 70)
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"  {col:15s}: [{df[col].min():.2f}, {df[col].max():.2f}]")

print("\n" + "=" * 70)
print("13. CLASS-WISE CORRELATION WITH FLOOD")
print("=" * 70)
for cls in [0, 1]:
    subset = df[df['flood'] == cls].drop(columns=['flood', 'seasonal_sum'])
    print(f"\n--- Correlation matrix for flood={cls} ---")
    print(subset.corr().round(3).to_string())
