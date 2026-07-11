"""
Synthetic Flood Dataset Generator
===================================
Uses Class-wise Gaussian Copula approach (Priority 1).
Separates by flood class, models each independently,
preserves correlations, distributions, and constraints.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata, norm
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_PATH = os.path.join(BASE_DIR, 'flood_dataset.xlsx')
OUTPUT_PATH = os.path.join(BASE_DIR, 'flood_dataset_expanded.csv')


def load_original():
    """Load the original dataset."""
    df = pd.read_excel(ORIGINAL_PATH)
    print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Class distribution: flood=0: {(df['flood']==0).sum()}, flood=1: {(df['flood']==1).sum()}")
    return df


def fit_marginals(data, col):
    """Fit the best marginal distribution for a column."""
    x = data[col].values.astype(float)

    # Try multiple distributions and pick the best by KS test
    candidates = {
        'norm': stats.norm,
        'lognorm': stats.lognorm,
        'gamma': stats.gamma,
        'beta': stats.beta,
    }

    best_name = None
    best_params = None
    best_ks = np.inf

    for name, dist in candidates.items():
        try:
            params = dist.fit(x)
            ks_stat, _ = stats.kstest(x, name, args=params)
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_name = name
                best_params = params
        except Exception:
            continue

    # Fallback: always try normal
    if best_name is None:
        best_name = 'norm'
        best_params = stats.norm.fit(x)

    return best_name, best_params


def data_to_copula_space(data, feature_cols):
    """Transform data to uniform [0,1] space using empirical CDF (rank-based)."""
    n = len(data)
    U = np.zeros((n, len(feature_cols)))
    for j, col in enumerate(feature_cols):
        x = data[col].values.astype(float)
        # Use rank-based empirical CDF, bounded away from 0 and 1
        ranks = rankdata(x, method='average')
        U[:, j] = ranks / (n + 1)  # Ensures (0, 1) range
    return U


def fit_gaussian_copula(U):
    """Fit a Gaussian copula by computing the correlation of the normal-transformed data."""
    Z = norm.ppf(U)
    # Handle any inf values from edge ranks
    Z = np.clip(Z, -6, 6)
    corr_matrix = np.corrcoef(Z, rowvar=False)
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)
    return corr_matrix


def sample_from_copula(corr_matrix, n_samples):
    """Sample from the Gaussian copula."""
    k = corr_matrix.shape[0]
    Z = np.random.multivariate_normal(np.zeros(k), corr_matrix, size=n_samples)
    U = norm.cdf(Z)
    return U


def uniform_to_marginal(u_samples, original_values):
    """
    Convert uniform samples back to original scale using the empirical
    quantile function (inverse CDF) of the original data.
    Uses linear interpolation between sorted original values.
    """
    sorted_vals = np.sort(original_values)
    n = len(sorted_vals)
    # Create empirical CDF points
    ecdf_x = sorted_vals
    ecdf_y = np.arange(1, n + 1) / (n + 1)

    # Interpolate: given uniform u, find x
    result = np.interp(u_samples, ecdf_y, ecdf_x)

    # Add small jitter to avoid exact copies of original data
    std = np.std(original_values)
    jitter = np.random.normal(0, std * 0.03, size=len(result))
    result = result + jitter

    return result


def enforce_constraints(df_synth):
    """Enforce physically realistic constraints on generated data."""

    # Temperature: integer, realistic range [28, 31] based on original
    df_synth['Temp'] = np.clip(np.round(df_synth['Temp']), 28, 31).astype(int)

    # Humidity: integer, range [70, 79] based on original
    df_synth['Humidity'] = np.clip(np.round(df_synth['Humidity']), 70, 79).astype(int)

    # Cloud Cover: integer, range [30, 44] based on original
    df_synth['Cloud Cover'] = np.clip(np.round(df_synth['Cloud Cover']), 30, 44).astype(int)

    # All rainfall values must be non-negative
    for col in ['ANNUAL', 'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec', 'avgjune', 'sub']:
        df_synth[col] = np.maximum(df_synth[col], 0.0)
        df_synth[col] = np.round(df_synth[col], 1)

    # ANNUAL should equal sum of seasonal columns (original data shows this)
    df_synth['ANNUAL'] = df_synth['Jan-Feb'] + df_synth['Mar-May'] + df_synth['Jun-Sep'] + df_synth['Oct-Dec']
    df_synth['ANNUAL'] = np.round(df_synth['ANNUAL'], 1)

    # flood must be integer 0 or 1
    df_synth['flood'] = df_synth['flood'].astype(int)

    return df_synth


def generate_class_samples(class_data, feature_cols, n_samples, flood_label):
    """Generate synthetic samples for one class using Gaussian Copula."""
    print(f"\n  Generating {n_samples} samples for flood={flood_label}...")
    print(f"  Original class size: {len(class_data)}")

    # Step 1: Transform to copula space
    U = data_to_copula_space(class_data, feature_cols)
    print(f"  Transformed to copula space: {U.shape}")

    # Step 2: Fit Gaussian copula
    corr_matrix = fit_gaussian_copula(U)
    print(f"  Fitted Gaussian copula correlation matrix: {corr_matrix.shape}")

    # Step 3: Sample from copula
    U_new = sample_from_copula(corr_matrix, n_samples)
    print(f"  Sampled from copula: {U_new.shape}")

    # Step 4: Transform back to original marginals
    synth_data = {}
    for j, col in enumerate(feature_cols):
        original_values = class_data[col].values.astype(float)
        synth_data[col] = uniform_to_marginal(U_new[:, j], original_values)

    df_synth = pd.DataFrame(synth_data)
    df_synth['flood'] = flood_label

    return df_synth


def remove_near_duplicates(df_synth, df_orig, feature_cols, threshold=0.01):
    """Remove synthetic rows that are too close to original data."""
    orig_vals = df_orig[feature_cols].values
    synth_vals = df_synth[feature_cols].values

    # Normalize for distance computation
    std = np.std(orig_vals, axis=0) + 1e-8
    orig_norm = orig_vals / std
    synth_norm = synth_vals / std

    keep_mask = np.ones(len(df_synth), dtype=bool)
    for i in range(len(synth_norm)):
        dists = np.sqrt(np.sum((orig_norm - synth_norm[i]) ** 2, axis=1))
        if np.min(dists) < threshold:
            keep_mask[i] = False

    removed = (~keep_mask).sum()
    print(f"  Removed {removed} near-duplicate rows")
    return df_synth[keep_mask].reset_index(drop=True)


def validate_synthetic(df_orig, df_synth, feature_cols):
    """Comprehensive validation of the synthetic dataset."""
    report = []
    report.append("=" * 70)
    report.append("VALIDATION REPORT: Original vs Synthetic Dataset")
    report.append("=" * 70)

    report.append(f"\nOriginal: {len(df_orig)} rows | Synthetic: {len(df_synth)} rows")

    # 1. Class Distribution
    report.append("\n--- CLASS DISTRIBUTION ---")
    orig_dist = df_orig['flood'].value_counts(normalize=True).sort_index()
    synth_dist = df_synth['flood'].value_counts(normalize=True).sort_index()
    report.append(f"  Original:  flood=0: {(df_orig['flood']==0).sum()} ({orig_dist.get(0,0)*100:.1f}%)  flood=1: {(df_orig['flood']==1).sum()} ({orig_dist.get(1,0)*100:.1f}%)")
    report.append(f"  Synthetic: flood=0: {(df_synth['flood']==0).sum()} ({synth_dist.get(0,0)*100:.1f}%)  flood=1: {(df_synth['flood']==1).sum()} ({synth_dist.get(1,0)*100:.1f}%)")

    # 2. Mean comparison
    report.append("\n--- MEAN COMPARISON ---")
    report.append(f"  {'Feature':15s}  {'Original':>12s}  {'Synthetic':>12s}  {'Diff%':>8s}")
    for col in feature_cols:
        orig_mean = df_orig[col].mean()
        synth_mean = df_synth[col].mean()
        diff_pct = abs(orig_mean - synth_mean) / (abs(orig_mean) + 1e-8) * 100
        report.append(f"  {col:15s}  {orig_mean:12.2f}  {synth_mean:12.2f}  {diff_pct:7.1f}%")

    # 3. Median comparison
    report.append("\n--- MEDIAN COMPARISON ---")
    report.append(f"  {'Feature':15s}  {'Original':>12s}  {'Synthetic':>12s}  {'Diff%':>8s}")
    for col in feature_cols:
        orig_med = df_orig[col].median()
        synth_med = df_synth[col].median()
        diff_pct = abs(orig_med - synth_med) / (abs(orig_med) + 1e-8) * 100
        report.append(f"  {col:15s}  {orig_med:12.2f}  {synth_med:12.2f}  {diff_pct:7.1f}%")

    # 4. Std comparison
    report.append("\n--- STANDARD DEVIATION COMPARISON ---")
    report.append(f"  {'Feature':15s}  {'Original':>12s}  {'Synthetic':>12s}  {'Diff%':>8s}")
    for col in feature_cols:
        orig_std = df_orig[col].std()
        synth_std = df_synth[col].std()
        diff_pct = abs(orig_std - synth_std) / (abs(orig_std) + 1e-8) * 100
        report.append(f"  {col:15s}  {orig_std:12.2f}  {synth_std:12.2f}  {diff_pct:7.1f}%")

    # 5. Correlation comparison
    report.append("\n--- CORRELATION WITH TARGET (flood) ---")
    report.append(f"  {'Feature':15s}  {'Orig Corr':>12s}  {'Synth Corr':>12s}  {'Diff':>8s}")
    for col in feature_cols:
        orig_corr = df_orig[col].corr(df_orig['flood'])
        synth_corr = df_synth[col].corr(df_synth['flood'])
        diff = abs(orig_corr - synth_corr)
        report.append(f"  {col:15s}  {orig_corr:12.4f}  {synth_corr:12.4f}  {diff:8.4f}")

    # 6. Min/Max range check
    report.append("\n--- RANGE CHECK (Synthetic within realistic bounds) ---")
    for col in feature_cols:
        s_min = df_synth[col].min()
        s_max = df_synth[col].max()
        o_min = df_orig[col].min()
        o_max = df_orig[col].max()
        status = "OK" if s_min >= o_min * 0.8 and s_max <= o_max * 1.2 else "WARN"
        report.append(f"  {col:15s}  Orig[{o_min:.1f}, {o_max:.1f}]  Synth[{s_min:.1f}, {s_max:.1f}]  [{status}]")

    # 7. Distribution overlap (KS test per feature)
    report.append("\n--- DISTRIBUTION OVERLAP (KS Test) ---")
    report.append(f"  {'Feature':15s}  {'KS Stat':>10s}  {'p-value':>10s}  {'Similar?':>10s}")
    for col in feature_cols:
        ks_stat, p_val = stats.ks_2samp(df_orig[col].values, df_synth[col].values)
        similar = "YES" if p_val > 0.05 else "NO"
        report.append(f"  {col:15s}  {ks_stat:10.4f}  {p_val:10.4f}  {similar:>10s}")

    # 8. Unique rows check
    n_unique = len(df_synth.drop_duplicates())
    report.append(f"\n--- DUPLICATE CHECK ---")
    report.append(f"  Total rows: {len(df_synth)}, Unique rows: {n_unique}, Duplicates: {len(df_synth) - n_unique}")

    report_text = "\n".join(report)
    print(report_text)

    # Save report
    report_path = os.path.join(BASE_DIR, 'synthetic_validation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nValidation report saved to: {report_path}")

    return report_text


def main():
    print("=" * 70)
    print("SYNTHETIC FLOOD DATASET GENERATOR")
    print("Method: Class-wise Gaussian Copula")
    print("=" * 70)

    # Load original
    df = load_original()

    feature_cols = ['Temp', 'Humidity', 'Cloud Cover', 'ANNUAL',
                    'Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec',
                    'avgjune', 'sub']

    # Separate by class
    df_class0 = df[df['flood'] == 0].reset_index(drop=True)
    df_class1 = df[df['flood'] == 1].reset_index(drop=True)

    # Generate synthetic samples
    print("\n" + "-" * 50)
    print("PHASE 1: Generating synthetic data per class")
    print("-" * 50)

    synth_class0 = generate_class_samples(df_class0, feature_cols, n_samples=420, flood_label=0)
    synth_class1 = generate_class_samples(df_class1, feature_cols, n_samples=420, flood_label=1)

    # Enforce constraints
    print("\n" + "-" * 50)
    print("PHASE 2: Enforcing physical constraints")
    print("-" * 50)
    synth_class0 = enforce_constraints(synth_class0)
    synth_class1 = enforce_constraints(synth_class1)

    # Remove near-duplicates
    print("\n" + "-" * 50)
    print("PHASE 3: Removing near-duplicates")
    print("-" * 50)
    synth_class0 = remove_near_duplicates(synth_class0, df_class0, feature_cols)
    synth_class1 = remove_near_duplicates(synth_class1, df_class1, feature_cols)

    # Combine original + synthetic
    df_combined = pd.concat([df, synth_class0, synth_class1], ignore_index=True)

    # Remove exact duplicates
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    print(f"\n  Removed {before - len(df_combined)} exact duplicates from combined dataset")

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure integer columns
    df_combined['Temp'] = df_combined['Temp'].astype(int)
    df_combined['Humidity'] = df_combined['Humidity'].astype(int)
    df_combined['Cloud Cover'] = df_combined['Cloud Cover'].astype(int)
    df_combined['flood'] = df_combined['flood'].astype(int)

    print(f"\n  Final dataset: {len(df_combined)} rows")
    print(f"  Class distribution: flood=0: {(df_combined['flood']==0).sum()}, flood=1: {(df_combined['flood']==1).sum()}")

    # Save
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved to: {OUTPUT_PATH}")

    # Also save as xlsx
    xlsx_path = OUTPUT_PATH.replace('.csv', '.xlsx')
    df_combined.to_excel(xlsx_path, index=False)
    print(f"  Saved to: {xlsx_path}")

    # Validate
    print("\n" + "-" * 50)
    print("PHASE 4: Validation")
    print("-" * 50)
    validate_synthetic(df, df_combined, feature_cols)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
