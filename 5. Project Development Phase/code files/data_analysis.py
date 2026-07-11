import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'flood_dataset_expanded.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

os.makedirs(REPORTS_DIR, exist_ok=True)

print("Loading dataset...")
dataset = pd.read_csv(DATA_PATH)

# --- 1. Descriptive Analysis ---
print("Performing Descriptive Analysis...")
desc_file = os.path.join(REPORTS_DIR, 'descriptive_analysis.txt')
with open(desc_file, 'w') as f:
    f.write("=== Dataset Head (First 5 Rows) ===\n")
    f.write(dataset.head().to_string())
    
    f.write("\n\n=== Dataset Statistical Summary (Describe) ===\n")
    f.write(dataset.describe().to_string())
    
    f.write("\n\n=== Dataset Info (Data Types and Missing Values) ===\n")
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    f.write(buffer.getvalue())

# --- 2. Univariate Analysis ---
print("Performing Univariate Analysis (Distributions & Boxplots)...")
features_to_plot = ['Temp', 'Humidity', 'Cloud Cover', 'ANNUAL', 'Jun-Sep']

# Distribution plots
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(dataset[feature], kde=True, color='#4facfe')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'univariate_distributions.png'))
plt.close()

# Box plots (for outlier detection)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=dataset[feature], color='#00f2fe')
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'univariate_boxplots.png'))
plt.close()

# --- 3. Multivariate Analysis ---
print("Performing Multivariate Analysis (Heatmap)...")
plt.figure(figsize=(14, 12))
corr = dataset.corr()
# Using the specific parameters from your screenshot
sns.heatmap(corr, annot=True, cmap='summer', linewidths=1, linecolor='k', square=True, fmt=".2f")
plt.title("Correlation Heatmap of All Features")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'multivariate_heatmap.png'))
plt.close()

print(f"✅ All analysis complete! Outputs successfully saved to: {REPORTS_DIR}")
