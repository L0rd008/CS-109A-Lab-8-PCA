import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns # For potentially nicer plots
import os

# Optional: settings for better display in notebooks/consoles
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

output_plot_dir = r"M:\Documents\Wine\results"
os.makedirs(output_plot_dir, exist_ok=True) # Create the directory if it doesn't exist

# File paths provided by you
fp_white = r"M:\Documents\Wine\winequality-white.csv"
fp_red = r"M:\Documents\Wine\winequality-red.csv"

# Load the datasets
# The UCI datasets are typically semicolon-separated.
# If your CSVs are comma-separated and already have headers from the first line, standard read_csv is fine.
# If they are like the raw UCI files, you might need: sep=';'
# Here we need sep=';' as they are UCI datasets
try:
    white_df = pd.read_csv(fp_white, sep=';')
    red_df = pd.read_csv(fp_red, sep=';')
except FileNotFoundError:
    print(f"Ensure the files are at the specified paths:\n{fp_white}\n{fp_red}")
    # Exit or handle error appropriately
    exit()

# Add a column to distinguish wine type before combining
white_df['is_red'] = 0
red_df['is_red'] = 1

# Combine the datasets
wines_df = pd.concat([red_df, white_df], ignore_index=True)

# Clean up column names (remove quotes if any)
wines_df.columns = wines_df.columns.str.replace('"', '').str.strip()

# Recode 'quality'
# Using a copy as in the lab
wines_df_recode = wines_df.copy()
wines_df_recode['quality_category'] = pd.cut(wines_df_recode['quality'],
                                             bins=[0, 5, 7, 10],
                                             labels=[0, 1, 2], # 0=Bad, 1=Average, 2=Great
                                             right=True, include_lowest=True)

# Drop original quality and any rows with NaN in quality_category if they arise
wines_df_recode = wines_df_recode.drop('quality', axis=1)
wines_df_recode = wines_df_recode.rename(columns={'quality_category': 'quality'}) # Use 'quality' for the target
wines_df_recode.dropna(subset=['quality'], inplace=True)


# Separate features (X) and target (y)
# 'is_red' is now a feature.
X = wines_df_recode.drop('quality', axis=1)
y = wines_df_recode['quality'].astype(int) # Ensure y is integer type for scikit-learn classifiers

# Split data
# Using random_state=8 as in the lab for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA is only applied to continuous input features.
# The target `y` (quality class) is categorical and should not be scaled or transformed.

# For convenience, convert scaled arrays back to DataFrames with column names
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Data preparation complete.")
print(f"X_train_scaled_df shape: {X_train_scaled_df.shape}")
print(f"y_train shape: {y_train.shape}")

print("\n--- Question: .coef_ and .intercept_ ---")
# Fit Logistic Regression (similar to lab's initial LogReg)
# The lab uses C=1000000, solver='lbfgs', multi_class='ovr', max_iter=10000
log_reg_q1 = LogisticRegression(C=1000000, solver='lbfgs', multi_class='ovr', max_iter=10000)
log_reg_q1.fit(X_train_scaled_df, y_train)

print("Coefficients (lr.coef_):")
print(log_reg_q1.coef_)
print(f"Shape of .coef_: {log_reg_q1.coef_.shape}")
print("\nIntercepts (lr.intercept_):")
print(log_reg_q1.intercept_)
print(f"Shape of .intercept_: {log_reg_q1.intercept_.shape}")

print("\nNumber of classes:", len(np.unique(y_train)))
print("Number of features:", X_train_scaled_df.shape[1])

# Classification Report for the Initial Model (on Test Data for Consistency)
print("\nClassification Report for log_reg_q1 (trained for coefficient inspection, evaluated on test set):")
y_pred_q1_test = log_reg_q1.predict(X_test_scaled_df)
print(classification_report(y_test, y_pred_q1_test, zero_division=0))

print("\n--- Question: Cross-Validation Results ---")
# Logistic Regression without CV (already fit as log_reg_q1, or re-fit for clarity)
lr_no_cv = LogisticRegression(C=1000000, solver='lbfgs', multi_class='ovr', max_iter=10000).fit(X_train_scaled_df, y_train)
lr_no_cv_test_accuracy = accuracy_score(y_test, lr_no_cv.predict(X_test_scaled_df))
print(f"Logistic Regression (no CV) Test Accuracy: {lr_no_cv_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression (no CV):")
print(classification_report(y_test, lr_no_cv.predict(X_test_scaled_df), zero_division=0))

# Logistic Regression with CV (Lasso as in lab)
# Note: The lab uses 'liblinear' for L1 penalty. 'lbfgs' supports L2.
lr_cv_lasso = LogisticRegressionCV(solver='liblinear', multi_class='ovr', penalty='l1',
                                   max_iter=10000, cv=5, random_state=8) # cv=5 or 10
lr_cv_lasso.fit(X_train_scaled_df, y_train)
lr_cv_lasso_test_accuracy = accuracy_score(y_test, lr_cv_lasso.predict(X_test_scaled_df))
print(f"Logistic Regression CV (L1 penalty) Test Accuracy: {lr_cv_lasso_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression CV (L1 penalty):")
print(classification_report(y_test, lr_cv_lasso.predict(X_test_scaled_df), zero_division=0))

# For a more direct comparison with lbfgs, use L2 with CV
lr_cv_l2 = LogisticRegressionCV(solver='lbfgs', multi_class='ovr', penalty='l2', # lbfgs typically uses l2
                                max_iter=10000, cv=5, random_state=8)
lr_cv_l2.fit(X_train_scaled_df, y_train)
lr_cv_l2_test_accuracy = accuracy_score(y_test, lr_cv_l2.predict(X_test_scaled_df))
print(f"Logistic Regression CV (L2 penalty) Test Accuracy: {lr_cv_l2_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression CV (L2 penalty):")
print(classification_report(y_test, lr_cv_l2.predict(X_test_scaled_df), zero_division=0))


print("\n--- Question: Meaning of 2D PCA Features ---")
pca_2d_transformer = PCA(n_components=2, random_state=8) # Use random_state for reproducibility of PCA if solver involves randomness
X_train_2D = pca_2d_transformer.fit_transform(X_train_scaled_df)
# X_test_2D = pca_2d_transformer.transform(X_test_scaled_df) # For later model training

print("First 2 PCA Components (Eigenvectors):\n", pca_2d_transformer.components_)
print("Shape of components_:", pca_2d_transformer.components_.shape)

print("\n--- Question: Critique PCA Plot (Quality) ---")
plt.figure(figsize=(10, 7))
pca_df_quality = pd.DataFrame(X_train_2D, columns=['PC1', 'PC2'])
# y_train is a Pandas Series. Its values correspond row-wise to X_train_2D.
pca_df_quality['quality_numeric'] = y_train.values # y_train.values to ensure order matches X_train_2D
label_text_map = {0: "Bad Wines", 1: "Average Wines", 2: "Great Wines"}
pca_df_quality['quality_label'] = pca_df_quality['quality_numeric'].map(label_text_map)

custom_palette = {
    "Bad Wines": "red",
    "Average Wines": "cyan",
    "Great Wines": "blue"
}

sns.scatterplot(data=pca_df_quality, x='PC1', y='PC2', hue='quality_label',
                palette=custom_palette, # Use a color palette that distinguishes the classes well
                # style='quality_label', # Different markers for different quality categories
                hue_order=["Bad Wines", "Average Wines", "Great Wines"], # Consistent legend order
                alpha=0.7)



plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("2D PCA of Wine Data (Colored by Quality) - Seaborn")
plt.savefig(os.path.join(output_plot_dir, "pca_plot_by_quality_seaborn.png"))
print(f"Saved pca_plot_by_quality_seaborn.png to {output_plot_dir}")
plt.show()

print("\n--- Question: PCA Plot by Wine Color ---")
plt.figure(figsize=(8, 6))
colors_type = ['red', 'lightgray'] # Red wines, White wines
label_text_type = {1: "Red Wines", 0: "White Wines"} # Assuming is_red=1 for red, 0 for white

# We need X_train to get the 'is_red' column for coloring.
# X_train_2D are the PCA-transformed coordinates.
# Their indices should align.
for wine_type_val in [0, 1]: # 0 for white, 1 for red
    # The mask should come from X_train which has the 'is_red' column
    # The plotting coordinates come from X_train_2D
    # Ensure mask aligns with X_train_2D (which is a numpy array from X_train_scaled_df)
    # X_train['is_red'] has original indices. We need to match its boolean values to the row order of X_train_2D.
    # X_train_scaled_df (from which X_train_2D was derived) has the same index as X_train.
    # So, X_train_2D[X_train['is_red'].values == wine_type_val] should work if y_train.values was used for quality plot.
    mask = (X_train['is_red'] == wine_type_val) # Using .values to align with numpy array X_train_2D
    plt.scatter(X_train_2D[mask, 0], X_train_2D[mask, 1],
                c=colors_type[wine_type_val], label=label_text_type[wine_type_val], alpha=0.5)

plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("2D PCA of Wine Data (Colored by Wine Type)")
plt.legend()
plt.savefig(os.path.join(output_plot_dir, "pca_plot_by_wine_type.png"))
print(f"Saved pca_plot_by_wine_type.png to {output_plot_dir}")
plt.show()

print("\n--- Question: Logistic Regression on 2D PCA Data ---")
# We already have X_train_2D from a previous step. Need X_test_2D.
# If pca_2d_transformer was fit on X_train_scaled_df:
X_test_2D = pca_2d_transformer.transform(X_test_scaled_df)

# Logistic Regression without CV on PCA data
lr_pca_no_cv = LogisticRegression(C=1000000, solver='lbfgs', multi_class='ovr', max_iter=10000)
lr_pca_no_cv.fit(X_train_2D, y_train)
lr_pca_no_cv_test_accuracy = accuracy_score(y_test, lr_pca_no_cv.predict(X_test_2D))
print(f"Logistic Regression (no CV) on 2D PCA Test Accuracy: {lr_pca_no_cv_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression (no CV) on 2D PCA:")
print(classification_report(y_test, lr_pca_no_cv.predict(X_test_2D), zero_division=0))

# Logistic Regression with CV on PCA data
lr_pca_cv_l2 = LogisticRegressionCV(solver='lbfgs', multi_class='ovr', penalty='l2',
                                    max_iter=10000, cv=5, random_state=8)
lr_pca_cv_l2.fit(X_train_2D, y_train)
lr_pca_cv_l2_test_accuracy = accuracy_score(y_test, lr_pca_cv_l2.predict(X_test_2D))
print(f"Logistic Regression CV (L2 penalty) on 2D PCA Test Accuracy: {lr_pca_cv_l2_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression CV (L2 penalty) on 2D PCA:")
print(classification_report(y_test, lr_pca_cv_l2.predict(X_test_2D), zero_division=0))

# The lab specifically mentions using L1 (Lasso) for CV models elsewhere, let's include that too:
lr_pca_cv_lasso = LogisticRegressionCV(solver='liblinear', multi_class='ovr', penalty='l1',
                                       max_iter=10000, cv=5, random_state=8)
lr_pca_cv_lasso.fit(X_train_2D, y_train)
lr_pca_cv_lasso_test_accuracy = accuracy_score(y_test, lr_pca_cv_lasso.predict(X_test_2D))
print(f"Logistic Regression CV (L1 penalty) on 2D PCA Test Accuracy: {lr_pca_cv_lasso_test_accuracy:.4f}")

# Classification Report
print("\nClassification Report for Logistic Regression CV (L1 penalty) on 2D PCA:")
print(classification_report(y_test, lr_pca_cv_lasso.predict(X_test_2D), zero_division=0))

print("\n--- Question: 10-Component PCA Analysis ---")
# 1. Fit PCA with 10 components
pca_10d_transformer = PCA(n_components=10, random_state=8)
X_train_10D = pca_10d_transformer.fit_transform(X_train_scaled_df)

# 2. Cumulative variance explained
explained_variance_ratio_10d = pca_10d_transformer.explained_variance_ratio_
cumulative_variance_10d = np.cumsum(explained_variance_ratio_10d)

print("Variance explained by each of the 10 PCA components:")
for i, var in enumerate(explained_variance_ratio_10d):
    print(f"  PC{i+1}: {var:.4f}")

print("\nCumulative variance explained by n PCA dimensions (n=1 to 10):")
for i, cum_var in enumerate(cumulative_variance_10d):
    print(f"  Up to PC{i+1}: {cum_var:.4f}")

# 3. Agreement with 2D PCA
# Get variance explained from the earlier 2D PCA
var_explained_2d = pca_2d_transformer.explained_variance_ratio_ # Assuming pca_2d_transformer is still in scope and fitted
print(f"\nVariance explained by first two components from 2D PCA: PC1={var_explained_2d[0]:.4f}, PC2={var_explained_2d[1]:.4f}")
print(f"Variance explained by first two components from 10D PCA: PC1={explained_variance_ratio_10d[0]:.4f}, PC2={explained_variance_ratio_10d[1]:.4f}")

# 4. Plot cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), cumulative_variance_10d, marker='o', linestyle='--')
plt.title('Cumulative Variance Explained by Number of PCA Components')
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(range(1, 11))
plt.grid(True)
plt.savefig(os.path.join(output_plot_dir, "cumulative_variance_explained_10_components.png"))
print(f"Saved cumulative_variance_explained_10_components.png to {output_plot_dir}")
print("Suggested elbow point: typically where explained variance growth flattens (look for knee in the curve)")
plt.show()

