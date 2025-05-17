# Lab PCA Exercise Answers

Based on the Python script execution and results from `machine.py`.

---

**Exercise:** What is stored in `.coef_` and `.intercept_`? Why are there so many of them?

*   **What is stored:**
    *   **`.coef_`**: This attribute stores the coefficients (or weights) learned by the Logistic Regression model for each of the input features. In this case, with 12 features, each set of coefficients will have 12 values.
    *   **`.intercept_`**: This attribute stores the intercept (or bias) term learned by the model.

*   **Why are there so many of them?**
    The script output shows:
    ```
    Shape of .coef_: (3, 12)
    Shape of .intercept_: (3,)
    Number of classes: 3
    Number of features: 12
    ```
    *   There are 3 sets of coefficients and 3 intercepts because we are performing a multi-class classification with 3 target classes ('Bad Wines', 'Average Wines', 'Great Wines').
    *   The `LogisticRegression` model, when `multi_class='ovr'` (One-vs-Rest) is specified (as it is in your `log_reg_q1` and other non-CV logistic regression models), trains a separate binary classifier for each class against all other classes.
        *   Model 1: Class 0 (Bad) vs. (Average + Great)
        *   Model 2: Class 1 (Average) vs. (Bad + Great)
        *   Model 3: Class 2 (Great) vs. (Bad + Average)
    *   Each of these 3 binary classifiers learns its own set of 12 feature coefficients and its own single intercept term. This results in a `(3, 12)` shape for `.coef_` and a `(3,)` shape for `.intercept_`.
    *   The `Classification Report for log_reg_q1` (accuracy 0.71) shows the model struggles significantly with 'Class 2' (Great Wines), likely due to class imbalance (only 40 support in the test set).

---

**Exercise:** Hmm, cross-validation didn't seem to offer improved results. Is this correct? Is it possible for cross-validation to not yield better results than non-cross-validation? If so, how and why?

*   **Is this correct?**
    Yes, based on the script's output, cross-validation did not offer improved results for overall test accuracy:
    ```
    Logistic Regression (no CV) Test Accuracy: 0.7085
    Logistic Regression CV (L1 penalty) Test Accuracy: 0.6985
    Logistic Regression CV (L2 penalty) Test Accuracy: 0.7085
    ```
    *   The model without cross-validation achieved an accuracy of `0.7085`.
    *   The `LogisticRegressionCV` model with an L1 penalty (Lasso) achieved `0.6985` (slightly worse).
    *   The `LogisticRegressionCV` model with an L2 penalty (Ridge) achieved `0.7085` (identical to no CV).
    *   The classification reports for all three models also showed very similar performance, including the inability to predict Class 2 (Great wines).

*   **Is it possible for cross-validation to not yield better results than non-cross-validation? If so, how and why?**
    Yes, it is entirely possible. Here are a few reasons why:
    1.  **Optimal Default Hyperparameters:** The specific hyperparameters of the non-CV model (e.g., the default `C` value in `LogisticRegression` if not specified, or the `C=1000000` used) might happen to be very close to optimal for your particular random train-test split. `LogisticRegressionCV` searches for the best `C` from a predefined range across folds.
    2.  **Nature of the Dataset:** The dataset might be such that the model's performance is not very sensitive to the range of hyperparameters explored by `LogisticRegressionCV`. The underlying data structure and separability might be the limiting factors.
    3.  **No Significant Overfitting:** Cross-validation is particularly effective at finding hyperparameters that help prevent overfitting. If the non-CV model is not significantly overfitting the training data, then the regularization provided by tuning `C` through CV might not lead to a noticeable improvement on the test set.
    4.  **Data Representativeness:** While CV gives a more robust estimate of performance, the single test set split might, by chance, favor the non-CV model's parameters.
    5.  **Specific Task and Metric:** For this specific task and dataset, especially with the severe class imbalance for "Great" wines, tuning the `C` parameter might not be the key to unlocking better performance. Other strategies like addressing class imbalance or feature engineering might be more impactful.

---

**Exercise:** Why didn't we scale the y-values (class labels) or transform them with PCA? Is this a mistake?

*   **Why not scale y-values?**
    *   The y-values (target variable) represent categorical class labels: `0` (Bad), `1` (Average), and `2` (Great).
    *   Scaling (like `StandardScaler`) is typically applied to continuous input features to bring them to a similar range of magnitudes. This is important for algorithms sensitive to feature scales (e.g., SVM, k-NN, or gradient descent-based algorithms like Logistic Regression, as it helps with convergence and prevents features with larger values from dominating).
    *   Scaling categorical labels would convert them into arbitrary continuous numbers, distorting their meaning as distinct categories. It doesn't make sense in the context of classification.

*   **Why not transform y-values with PCA?**
    *   PCA is a dimensionality reduction technique primarily used for input features (X). Its goal is to find a lower-dimensional representation of the features that captures the maximum variance.
    *   Applying PCA to the target variable `y` is generally not done and conceptually incorrect for standard classification tasks:
        *   If `y` is a single column of class labels (as in this case), PCA is not applicable in a meaningful way.
        *   If `y` were multi-label (i.e., multiple target columns), one might consider dimensionality reduction techniques for the *target space*, but standard PCA as used for input features isn't the direct approach.
        *   The objective is to predict the original, interpretable class labels. Transforming `y` would change the prediction target and make interpretation difficult.

*   **Is this a mistake?**
    No, it is not a mistake. It is standard and correct practice to scale/transform input features (X) as needed, but to leave the categorical target variable (y) in its original categorical or numerically encoded categorical form for classification. "PCA is only applied to continuous input features. The target y (quality class) is categorical and should not be scaled or transformed."

---

**Exercise:** Our data only has 2 dimensions/features now. What do these features represent?
```
After applying PCA to reduce the dimensionality to 2, the script output for the components (eigenvectors) is:
First 2 PCA Components (Eigenvectors):
 [[ 0.26075704  0.35709348 -0.10733894 -0.24718049  0.29258856 -0.34442106
  -0.41182497  0.15054275  0.17611156  0.28571578  0.01416467  0.47172437]
 [ 0.26969643  0.03197613  0.21194876  0.39461092  0.23997229  0.17804473
   0.19888381  0.55772126 -0.19457259  0.12040579 -0.4775829   0.07256326]]
Shape of components_: (2, 12)
```
*   The two new dimensions (features) created by PCA are **linear combinations** of the original 12 scaled input features.
*   They do not directly correspond to any single original feature like "fixed acidity" or "alcohol." Instead, they are abstract constructs.
*   **Principal Component 1 (PC1)**: The first new feature (represented by the first row of the `components_` array) is the direction in the 12-dimensional feature space that captures the **maximum possible variance** from the original data. The values `[0.2607..., 0.3570..., ..., 0.4717...]` are the weights (loadings) indicating how much each of the original 12 features contributes to forming PC1.
*   **Principal Component 2 (PC2)**: The second new feature (represented by the second row) is the direction, orthogonal (uncorrelated) to PC1, that captures the **next highest amount of variance**. The values `[0.2696..., 0.0319..., ..., 0.0725...]` are the weights for PC2.
*   Essentially, these two new features represent the two most significant "axes of variation" in your dataset.

---

**Exercise:** Critique the PCA plot. Does it prove that wines are similar? Why/why not?

The PCA plot colored by quality (`pca_plot_by_quality_seaborn.png` saved to `M:\Documents\Wine\results`) is expected to show the three wine quality classes (Bad, Average, Great) in the 2D space defined by PC1 and PC2.

*   **Critique of the PCA Plot (Quality):**
    *   **Overlap:** Typically, this plot shows significant overlap between the different wine quality classes. The clusters for "Bad," "Average," and "Great" wines are not clearly separated in these two principal dimensions. As the [Lab PCA Summary] mentions, "wine classes (bad/average/great) overlap significantly in PCA space, making classification harder."
    *   **Information Loss:** The plot only represents the data in 2 dimensions, which (as seen later) capture about 52.8% of the total variance. Nearly half of the information from the original 12 dimensions has been discarded to create this 2D visualization. Differences crucial for quality separation might exist in the dimensions not shown.
    *   **Focus on Variance, Not Separability:** PCA finds dimensions that maximize variance, not necessarily dimensions that best separate predefined classes.

*   **Does it prove that wines are similar? Why/why not?**
    *   **It does not definitively *prove* that all wines are "hopelessly similar"** in an absolute sense.
    *   **Why it might suggest similarity (in these 2 PCs):** The significant overlap indicates that, based on the two principal components that capture the most overall variance from the 11 physicochemical features and the `is_red` feature, wines of different quality categories are not easily distinguishable. They share similar characteristics along these dominant axes of variation.
    *   **Why it's not definitive proof of overall similarity:**
        1.  **Dimensionality Reduction:** As mentioned, this is a 2D projection. The classes might be more separable in a higher-dimensional space (using more PCs or the original features).
        2.  **Nature of PCA:** PCA prioritizes variance. If the features that distinguish wine quality have less variance than features that distinguish, say, wine color or other general characteristics, quality-related separation might be obscured in the top PCs.
        3.  **Unmeasured Features:** The original 11 features might not capture all aspects relevant to perceived wine quality. Other chemical compounds, terroir, aging, etc., are not included.
        4.  **Subjectivity of "Quality":** Wine quality perception can be subjective. The defined bins (0-5, 6-7, 8-10) are a simplification.

---

**Exercise:** The wine data we've used so far consist entirely of continuous predictors. Would PCA work with categorical data? Why or why not?

*   **Current Data:** Your current `X` data includes 11 original physicochemical features (which are continuous) and an `is_red` feature (which is binary categorical, encoded as 0 or 1). Standard PCA can handle numerically encoded binary features.

*   **Would PCA work with categorical data?**
    *   **Standard PCA is designed for continuous numerical data:** It relies on calculating variance and covariance between features. These concepts are most directly applicable to continuous variables.
    *   **Binary Categorical Data (like `is_red`):** If encoded numerically (e.g., 0 and 1), standard PCA can technically be applied. The "variance" of a binary variable is related to its proportions, and it will contribute to the principal components. However, interpreting its loading might require care.
    *   **Nominal Categorical Data (Multiple Categories, No Order):**
        *   If you have a feature like "Region" with values "A", "B", "C", standard PCA cannot directly process these.
        *   They typically need to be transformed first, most commonly using **one-hot encoding** (creating new binary columns like `is_Region_A`, `is_Region_B`, `is_Region_C`).
        *   PCA can then be run on this expanded set of binary features. However, this can lead to a very high-dimensional and sparse dataset, and the resulting principal components might be harder to interpret.
    *   **Ordinal Categorical Data (Ordered Categories):**
        *   If categories have a meaningful order (e.g., "low," "medium," "high"), they can sometimes be mapped to numerical values (e.g., 1, 2, 3). Standard PCA can then be applied. The assumption is that the numerical spacing between categories is meaningful.
    *   **Specialized Techniques:**
        *   For datasets with predominantly categorical features, or mixed data types, specialized techniques like **Categorical PCA (CATPCA)** or **Multiple Correspondence Analysis (MCA)** are often more appropriate than standard PCA. These methods are designed to handle the nature of categorical variables directly.

*   **Why or why not (for standard PCA with non-binary categorical data)?**
    *   **Why not directly:** Standard PCA's mathematical underpinnings (variance, covariance, linear combinations assuming continuous space) are not well-suited for non-numeric or non-binary categorical data.
    *   **Why it might "work" with encoding (but with caveats):** After encoding (like one-hot), the data is numerical, so the algorithm runs. But the interpretation of components as "linear combinations" of these dummy variables can be less intuitive, and issues like high dimensionality and sparsity can arise.

---

**Exercise:** Clusters overlap despite PCA. What could cause this [two disjoint clusters in the PCA plot when colored by quality]? What does this mean?

This question likely refers to the observation of the PCA plot *before* it's explicitly colored by wine type, where one might see general groupings that are not well-explained by quality.

*   **What could cause this?**
    1.  **Dominant Underlying Factor:** The most significant variations in the dataset (which PCA is designed to capture) might be driven by a characteristic other than wine quality. If there's another feature or an implicit grouping that causes large differences in the input variables, PCA will prioritize representing this variation. (The lab strongly hints this is wine type: red vs. white).
    2.  **Natural Groupings in Data:** The physicochemical properties of the wines might naturally fall into distinct groups based on processing methods, grape varietals (though not explicitly in the dataset), or regional characteristics that aren't directly labeled as "quality."
    3.  **Sensitivity of PCA to Feature Scaling:** While you have scaled the data, if certain original features inherently have very large variances even after scaling (unlikely with `StandardScaler` but possible with other scaling or no scaling), they could disproportionately influence the PCs.

*   **What does this mean?**
    *   It means that the primary axes of variation (PC1 and PC2) found by PCA are capturing differences related to this other underlying factor more strongly than they are capturing differences related to the labeled wine quality.
    *   If you see two somewhat disjoint clusters, it implies that the data can be roughly divided into two main groups based on the linear combinations of features that form PC1 and PC2. The characteristics separating these two groups are responsible for a large part of the data's total variance.
    *   The overlap of *quality* labels *within* these (potentially type-driven) clusters means that both "good" and "bad" (or "average") wines exist within each of these larger, structurally different groups.

---

**Exercise:** Wow. Look at that separation [by wine color]. Too bad we aren't trying to predict if a wine is red or white. Does this graph help you answer our previous question [about the disjoint clusters]? Does it change your thoughts? What new insights do you gain?

The plot `pca_plot_by_wine_type.png` (saved to `M:\Documents\Wine\results`) should show clear separation when colored by the `is_red` feature.

*   **Does this graph help you answer our previous question?**
    *   Yes, absolutely. If the PCA plot colored by wine type shows two clearly distinct clusters corresponding to red and white wines, this provides a strong explanation for the "two disjoint clusters" that might have been ambiguously visible or hypothesized in the plot colored by quality. The main source of variation captured by PC1 and PC2 is very likely the difference between red and white wines.

*   **Does it change your thoughts?**
    *   It clarifies the interpretation of the PCA results. Instead of assuming the initial PCA plot's structure (or lack thereof regarding quality) is solely about quality, it shows that PCA is effectively identifying a more dominant, structurally differentiating characteristic of the wines: their color/type.
    *   It reinforces that PCA captures variance. The chemical differences between red and white wines are substantial, leading to high variance along dimensions that separate them.

*   **What new insights do you gain?**
    1.  **Feature Importance for Type:** The 11 physicochemical features (plus `is_red` itself, though its direct contribution to separation if it was an original feature would be obvious) are very effective at distinguishing red wines from white wines.
    2.  **Dominance of Wine Type:** The chemical differences between red and white wines are a more dominant source of variance in this dataset than the chemical differences that define the quality categories (Bad, Average, Great).
    3.  **Challenge for Quality Prediction:** If trying to predict quality across *both* red and white wines using these features, the strong signal from wine type might "mask" or complicate the identification of more subtle quality-related patterns, especially in lower-dimensional PCA projections.
    4.  **PCA's Utility:** This demonstrates PCA's power in uncovering underlying structure in data. Even without explicitly telling it about wine types, it found dimensions that highlight this fundamental difference.

---

**Exercise:** Use Logistic Regression (with and without cross-validation) on the PCA-transformed data. Do you expect this to outperform our original 75% accuracy? What are your results? Does this seem reasonable?

*   **Expectation:**
    *   The original (non-PCA) Logistic Regression test accuracy from your script was `0.7085`. The lab text mentions a figure of ~75% as a general benchmark they achieved.
    *   Given that the 2D PCA only captures about 52.8% of the variance (PC1: 0.3166, PC2: 0.2114), and the PCA plot colored by quality showed significant class overlap, it is **not expected** that Logistic Regression on this 2D PCA-transformed data will outperform the original accuracy. A significant amount of information has been discarded.

*   **Your Results (from script output):**
    ```
    Logistic Regression (no CV) on 2D PCA Test Accuracy: 0.6069
    Logistic Regression CV (L2 penalty) on 2D PCA Test Accuracy: 0.6077
    Logistic Regression CV (L1 penalty) on 2D PCA Test Accuracy: 0.5815
    ```
    *   The accuracies are indeed much lower (around 0.58 to 0.61) than the `0.7085` achieved with the full feature set.
    *   The classification reports for these PCA-based models show even poorer performance in distinguishing Class 0 (Bad wines) and continued failure for Class 2 (Great wines), with overall F1-scores being notably lower.

*   **Does this seem reasonable?**
    *   Yes, these results are reasonable and expected.
    *   **Information Loss:** By reducing 12 dimensions to 2, a substantial amount of information (nearly 47% of the variance) has been removed. If some of this discarded information was relevant for distinguishing wine quality, the model's performance will naturally degrade.
    *   **PCA's Objective:** PCA aims to maximize variance retention, not necessarily to preserve or enhance class separability for a specific target variable. The dimensions that are best for overall variance might not be the best for classifying wine quality.
    *   **Visual Confirmation:** The PCA plot colored by quality visually confirmed that the classes are not well-separated in these top 2 principal components, foreshadowing poor classification performance.

---

**Exercise:**
    1.  Fit a PCA that finds the first 10 PCA components of our training data
    2.  Use `np.cumsum()` to print out the variance we'd be able to explain by using n PCA dimensions for n=1 through 10
    3.  Does the 10-dimension PCA agree with the 2d PCA on how much variance the first components explain? **Do the 10d and 2d PCAs find the same first two dimensions? Why or why not?**
    4.  Make a plot of number of PCA dimensions against total variance explained. What PCA dimension looks good to you?

*   **1. Fit a PCA that finds the first 10 PCA components:**
    *   This was done in your script using `pca_10d_transformer = PCA(n_components=10, random_state=8)`.

*   **2. Use `np.cumsum()` to print out the variance explained:**
    Your script output:
    ```
    Variance explained by each of the 10 PCA components:
      PC1: 0.3166
      PC2: 0.2114
      PC3: 0.1308
      PC4: 0.0810
      PC5: 0.0603
      PC6: 0.0510
      PC7: 0.0450
      PC8: 0.0413
      PC9: 0.0294
      PC10: 0.0217

    Cumulative variance explained by n PCA dimensions (n=1 to 10):
      Up to PC1: 0.3166
      Up to PC2: 0.5280
      Up to PC3: 0.6588
      Up to PC4: 0.7398
      Up to PC5: 0.8002
      Up to PC6: 0.8512
      Up to PC7: 0.8962
      Up to PC8: 0.9375
      Up to PC9: 0.9669
      Up to PC10: 0.9885
    ```

*   **3. Does the 10-dimension PCA agree with the 2d PCA on how much variance the first components explain? Do the 10d and 2d PCAs find the same first two dimensions? Why or why not?**
    Your script output:
    ```
    Variance explained by first two components from 2D PCA: PC1=0.3166, PC2=0.2114
    Variance explained by first two components from 10D PCA: PC1=0.3166, PC2=0.2114
    ```
    *   **Agreement on variance:** Yes, the 10-dimension PCA and the 2-dimension PCA agree perfectly on the amount of variance explained by the first two components (PC1 explains ~31.66%, PC2 explains ~21.14%).
    *   **Same first two dimensions?** Yes, they find the mathematically identical first two dimensions.
    *   **Why or why not?** PCA is a deterministic algorithm. The first principal component (PC1) is defined as the direction in the feature space that captures the maximum variance. The second principal component (PC2) is the direction orthogonal to PC1 that captures the maximum remaining variance, and so on. The process of finding these components is sequential and unique. Requesting more components (e.g., 10 instead of 2) does not alter how the initial components are identified; it simply continues the process to find additional orthogonal components that capture the remaining variance.

*   **4. Make a plot of number of PCA dimensions against total variance explained. What PCA dimension looks good to you?**
    *   The plot `cumulative_variance_explained_10_components.png` was saved to `M:\Documents\Wine\results`.
    *   **What PCA dimension looks good?** This involves finding the "elbow point" in the cumulative explained variance plot. Looking at the cumulative variances:
        *   1 component: 31.66%
        *   2 components: 52.80% (+21.14%)
        *   3 components: 65.88% (+13.08%)
        *   4 components: 73.98% (+8.10%)
        *   5 components: 80.02% (+6.04%)
        *   6 components: 85.12% (+5.10%)
        *   7 components: 89.62% (+4.50%)
        *   8 components: 93.75% (+4.13%)
        *   9 components: 96.69% (+2.94%)
        *   10 components: 98.85% (+2.16%)
        The rate of increase in explained variance starts to slow more noticeably after about 7 or 8 components.
        *   Using **7 components** captures ~89.6% of the variance.
        *   Using **8 components** captures ~93.75% of the variance.
        A choice between 7 or 8 components would be reasonable if aiming to retain around 90-95% of the variance while achieving significant dimensionality reduction from the original 12 features. The exact choice can depend on the trade-off between model complexity and information retention desired for a specific task. The script suggests looking for a "knee in the curve."

---

**Exercise:** Looking at your graph [of cumulative variance explained], what is the 'elbow' point / how many PCA components do you think we should use? Does this number of components imply that predictive performance will be optimal at this point? Why or why not?

*   **Elbow point / How many PCA components to use?**
    *   As discussed above, by examining the plot `cumulative_variance_explained_10_components.png` (and the numerical cumulative variances), the "elbow" or point of diminishing returns appears to be around **7 to 8 components**. After 8 components (which capture 93.75% of the variance), the incremental gain in variance explained by adding more components drops below 3% per component.

*   **Does this number of components imply that predictive performance will be optimal at this point? Why or why not?**
    *   **Not necessarily.**
    *   **Why/Why not:**
        1.  **Variance vs. Predictive Information:** The elbow point in a PCA explained variance plot indicates an efficient trade-off for retaining *overall variance* with fewer dimensions. However, the components that explain the most variance are not always the ones that contain the most *predictive information* for a specific target variable.
        2.  **Information in Later Components:** Some principal components that explain a smaller fraction of the total variance (i.e., those beyond the elbow) might still capture subtle patterns or relationships that are crucial for discriminating between the classes in your prediction task. Discarding them based purely on the elbow point could harm predictive performance.
        3.  **Noise in Early Components:** Conversely, it's also possible that some of the initial high-variance components capture noise or variations that are irrelevant or even detrimental to the specific classification task (e.g., the strong red/white wine signal when trying to predict quality).
        4.  **Model Specificity:** The optimal number of components for predictive performance can also depend on the specific classification model being used. Some models might be more sensitive to irrelevant features or noise than others.
        *   Ultimately, while the elbow point provides a good heuristic for dimensionality reduction from an information theory (variance) perspective, the optimal number of PCA components for *predictive performance* should ideally be determined through experimentation, such as by training and evaluating the predictive model with different numbers of components (e.g., using cross-validation to select the number of components that maximizes predictive accuracy or another relevant metric).
