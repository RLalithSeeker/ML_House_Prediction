# ML Technical Terms & Formulas — Complete Reference

---

## 1. What is a "Model"?

A model is a **mathematical function** that takes inputs (house features) and outputs a prediction (price).

```
f(OverallQual, GrLivArea, YearBuilt, ...) = SalePrice
```

The goal of training is to find the **best version of f** — the one that makes the smallest errors.

---

## 2. Evaluation Metrics

### MAE — Mean Absolute Error

Average dollar amount the model is off by.

```
MAE = (1/n) × Σ |actual_i - predicted_i|
```

- Add up all errors (ignoring + or −), divide by number of houses
- **Our result:** $13,501 → on average, off by $13,501
- **Best for:** Easy human interpretation ("we're off by $13k on average")

---

### RMSE — Root Mean Squared Error

Like MAE but punishes **large errors more heavily**.

```
RMSE = √[ (1/n) × Σ (actual_i - predicted_i)² ]
```

- Squaring means a $50k error is 25× worse than a $10k error (not just 5×)
- **Our result:** $22,596
- RMSE is **always ≥ MAE** — the gap between them tells you how many big outlier errors exist
- **Best for:** When large errors are unacceptable

---

### R² — R-Squared (Coefficient of Determination)

Percentage of price variation the model explains.

```
R² = 1 - (Σ(actual - predicted)²) / (Σ(actual - mean)²)
```

- Numerator = how much error our model still makes
- Denominator = how much error a "dumb model" (always predicts the average) makes
- Range: 0.0 to 1.0 (higher = better)

| R² | Meaning |
|---|---|
| 1.0 | Perfect predictions every time |
| 0.9 | Explains 90% of price variation |
| 0.5 | Barely better than just guessing the average |
| 0.0 | No better than the average at all |
| < 0 | Worse than guessing the average — model is broken |

- **Our result:** 0.9363 → explains 93.63% of why prices differ between houses

---

## 3. Linear Regression

Fits a straight line through the data.

```
Price = w₁×(LivingArea) + w₂×(Quality) + w₃×(YearBuilt) + ... + b
```

- `w₁, w₂, ...` = **weights** — how much each feature contributes
- `b` = **bias** — a base price before any features
- Training finds the best weights by minimising **Sum of Squared Errors (SSE)**:

```
SSE = Σ (actual - predicted)²
```

- **Our result:** R² = 0.846
- **Limitation:** Assumes a straight-line relationship. Real house prices curve and interact in complex ways.

---

## 4. Ridge Regression — L2 Regularisation

Adds a penalty for **large weights** to prevent overfitting.

```
Loss = Σ(actual - predicted)² + α × Σ(weight_j²)
```

- First part = normal error (wants to be small)
- Second part = penalty for big weights (keeps model simple)
- `α` (alpha) = penalty strength. Higher = stronger control. We tuned: **best α = 300**

**Why it works:** With 286 features, many are noisy. Ridge forces the model to spread weights evenly rather than relying on 1–2 features.

- **Our result:** R² = 0.890

---

## 5. Lasso Regression — L1 Regularisation

Like Ridge but uses absolute value of weights.

```
Loss = Σ(actual - predicted)² + α × Σ|weight_j|
```

**Key difference from Ridge:** L1 penalty causes some weights to become **exactly 0** — automatically removing useless features.

- **Why use it:** Built-in feature selection. With 286 features, many may be irrelevant.
- **Our result:** R² = 0.886, α = 0.0005

---

## 6. ElasticNet

A mix of Ridge (L2) and Lasso (L1) penalties.

```
Loss = Σ(actual - predicted)² + α × [ρ × Σ|w_j| + (1-ρ) × Σw_j²]
```

- `ρ` = l1_ratio. We used 0.5 = 50% Lasso + 50% Ridge
- **Why use it:** When uncertain which penalty fits better — ElasticNet hedges both
- **Our result:** R² = 0.878

---

## 7. Decision Tree (Building Block)

Asks yes/no questions to split data into groups:

```
Is OverallQual > 7?
  YES → Is TotalSF > 2000?
            YES → Predict $350,000
            NO  → Predict $250,000
  NO  → Is HouseAge < 20?
            YES → Predict $180,000
            NO  → Predict $130,000
```

- Each split is chosen to **maximally reduce error** in that group
- **Problem:** A single deep tree memorises training data → overfitting

---

## 8. Random Forest — Bagging

Builds many independent trees and averages their predictions.

```
Prediction = (1/T) × Σ prediction_from_tree_t
```

- `T` = number of trees (we used 300–400)
- Each tree trains on a **random sample** of data (bootstrap sampling)
- Each split considers only a **random subset of features**

**Why randomness helps:**  
Trees on different random subsets make different errors. Averaging cancels individual mistakes.  
This is called **Bagging (Bootstrap Aggregating)**.

**Feature Importance:**  
Measured by how much each feature reduces error across all splits in all trees.
- `OverallQual` = 0.483 → drives 48.3% of decisions
- `TotalSF` = 0.279 → drives 27.9%

- **Our result:** R² = 0.915

---

## 9. Gradient Boosting

Builds trees **sequentially**. Each new tree learns from the **mistakes** of the previous ones.

```
Final prediction = Tree₁ + η×Tree₂ + η×Tree₃ + ... + η×TreeT
```

- `η` (eta) = **learning rate** = 0.05: how much each new tree contributes
- Each new tree focuses on the **residuals** (errors) of all previous trees

**Step by step:**
1. Tree 1 predicts price → compute errors
2. Tree 2 is trained on those errors → it learns to correct them
3. Tree 3 is trained on remaining errors → refines further
4. Repeat 500 times

**Why it beats Random Forest:**  
Trees are not independent — each one specifically targets what the previous couldn't predict.

- **Our result:** R² = 0.9363 ← **best model**

---

## 10. Overfitting vs Underfitting

```
Train Accuracy | Test Accuracy | Diagnosis
──────────────────────────────────────────
    HIGH       |     LOW       | OVERFITTING — memorised, not learned
    LOW        |     LOW       | UNDERFITTING — model too simple
    HIGH       |     HIGH      | GOOD FIT ✅
```

**Prevention techniques we used:**
- Ridge/Lasso penalties (shrink weights)
- `max_depth=4` in GBM (limits tree complexity)
- `subsample=0.8` (each tree sees only 80% of data)
- Cross-validation (test on unseen data)

---

## 11. K-Fold Cross Validation (K=5)

Tests the model K times on different splits of the data.

```
Fold 1: [TEST | TRAIN | TRAIN | TRAIN | TRAIN]
Fold 2: [TRAIN | TEST | TRAIN | TRAIN | TRAIN]
Fold 3: [TRAIN | TRAIN | TEST | TRAIN | TRAIN]
...
CV Score = average of all K test scores
```

- **Why:** One test split could be lucky or unlucky. 5 splits gives a reliable estimate.
- **Our result:** CV R² = 0.9023 ± 0.0133
  - Mean 0.9023 → consistently good
  - ±0.0133 → very low variance → model is stable

---

## 12. GridSearchCV — Hyperparameter Tuning

**Hyperparameters** = settings chosen before training (tree count, alpha, depth).  
GridSearchCV tests every combination automatically using cross-validation.

```
For Ridge:
  alpha = [0.1, 1, 5, 10, 50, 100, 300]
  → Each tested with 5-fold CV
  → Best: alpha=300 (CV R²=0.8729)
```

**Time cost:** 3 parameters × 4 values × 5 folds = 60 model training runs.

---

## 13. StandardScaler

Normalises all features to the same scale.

```
x_scaled = (x - mean) / standard_deviation
```

**Result:** Every feature ends up with mean=0 and std=1

**Example:**
```
GrLivArea mean=1500, std=500
If house has GrLivArea=2000:
  x_scaled = (2000 - 1500) / 500 = 1.0
  (this house is 1 std above average in living area)
```

**Why needed:** Without scaling, `LotArea` (range 1k–215k) would dominate `FullBath` (range 0–4) just because its numbers are larger.

---

## 14. Log Transformation

Applied to `SalePrice` in Phase 2.

```
y_log = ln(SalePrice + 1)      ← training target
y_real = exp(y_log) - 1        ← reverse after prediction
```

- `+1` so we never take log(0) which is undefined
- Python: `np.log1p(y)` and `np.expm1(prediction)`

**Why:** Raw SalePrice is right-skewed (a few $750k houses pull the average up). Log-transform makes it bell-shaped → models learn better.

---

## 15. Residuals & Error Analysis

```
residual_i = actual_i - predicted_i
pct_error_i = |residual_i| / actual_i × 100
```

- **Residuals should be random** around 0 → good model
- A pattern in residuals → model is systematically missing something
- **Our median % error:** 5.51%
- **77.1% of predictions within 10%** → strong real-world performance

---

## Quick Reference Table

| Term | Formula | Our Result |
|---|---|---|
| MAE | avg of \|actual − predicted\| | $13,501 |
| RMSE | √(avg of squared errors) | $22,596 |
| R² | 1 − (model error / naive error) | 0.9363 |
| Ridge Loss | errors² + α × weights² | Best α=300 |
| Lasso Loss | errors² + α × \|weights\| | α=0.0005 |
| GBM | Σ(η × tree_t outputs) | 500 trees, η=0.05 |
| CV Score | average of 5 test fold scores | 0.9023 ± 0.013 |
| StandardScaler | (x − mean) / std | All 286 features |
| Log Transform | ln(y+1), reversed by exp(ŷ)−1 | Applied to SalePrice |
