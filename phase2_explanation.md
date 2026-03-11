# Phase 2 — Step-by-Step Explanation with Why Reasoning

---

## Overview

Phase 2 builds on Phase 1's Linear Regression baseline (R²=0.846) by adding:
- 11 engineered features
- 5 advanced models alongside Linear Regression
- Hyperparameter tuning (GridSearchCV)
- Cross-validation for robust evaluation
- Error analysis and model comparison

**Final result: Gradient Boosting with R²=0.9363**

---

## Step 1 — Load & Clean (Same as Phase 1)

**What:** Reload `data/train.csv` and apply all imputation rules.

**Why repeat it:** Phase 2 uses a fresh copy of the data because we'll engineer new features on it. We can't start from Phase 1's already-encoded data because feature engineering must happen on the original column names.

**Result:** 2,930 rows × 80 columns (Id dropped), 0 NaN values.

---

## Step 2 — Feature Engineering

**What:** Create 11 new columns from existing ones.

**Why do this at all:**  
The raw dataset has accurate numbers but doesn't always capture how humans think about houses. A buyer thinks "total living space" not "1st floor + 2nd floor separately." Feature engineering bridges domain knowledge and raw data.

### Each Feature Explained

**`HouseAge = YearSold − YearBuilt`**  
Why: Houses depreciate over time. A 50-year-old house sells for less than a new one, all else equal. Age is a direct depreciation proxy.

**`YearsSinceRemod = YearSold − YearRemodAdd`**  
Why: A recently renovated old house commands a premium. Two houses built in 1950 — one renovated in 2019, one untouched — sell at very different prices.

**`TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`**  
Why: Total living space is what buyers care about. Having three separate columns fragments a single concept. Combining them gives the model one clean signal.

**`TotalBath = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath`**  
Why: Not all bathrooms are equal. A half bath (no shower) is worth less than a full bath. The 0.5 weighting reflects this. Summing all bathrooms into one number is more predictive than four separate columns.

**`TotalPorch = OpenPorch + EnclosedPorch + 3SsnPorch + ScreenPorch`**  
Why: Outdoor liveable space adds value. Four separate porch columns are fragmented — total porch area is the meaningful number.

**`HasPool = 1 if PoolArea > 0 else 0`**  
Why: The presence of a pool matters far more than its exact size. A 200 sqft pool and a 500 sqft pool both signal "luxury home." Converting to binary captures this better.

**`HasGarage = 1 if GarageArea > 0 else 0`**  
Why: Same logic. Having any garage vs. street parking is the key distinction for many buyers.

**`HasBsmt = 1 if TotalBsmtSF > 0 else 0`**  
Why: Having a basement (extra storage, possible living space) is a categorical upgrade, not just more square footage.

**`HasFireplace = 1 if Fireplaces > 0 else 0`**  
Why: A binary comfort/luxury flag. 1 fireplace vs. 2 makes little price difference; having any vs. none does.

**`IsNew = 1 if YearBuilt >= 2000 else 0`**  
Why: "New construction" is treated as a premium category by real estate markets — modern insulation, appliances, layouts. A binary flag captures this market psychology better than year alone.

**`LivAreaPerRoom = GrLivArea / (TotRmsAbvGrd + 1)`**  
Why: Density matters. 1,500 sqft with 3 rooms is spacious; 1,500 sqft with 9 small cramped rooms is not. This ratio captures livability.

**Chart saved:** `step2_feature_engineering.png` — horizontal bar showing each new feature's Pearson correlation with SalePrice (green = positive, red = negative).

**Chart saved:** `step2_engineered_distributions.png` — histograms of TotalSF, HouseAge, TotalBath, TotalPorch distributions.

**Why these charts:** Confirms the features have meaningful signal before we train on them. If a new feature had near-zero correlation with price, we might reconsider including it.

---

## Step 3 — Train 6 Models

**What:** Train and evaluate 6 different model types on the same data.

**Why train 6 instead of just the best one:**  
We don't know which model will win until we test them all. Different models make different assumptions — if one consistently beats others on THIS data, those assumptions fit the data best. Comparing also gives us confidence in the winner.

**Why log-transform the target in Phase 2:**  
`y = log(SalePrice + 1)` is used instead of raw SalePrice.
- Raw prices range from $12k to $755k — heavily right-skewed
- Log-transformed prices are roughly bell-shaped
- Linear models perform better on normally distributed targets
- After prediction, we reverse: `price = exp(prediction) − 1`

### Model Results

| Model | R² | MAE | Why this rank |
|---|---|---|---|
| Gradient Boosting | 0.9363 | $13,501 | Sequential error correction is most powerful |
| Random Forest | 0.9152 | $15,821 | Good ensemble but trees don't learn from each other |
| Ridge | 0.8906 | $14,624 | Linear but regularised — better than plain linear |
| Lasso | 0.8862 | $14,342 | Kills some features, loses a bit of signal |
| ElasticNet | 0.8788 | $14,359 | Middle ground but not optimal for this data |
| Linear Regression | 0.8574 | $14,562 | Too simple — can't capture complex relationships |

---

## Step 4 — Cross-Validation (5-Fold)

**What:** Test Gradient Boosting on 5 different data splits, average the results.

**Why not just rely on one test score:**  
Our 80/20 split might have accidentally put easy-to-predict houses in the test set. Cross-validation tests on 5 different subsets — if the model scores well on all 5, it's genuinely good, not lucky.

**How it works:**
```
Split data into 5 equal parts:
Fold 1: Test on part 1, train on parts 2-3-4-5
Fold 2: Test on part 2, train on parts 1-3-4-5
Fold 3: Test on part 3, train on parts 1-2-4-5
... etc.
Final CV Score = average of 5 test scores
```

**Our result:**
- CV R² = 0.9023 ± 0.0133
- Low standard deviation (±0.013) → model is consistent, not fragile

**Chart saved:** `step4_cross_validation.png` — two bar charts showing R² and RMSE across all 5 folds, with red dashed line at the mean.

---

## Step 5 — GridSearchCV (Hyperparameter Tuning)

**What:** Automatically find the best settings for Ridge and Random Forest.

**Why hyperparameter tuning matters:**  
Every model has settings that control its behaviour. Choosing wrong settings leads to a worse model. Instead of guessing, GridSearchCV tests every combination.

**Ridge alpha tuning:**
- Tested: `alpha ∈ [0.1, 1, 5, 10, 50, 100, 300]`
- Each tested with 5-fold CV
- Winner: **alpha=300**
- Why high alpha: We have 286 features, many correlated. Strong penalty keeps weights from going extreme.

**Random Forest tuning:**
- Tested: `n_estimators ∈ [200,400]`, `max_depth ∈ [None,20]`, `min_samples_split ∈ [2,5]`
- Winner: `400 trees, max_depth=20, min_samples_split=2`
- More trees → more stable. Limiting depth → prevents memorisation.

**Chart saved:** `step5_ridge_gridsearch.png` — line graph of R² vs alpha on log scale. The curve peaks at alpha=300 (marked in red).

---

## Step 6 — Model Comparison Dashboard

**What:** Side-by-side comparison of all 6 models on 3 metrics.

**Why compare on 3 metrics instead of just R²:**  
- R² tells you overall fit
- MAE tells you average dollar error (what buyers care about)
- RMSE tells you if there are any catastrophically wrong predictions

A model might have good R² but terrible MAE. Looking at all three gives a complete picture.

**Leaderboard:**
```
Model               MAE ($)   RMSE ($)   R²
Gradient Boosting   13,501    22,596     0.9363  ← Winner
Random Forest       15,821    26,067     0.9152
Ridge Regression    14,624    29,610     0.8906
Lasso Regression    14,342    30,209     0.8862
ElasticNet          14,359    31,178     0.8788
Linear Regression   14,562    33,814     0.8574
```

**Chart saved:** `step6_model_comparison.png` — 3 horizontal bar charts side by side.

---

## Step 7 — Feature Importance (Random Forest)

**What:** Find which features the Random Forest relies on most when making predictions.

**Why Random Forest for importance (not GBM):**  
Random Forest's importance scores are well-defined: each feature's contribution is measured by how much it reduces prediction error across all trees and all splits. GBM importance is slightly more complex to interpret.

**How it works:**  
At every split in every tree, the model records how much prediction error was reduced by splitting on that feature. Sum this across all 300+ trees → importance score.

**Top 5 features:**
| Feature | Score | Meaning |
|---|---|---|
| `OverallQual` | 0.483 | Quality of materials drives 48% of model decisions |
| `TotalSF` | 0.279 | Total size drives 28% — our engineered feature! |
| `TotalBath` | 0.021 | Bathroom count — 2% |
| `CentralAir_Y` | 0.014 | Having AC is a premium signal |
| `GarageArea` | 0.012 | Garage size matters |

**Insight:** `TotalSF` (which WE created in feature engineering) is the 2nd most important feature — validating our engineering work.

**Chart saved:** `step7_feature_importance.png` — horizontal bar chart of top 20 features, colour-coded red-yellow-green.

---

## Step 8 — Predicted vs Actual (Final Gradient Boosting Evaluation)

**What:** Test the final Gradient Boosting model on the 586 houses it has never seen.

**Why retrain one more time:**  
Previous steps evaluated models directly after training on training data. Now we train from scratch on the full training set and evaluate cleanly on the test set — the true measure of real-world performance.

**Results:**
- MAE = $13,501 → average prediction is off by $13.5k
- RMSE = $22,596 → with penalty for big errors
- R² = 0.9363 → explains 93.6% of price variation

**Chart saved:** `step8_predictions.png` — two charts side by side:

- **Left (Predicted vs Actual):** Each dot = one house. If the dot sits on the red diagonal line → perfect prediction. Dots close to the line = accurate. Our dots are tightly clustered around it.
- **Right (Residual Plot):** Y-axis = error (actual − predicted). Dots scattered randomly around 0 = good. A pattern would mean we're systematically wrong somewhere.

---

## Step 9 — Error Analysis

**What:** Deep dive into how the errors are distributed, not just their average.

**Why this matters:**  
An average error of $13k could mean:
- Scenario A: Every house predicted within $13k (consistent)
- Scenario B: 90% predicted within $5k but 10% off by $80k (inconsistent)

The distribution tells you which scenario you're in.

**Charts saved:** `step9_error_analysis.png` — three charts:

**Left — Residuals Histogram:**  
Shows the distribution of dollar errors. Ideally bell-shaped centred at 0. Ours is approximately symmetric → no systematic bias.

**Middle — Absolute % Error Histogram:**  
Converts errors to percentages of the actual price. Median = **5.51%** → half of all predictions are within 5.51% of the real price.

**Right — Accuracy Pie Chart:**
```
Within 5% error  → ~48% of test houses
5-10% error      → ~29% of test houses  →  77.1% within 10%
10-20% error     → ~16% of test houses
Over 20% error   → ~7%  of test houses
```

Industry standard for automated valuation models: >80% within 10%. We hit 77.1% — very close, and our dataset is smaller than commercial AVM training sets.

---

## Summary of Phase 2 Decisions

| Decision | Choice Made | Why |
|---|---|---|
| Feature engineering | 11 new columns | Raw features miss domain knowledge |
| Target | log(SalePrice) | Normalise skewed distribution |
| Number of models | 6 | Can't know the winner without comparing |
| Best model | Gradient Boosting | Sequential error correction wins on tabular data |
| Validation | 5-Fold CV | More reliable than single train/test split |
| Tuning | GridSearchCV | Systematic beats guessing |
| Trees in GBM | 500, lr=0.05 | Low learning rate + many trees = precise learning |
| GBM max_depth | 4 | Prevents overfitting individual trees |
| RF importance | Top 20 shown | Validates which features matter most |
| Error analysis | 3-chart breakdown | Average error alone is insufficient |
