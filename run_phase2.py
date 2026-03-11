"""
Phase 2: Advanced Models & Showcase — House Price Prediction
Ames Housing Dataset  |  Run: python run_phase2.py
Saves one plot per step automatically.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
import warnings, os, pickle, json
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
plt.rcParams['figure.figsize'] = (12, 7)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("=" * 58)
print("  PHASE 2: Advanced Models & Showcase")
print("=" * 58)

# ── Load & Clean (same as Phase 1) ────────────────────────
print("\n[Step 1] Load & Clean Dataset...")
df = pd.read_csv('data/train.csv')
df.drop(columns=['Id'], inplace=True, errors='ignore')
TARGET = 'SalePrice'

NONE_COLS = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',
             'GarageType','GarageFinish','GarageQual','GarageCond',
             'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','MasVnrType']
for c in NONE_COLS:
    if c in df: df[c] = df[c].fillna('None')
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'] \
                      .transform(lambda x: x.fillna(x.median()))
for c in ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
          'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']:
    if c in df: df[c] = df[c].fillna(0)
for c in df.select_dtypes('object'):
    df[c] = df[c].fillna(df[c].mode()[0])
df = df.fillna(df.median(numeric_only=True)).fillna(0)
print(f"  Loaded: {df.shape}  |  Nulls: {df.isnull().sum().sum()} ✓")

# ── Step 2: Feature Engineering ───────────────────────────
print("\n[Step 2] Feature Engineering...")
df['HouseAge']        = df['YrSold']  - df['YearBuilt']
df['YearsSinceRemod'] = df['YrSold']  - df['YearRemodAdd']
df['TotalSF']         = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath']       = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
df['TotalPorch']      = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
df['HasPool']         = (df['PoolArea']    > 0).astype(int)
df['HasGarage']       = (df['GarageArea']  > 0).astype(int)
df['HasBsmt']         = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace']    = (df['Fireplaces']  > 0).astype(int)
df['IsNew']           = (df['YearBuilt']   >= 2000).astype(int)
df['LivAreaPerRoom']  = df['GrLivArea'] / (df['TotRmsAbvGrd'] + 1)

ENG = ['HouseAge','YearsSinceRemod','TotalSF','TotalBath','TotalPorch',
       'HasPool','HasGarage','HasBsmt','HasFireplace','IsNew','LivAreaPerRoom']
print(f"  Created {len(ENG)} new features")

# PLOT 2-1: Engineered Feature Correlations
eng_corr = df[ENG + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
colors   = ['#27ae60' if v > 0 else '#e74c3c' for v in eng_corr]
fig, ax  = plt.subplots(figsize=(10, 6))
eng_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='white', alpha=0.88)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Step 2: Engineered Feature Correlations with SalePrice',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson r')
plt.tight_layout(); plt.savefig('step2_feature_engineering.png', dpi=150); plt.close()
print("  step2_feature_engineering.png saved")

# PLOT 2-2: Distribution of 4 key engineered features
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
plot_feats = [('TotalSF','Total Square Footage'), ('HouseAge','House Age (years)'),
              ('TotalBath','Total Bathrooms'), ('TotalPorch','Total Porch (sqft)')]
for i, (col, label) in enumerate(plot_feats):
    axes[i].hist(df[col], bins=40, color=sns.color_palette('husl',4)[i],
                 edgecolor='white', alpha=0.85)
    axes[i].set_title(f'{label}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel(label); axes[i].set_ylabel('Count')
plt.suptitle('Step 2: Engineered Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('step2_engineered_distributions.png', dpi=150); plt.close()
print("  step2_engineered_distributions.png saved")

# ── Encode + Split ─────────────────────────────────────────
cat_cols = df.select_dtypes('object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop(columns=[TARGET])
y = np.log1p(df[TARGET])   # log-transform target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Features: {X_train.shape[1]}  Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ── Step 3: Train Advanced Models ─────────────────────────
from sklearn.linear_model  import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score

print("\n[Step 3] Training Advanced Models...")

def eval_model(name, model):
    model.fit(X_train_sc, y_train)
    pred_real = np.expm1(model.predict(X_test_sc))
    true_real = np.expm1(y_test)
    mae  = mean_absolute_error(true_real, pred_real)
    rmse = np.sqrt(mean_squared_error(true_real, pred_real))
    r2   = r2_score(true_real, pred_real)
    print(f"  {name:<28} MAE ${mae:>9,.0f}  RMSE ${rmse:>9,.0f}  R2 {r2:.4f}")
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, '_m': model}

print(f"  {'Model':<28} {'MAE':>14}  {'RMSE':>14}  {'R2':>6}")
print("  " + "─" * 68)
res = []
res.append(eval_model("Linear Regression",  LinearRegression()))
res.append(eval_model("Ridge Regression",   Ridge(alpha=10)))
res.append(eval_model("Lasso Regression",   Lasso(alpha=0.0005)))
res.append(eval_model("ElasticNet",         ElasticNet(alpha=0.0005, l1_ratio=0.5)))
res.append(eval_model("Random Forest",      RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)))
res.append(eval_model("Gradient Boosting",  GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)))

res_df = pd.DataFrame([{k: v for k, v in r.items() if k != '_m'} for r in res])
res_df = res_df.sort_values('R2', ascending=False).reset_index(drop=True)

# ── Step 4: Cross-Validation ───────────────────────────────
print("\n[Step 4] Cross-Validation (5-Fold) on Gradient Boosting...")
from sklearn.model_selection import cross_val_score
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                               max_depth=4, subsample=0.8, random_state=42)
cv_r2   = cross_val_score(gb, X_train_sc, y_train, cv=5, scoring='r2', n_jobs=-1)
cv_rmse = cross_val_score(gb, X_train_sc, y_train, cv=5,
                          scoring='neg_root_mean_squared_error', n_jobs=-1)
print(f"  CV R2  : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  CV RMSE: {-cv_rmse.mean():.5f} ± {cv_rmse.std():.5f}  (log scale)")

# PLOT 4: Cross-Validation Scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
folds = [f'Fold {i+1}' for i in range(5)]
axes[0].bar(folds, cv_r2, color=sns.color_palette('husl', 5), edgecolor='white', alpha=0.88)
axes[0].axhline(cv_r2.mean(), color='red', ls='--', lw=1.5, label=f'Mean: {cv_r2.mean():.4f}')
axes[0].set_title('Step 4: 5-Fold CV — R² per Fold', fontweight='bold', fontsize=12)
axes[0].set_ylabel('R²'); axes[0].legend(); axes[0].set_ylim(0.8, 1.0)

axes[1].bar(folds, -cv_rmse, color=sns.color_palette('Set2', 5), edgecolor='white', alpha=0.88)
axes[1].axhline(-cv_rmse.mean(), color='red', ls='--', lw=1.5, label=f'Mean: {-cv_rmse.mean():.5f}')
axes[1].set_title('Step 4: 5-Fold CV — RMSE per Fold', fontweight='bold', fontsize=12)
axes[1].set_ylabel('RMSE (log scale)'); axes[1].legend()
plt.suptitle('Gradient Boosting — Cross-Validation (5 Folds)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('step4_cross_validation.png', dpi=150); plt.close()
print("  step4_cross_validation.png saved")

# ── Step 5: GridSearchCV ───────────────────────────────────
print("\n[Step 5] GridSearchCV: Ridge + Random Forest...")
from sklearn.model_selection import GridSearchCV

ridge_gs = GridSearchCV(Ridge(), {'alpha':[0.1,1,5,10,50,100,300]}, cv=5, scoring='r2', n_jobs=-1)
ridge_gs.fit(X_train_sc, y_train)
best_ridge = ridge_gs.best_estimator_
print(f"  Ridge best alpha={ridge_gs.best_params_['alpha']}  CV R2={ridge_gs.best_score_:.4f}")

# PLOT 5-1: Ridge alpha search
alphas = [0.1,1,5,10,50,100,300]
cv_scores = ridge_gs.cv_results_['mean_test_score']
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(alphas, cv_scores, 'o-', color='steelblue', lw=2, ms=8)
ax.axvline(ridge_gs.best_params_['alpha'], color='red', ls='--', lw=1.5,
           label=f"Best alpha={ridge_gs.best_params_['alpha']}")
ax.set_title('Step 5: GridSearchCV — Ridge Alpha Tuning', fontsize=13, fontweight='bold')
ax.set_xlabel('Alpha (log scale)'); ax.set_ylabel('CV R²'); ax.legend()
plt.tight_layout(); plt.savefig('step5_ridge_gridsearch.png', dpi=150); plt.close()
print("  step5_ridge_gridsearch.png saved")

print("  Running RF GridSearch (~1-2 min)...")
rf_gs = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {'n_estimators':[200,400], 'max_depth':[None,20], 'min_samples_split':[2,5]},
    cv=3, scoring='r2', n_jobs=-1, verbose=0
)
rf_gs.fit(X_train_sc, y_train)
best_rf = rf_gs.best_estimator_
print(f"  RF best: {rf_gs.best_params_}  CV R2={rf_gs.best_score_:.4f}")

# ── Step 6: Model Comparison Dashboard ────────────────────
print("\n[Step 6] Model Comparison Dashboard...")
print("=" * 62)
print("       HOUSE PRICE PREDICTION — MODEL LEADERBOARD")
print("=" * 62)
print(res_df[['Model','MAE','RMSE','R2']].to_string(index=False))
print("=" * 62)

# PLOT 6: 3-panel comparison
colors = sns.color_palette('husl', len(res_df))
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for ax, metric, title in zip(axes,
        ['R2','MAE','RMSE'],
        ['R² Score (higher = better)','MAE — lower is better','RMSE — lower is better']):
    bars = ax.barh(res_df['Model'], res_df[metric], color=colors, edgecolor='white', alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.bar_label(bars, fmt='%.4f' if metric=='R2' else '%.0f', padding=4, fontsize=8)
    ax.set_xlabel(metric)
plt.suptitle('Step 6: Model Comparison Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('step6_model_comparison.png', dpi=150); plt.close()
print("  step6_model_comparison.png saved")

# ── Step 7: Feature Importance ────────────────────────────
print("\n[Step 7] Feature Importance (Random Forest)...")
importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
top20 = importances.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 9))
bar_colors = sns.color_palette('RdYlGn', 20)[::-1]
top20.sort_values().plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white', alpha=0.9)
ax.set_title('Step 7: Top 20 Feature Importances (Random Forest)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout(); plt.savefig('step7_feature_importance.png', dpi=150); plt.close()
print("  step7_feature_importance.png saved")
print("  Top 5 features:")
for feat, score in top20.head(5).items():
    print(f"    {feat:<30} {score:.4f}")

# ── Step 8: Predicted vs Actual ───────────────────────────
print("\n[Step 8] Predicted vs Actual — Gradient Boosting...")
gb_final = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, subsample=0.8, random_state=42)
gb_final.fit(X_train_sc, y_train)
y_pred_real = np.expm1(gb_final.predict(X_test_sc))
y_true_real = np.expm1(y_test)
residuals   = y_true_real - y_pred_real

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
lim = [y_true_real.min()*0.9, y_true_real.max()*1.04]
axes[0].scatter(y_true_real, y_pred_real, alpha=0.4, color='steelblue', s=15)
axes[0].plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel('Actual SalePrice ($)'); axes[0].set_ylabel('Predicted ($)')
axes[0].set_title('Step 8: Gradient Boosting — Predicted vs Actual',
                  fontsize=12, fontweight='bold')
axes[0].legend()
axes[1].scatter(y_pred_real, residuals, alpha=0.4, color='coral', s=15)
axes[1].axhline(0, color='black', lw=1.5, ls='--')
axes[1].set_xlabel('Predicted ($)'); axes[1].set_ylabel('Residual ($)')
axes[1].set_title('Step 8: Residual Plot — Gradient Boosting',
                  fontsize=12, fontweight='bold')
plt.tight_layout(); plt.savefig('step8_predictions.png', dpi=150); plt.close()
print("  step8_predictions.png saved")

mae_gb  = mean_absolute_error(y_true_real, y_pred_real)
rmse_gb = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
r2_gb   = r2_score(y_true_real, y_pred_real)
print(f"  GBM Final  MAE=${mae_gb:,.0f}  RMSE=${rmse_gb:,.0f}  R2={r2_gb:.4f}")

# ── Step 9: Error Analysis ─────────────────────────────────
print("\n[Step 9] Error Distribution Analysis...")
pct_err = np.abs(residuals) / y_true_real * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(residuals, bins=50, color='mediumpurple', edgecolor='white', alpha=0.85)
axes[0].axvline(0, color='red', ls='--', lw=1.5)
axes[0].set_title('Residuals Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Residual ($)'); axes[0].set_ylabel('Count')

axes[1].hist(pct_err, bins=50, color='gold', edgecolor='white', alpha=0.85)
axes[1].axvline(pct_err.median(), color='red', ls='--', lw=1.5,
                label=f'Median: {pct_err.median():.1f}%')
axes[1].set_title('Absolute % Error Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('% Error'); axes[1].legend()

# Accuracy bands pie chart
within_5  = (pct_err < 5).mean() * 100
within_10 = ((pct_err >= 5) & (pct_err < 10)).mean() * 100
within_20 = ((pct_err >= 10) & (pct_err < 20)).mean() * 100
over_20   = (pct_err >= 20).mean() * 100
slices = [within_5, within_10, within_20, over_20]
labels = ['<5% error','5-10% error','10-20% error','>20% error']
pie_colors = ['#27ae60','#f1c40f','#e67e22','#e74c3c']
axes[2].pie(slices, labels=labels, colors=pie_colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10})
axes[2].set_title('Prediction Accuracy Bands', fontsize=11, fontweight='bold')

plt.suptitle('Step 9: Error Analysis — Gradient Boosting', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('step9_error_analysis.png', dpi=150); plt.close()
print("  step9_error_analysis.png saved")
print(f"  Median % error : {pct_err.median():.2f}%")
print(f"  Within 10%     : {(pct_err < 10).mean()*100:.1f}% of test set")

# ── Save Models ────────────────────────────────────────────
print("\n[Saving Models...]")
with open('models/gradient_boosting_final.pkl','wb') as f: pickle.dump(gb_final, f)
with open('models/random_forest_final.pkl','wb')     as f: pickle.dump(best_rf, f)
with open('models/ridge_final.pkl','wb')             as f: pickle.dump(best_ridge, f)
with open('models/scaler_phase2.pkl','wb')           as f: pickle.dump(scaler, f)
with open('models/feature_names_phase2.json','w')    as f: json.dump(list(X_train.columns), f)

print("\n" + "=" * 58)
print("  PHASE 2 COMPLETE!")
print("─" * 58)
print("  Images saved:")
imgs = ['step2_feature_engineering.png','step2_engineered_distributions.png',
        'step4_cross_validation.png','step5_ridge_gridsearch.png',
        'step6_model_comparison.png','step7_feature_importance.png',
        'step8_predictions.png','step9_error_analysis.png']
for img in imgs: print(f"    {img}")
print("─" * 58)
print(f"  Best model: Gradient Boosting")
print(f"  R2   = {r2_gb:.4f}")
print(f"  MAE  = ${mae_gb:,.0f}")
print(f"  RMSE = ${rmse_gb:,.0f}")
print("─" * 58)
print("  Next → streamlit run app.py")
print("=" * 58)
