

import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves plots as files
import matplotlib.pyplot as plt, seaborn as sns
import warnings, os, pickle, json
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("=" * 55)
print("  PHASE 1: House Price Prediction — Loading Data")
print("=" * 55)

# ── Step 1: Load ───────────────────────────────────────────
raw = pd.read_csv('data/train.csv')
print(f"\nRows : {raw.shape[0]}  |  Cols : {raw.shape[1]}")
print(f"Numeric  : {raw.select_dtypes('number').shape[1]}")
print(f"Categoric: {raw.select_dtypes('object').shape[1]}")
print(f"\nSalePrice: ${raw.SalePrice.min():,.0f} – ${raw.SalePrice.max():,.0f}  |  Mean: ${raw.SalePrice.mean():,.0f}")

# ── Step 2: Preprocess ─────────────────────────────────────
print("\n[Step 2] Preprocessing...")
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

# catch-all
df = df.fillna(df.median(numeric_only=True)).fillna(0)
print(f"  Nulls remaining: {df.isnull().sum().sum()} ✓")

# One-hot encode
cat_cols = df.select_dtypes('object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print(f"  After encoding: {df.shape}")

# Split & scale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

X = df.drop(columns=[TARGET]); y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Step 3: EDA Plots ──────────────────────────────────────
print("\n[Step 3] EDA — generating plots...")

# 3a: SalePrice Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(y, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(y.median(), color='red', ls='--', lw=1.5, label=f'Median: ${y.median():,.0f}')
axes[0].set_title('SalePrice Distribution (Raw)', fontweight='bold')
axes[0].set_xlabel('SalePrice ($)'); axes[0].set_ylabel('Count'); axes[0].legend()
axes[1].hist(np.log1p(y), bins=50, color='coral', edgecolor='white', alpha=0.85)
axes[1].set_title('SalePrice Distribution (Log)', fontweight='bold')
axes[1].set_xlabel('log(SalePrice + 1)')
plt.tight_layout(); plt.savefig('target_distribution.png', dpi=150); plt.close()
print("  target_distribution.png saved")

# 3b: Correlation Heatmap
num_df = pd.read_csv('data/train.csv').select_dtypes('number').drop(columns=['Id'], errors='ignore')
corr   = num_df.corr()
top15  = corr['SalePrice'].abs().sort_values(ascending=False).head(16).index
fig, ax = plt.subplots(figsize=(13, 11))
mask = np.triu(np.ones_like(corr.loc[top15, top15], dtype=bool))
sns.heatmap(corr.loc[top15, top15], annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
ax.set_title('Correlation Heatmap — Top 15 Features', fontweight='bold')
plt.tight_layout(); plt.savefig('correlation_heatmap.png', dpi=150); plt.close()
print("  correlation_heatmap.png saved")

# 3c: Scatter Plots
rdf  = pd.read_csv('data/train.csv')
KEY  = [('GrLivArea','Living Area (sqft)'), ('OverallQual','Overall Quality'),
        ('TotalBsmtSF','Basement Area (sqft)'), ('GarageCars','Garage Cars'),
        ('YearBuilt','Year Built'), ('1stFlrSF','1st Floor (sqft)')]
pal  = sns.color_palette('husl', 6)
fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.flatten()
for i, (col, label) in enumerate(KEY):
    pd_ = rdf[[col, 'SalePrice']].dropna()
    axes[i].scatter(pd_[col], pd_['SalePrice'], alpha=0.35, color=pal[i], s=14)
    z  = np.polyfit(pd_[col], pd_['SalePrice'], 1)
    xl = np.linspace(pd_[col].min(), pd_[col].max(), 200)
    axes[i].plot(xl, np.poly1d(z)(xl), 'r--', lw=1.8)
    r = pd_[[col,'SalePrice']].corr().iloc[0, 1]
    axes[i].set_title(f'{label}  (r={r:.3f})', fontsize=10, fontweight='bold')
    axes[i].set_xlabel(label, fontsize=9); axes[i].set_ylabel('SalePrice ($)', fontsize=9)
plt.suptitle('Key Features vs SalePrice', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('scatter_plots.png', dpi=150); plt.close()
print("  scatter_plots.png saved")

# 3d: Feature Correlation Bar
top_corr = corr['SalePrice'].drop('SalePrice').abs().sort_values(ascending=False).head(15)
colors   = ['#27ae60' if corr['SalePrice'][f] > 0 else '#e74c3c' for f in top_corr.index]
fig, ax  = plt.subplots(figsize=(12, 6))
top_corr.plot(kind='bar', ax=ax, color=colors, edgecolor='white', alpha=0.88)
ax.axhline(0.5, color='navy', ls='--', lw=1.5, label='r = 0.50')
ax.set_title('Top 15 Feature Correlations with SalePrice', fontweight='bold')
ax.set_ylabel('|Pearson r|'); ax.legend()
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig('feature_correlation_bar.png', dpi=150); plt.close()
print("  feature_correlation_bar.png saved")

# 3e: Neighbourhood & Quality
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rdf.boxplot(column='SalePrice', by='OverallQual', ax=axes[0])
axes[0].set_title('SalePrice by Overall Quality', fontweight='bold')
axes[0].set_xlabel('Overall Quality'); axes[0].set_ylabel('SalePrice ($)')
plt.sca(axes[0]); plt.title('')
top10 = rdf.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)
top10.plot(kind='bar', ax=axes[1], color='teal', edgecolor='white', alpha=0.85)
axes[1].set_title('Top 10 Neighbourhoods by Median SalePrice', fontweight='bold')
axes[1].set_ylabel('Median SalePrice ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig('neighbourhood_quality.png', dpi=150); plt.close()
print("  neighbourhood_quality.png saved")

# ── Step 4: Train Linear Regression ───────────────────────
print("\n[Step 4] Training Linear Regression...")
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
print("  Linear Regression trained!")

# ── Step 5: Evaluate ───────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae  = mean_absolute_error(y_test, y_pred_lr)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2   = r2_score(y_test, y_pred_lr)

print(f"\n{'='*50}")
print(f"  RESULTS: Linear Regression (Baseline)")
print(f"{'─'*50}")
print(f"  MAE   : ${mae:>12,.2f}  (avg dollar error)")
print(f"  RMSE  : ${rmse:>12,.2f}  (penalises big errors)")
print(f"  R2    : {r2:>13.4f}  (1.0 = perfect fit)")
print(f"{'='*50}")

# Evaluation Plots
residuals = y_test - y_pred_lr
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
lim = [min(y_test.min(), y_pred_lr.min())*0.95,
       max(y_test.max(), y_pred_lr.max())*1.02]
axes[0].scatter(y_test, y_pred_lr, alpha=0.4, color='steelblue', s=14)
axes[0].plot(lim, lim, 'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlim(lim); axes[0].set_ylim(lim)
axes[0].set_xlabel('Actual ($)'); axes[0].set_ylabel('Predicted ($)')
axes[0].set_title('Predicted vs Actual', fontweight='bold'); axes[0].legend()
axes[1].scatter(y_pred_lr, residuals, alpha=0.4, color='coral', s=14)
axes[1].axhline(0, color='black', lw=1.5, ls='--')
axes[1].set_xlabel('Predicted ($)'); axes[1].set_ylabel('Residual ($)')
axes[1].set_title('Residual Plot', fontweight='bold')
plt.tight_layout(); plt.savefig('phase1_predictions.png', dpi=150); plt.close()
print("  phase1_predictions.png saved")

# ── Save Models ────────────────────────────────────────────
with open('models/scaler.pkl','wb')            as f: pickle.dump(scaler, f)
with open('models/linear_regression.pkl','wb') as f: pickle.dump(lr, f)
with open('models/feature_names.json','w')     as f: json.dump(list(X_train.columns), f)

print(f"\n{'='*50}")
print("  PHASE 1 COMPLETE!")
print("  Models saved to models/")
print("  Plots saved to project folder")
print("  Next -> run_phase2.py")
print(f"{'='*50}")
