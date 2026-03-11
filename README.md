# 🏠 House Price Prediction — ML PBL Project

A complete machine learning pipeline for predicting house prices using the Kaggle House Prices dataset.

---

## 📁 Project Structure

```
ML_PBL/
├── data/                        # Raw and processed data
│   └── train.csv                # Kaggle House Prices dataset
├── phase1_notebook.ipynb        # Phase 1: EDA & Linear Regression
├── phase2_notebook.ipynb        # Phase 2: Advanced Models & Optimization
├── app.py                       # Streamlit prediction app
├── models/                      # Saved model artifacts
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

### Download Dataset
Download `train.csv` from Kaggle House Prices:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Place it in the `data/` folder.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `phase1_notebook.ipynb` | Data cleaning, EDA, Baseline Linear Regression |
| `phase2_notebook.ipynb` | Ridge/Lasso, Random Forest, Gradient Boosting, Hyperparameter Tuning |

---

## 🌐 Streamlit App

```bash
streamlit run app.py
```

Input house features and get an instant price prediction!

---

## 📊 Models Compared

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization + feature selection |
| Random Forest | Ensemble, handles non-linearity |
| Gradient Boosting | Best accuracy |

---

## 📏 Evaluation Metrics
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error  
- **R²** — Coefficient of Determination
