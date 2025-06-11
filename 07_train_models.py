#!/usr/bin/env python3
"""
07_train_models.py
 - 헤도닉 회귀 모델 학습 및 CV
 - Input: output/panel_model_transformed.parquet
 - Output:
     output/cv_metrics.csv
     output/fig_coef.png
     output/fig_pred_vs_true.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype  # numeric 컬럼 필터링

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Hedonic regression with CV")
parser.add_argument("--input", default="output/panel_model_transformed.parquet", help="모델용 데이터 경로")
parser.add_argument("--output_dir", default="output", help="결과 저장 디렉토리")
parser.add_argument("--target", default="ln_price", help="회귀 타겟 변수 (로그 가격)")
parser.add_argument("--n_folds", type=int, default=5, help="교차검증 폴드 수")
parser.add_argument("--top_coef", type=int, default=20, help="표시할 상위 회귀계수 개수")
args = parser.parse_args()

IN_PATH = Path(args.input)
OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 데이터 로드
print(f"▶ Loading data from {IN_PATH}")
df = pd.read_parquet(IN_PATH)
print(f"   Loaded {df.shape[0]} rows × {df.shape[1]} cols")

# 특성 및 타겟 설정
target = args.target
# numeric 타입 칼럼만 사용
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
features = [c for c in numeric_cols if c != target]
X = df[features]
y = df[target]

# 교차검증 설정
kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)

# 결과 저장용 컨테이너
metrics = []
coef_list = []
y_true_all = []
y_pred_all = []

# CV 수행
for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    print(f"▶ Fold {fold}/{args.n_folds}")
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # 상수항 추가
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    # 모델 학습 (robust SE)
    model = sm.OLS(y_train, X_train_sm).fit(cov_type='HC3')

    # 예측 및 메트릭
    y_pred = model.predict(X_test_sm)
    # R2 on test
    r2 = r2_score(y_test, y_pred)
    # MAPE on 원래 가격(exp of 로그)
    true_price = np.exp(y_test)
    pred_price = np.exp(y_pred)
    mape = mean_absolute_percentage_error(true_price, pred_price)

    metrics.append({"fold": fold, "r2": r2, "mape": mape})
    y_true_all.extend(true_price)
    y_pred_all.extend(pred_price)

    # 회귀계수 저장
    params = model.params.reset_index()
    params.columns = ['feature', 'coef']
    params['fold'] = fold
    coef_list.append(params)

# Metrics 저장
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT_DIR/"cv_metrics.csv", index=False)
print(f"▶ CV metrics saved → {OUT_DIR/'cv_metrics.csv'}")

# Save mean coefficients for downstream prediction
coef_df = pd.concat(coef_list, ignore_index=True)
coef_mean_df = coef_df.groupby('feature')['coef'].mean().reset_index()
coef_mean_df.to_csv(OUT_DIR/"coef_mean.csv", index=False)
print(f"▶ Mean coefficients saved → {OUT_DIR/'coef_mean.csv'}")

# 회귀계수 평균화 및 시각화
coef_mean = coef_df.groupby('feature')['coef'].mean().abs().sort_values(ascending=False).head(args.top_coef)
plt.figure(figsize=(8,6))
coef_mean.sort_values().plot(kind='barh')
plt.title('Top Regression Coefficients (abs mean)')
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_coef.png", dpi=150)
plt.close()
print(f"▶ Coefficient plot saved → {OUT_DIR/'fig_coef.png'}")

# 예측 vs 실제 시각화
plt.figure(figsize=(6,6))
plt.scatter(y_true_all, y_pred_all, alpha=0.3)
lims = [min(min(y_true_all), min(y_pred_all)), max(max(y_true_all), max(y_pred_all))]
plt.plot(lims, lims, 'k--')
plt.xlabel('True Price_per_m2')
plt.ylabel('Predicted Price_per_m2')
plt.title('Predicted vs True')
plt.tight_layout()
plt.savefig(OUT_DIR/"fig_pred_vs_true.png", dpi=150)
plt.close()
print(f"▶ Prediction plot saved → {OUT_DIR/'fig_pred_vs_true.png'}")

print("✅ Hedonic regression CV complete.") 