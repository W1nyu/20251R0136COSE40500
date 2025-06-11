#!/usr/bin/env python3
"""
06_transform_dist.py
 - 모델용 패널 데이터(panel_model.parquet)에 대해
   1) Yeo–Johnson 변환 적용
   2) 상·하위 1% Winsorize 적용
   3) 분포 히스토그램 및 QQ-플롯 생성
 - 결과: output/panel_model_transformed.parquet 저장
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import yeojohnson
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Distribution transform for model data")
parser.add_argument("--input", default="output/panel_model.parquet", help="모델용 데이터 경로")
parser.add_argument("--output", default="output/panel_model_transformed.parquet", help="변환된 데이터 저장 경로")
parser.add_argument("--output_dir", default="output", help="진단 플롯 저장 디렉토리")
args = parser.parse_args()

IN_PATH = Path(args.input)
OUT_PATH = Path(args.output)
PLOT_DIR = Path(args.output_dir)
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# 데이터 로드
print(f"▶ Loading model data from {IN_PATH}")
df = pd.read_parquet(IN_PATH)
print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

# 변환 대상 피처
features = ["price_per_m2", "ln_price"]

# 변환 및 플롯
lambdas = {}
bounds = {}
for feat in features:
    if feat not in df.columns:
        continue
    print(f"▶ Transforming feature: {feat}")
    series = df[feat].dropna()
    # Yeo-Johnson
    yj_data, lam = yeojohnson(series)
    lambdas[feat] = lam
    col_yj = f"yj_{feat}"
    df[col_yj] = np.nan
    df.loc[series.index, col_yj] = yj_data
    # Winsorize (1% 하/상위 절단)
    lower = np.nanpercentile(df[col_yj], 1)
    upper = np.nanpercentile(df[col_yj], 99)
    bounds[feat] = (lower, upper)
    col_w = f"{col_yj}_w"
    df[col_w] = df[col_yj].clip(lower, upper)
    # 히스토그램 및 QQ-플롯 저장
    for col in [feat, col_yj, col_w]:
        data = df[col].dropna()
        plt.figure(figsize=(6,4))
        plt.hist(data, bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"fig_hist_{col}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6,6))
        sm.qqplot(data, line="45", fit=True)
        plt.title(f"QQ-plot of {col}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"fig_qq_{col}.png", dpi=150)
        plt.close()

# 변환 정보 출력
print("▶ Yeo–Johnson λ values:")
for feat, lam in lambdas.items():
    print(f"   {feat}: λ = {lam:.4f}")
print("▶ Winsorize bounds (1%,99%):")
for feat, (low, high) in bounds.items():
    print(f"   yj_{feat}: low={low:.4f}, high={high:.4f}")

# 저장
df.to_parquet(OUT_PATH, index=False)
print(f"✅ Transformed data saved → {OUT_PATH} ({df.shape[0]} rows × {df.shape[1]} cols)") 