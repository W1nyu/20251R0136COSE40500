#!/usr/bin/env python3
"""
09_predict_3rd.py
 - Predict hedonic prices for target complexes at launch and future tau
 - Input:
     panel_model_transformed.parquet
     meta A pickle (3rd complexes) with launch_date column
     model_coef.csv (coef_mean.csv)
     scale_curve.csv
 - Output:
     output/pred_3rd.csv
     output/fig_pred_3rd.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# CLI
parser = argparse.ArgumentParser(description="Predict 3rd phase hedonic prices with tau adjustment")
parser.add_argument("--panel", default="output/panel_model_transformed.parquet", help="Transformed panel data path")
parser.add_argument("--meta_a", default="output/layer_A_complex_meta.pickle", help="A layer meta pickle with launch date")
parser.add_argument("--launch_date_col", default="사용승인일", help="column name in meta A for launch date")
parser.add_argument("--model_coef", default="output/coef_mean.csv", help="CSV of mean regression coefficients")
parser.add_argument("--scale_curve", default="output/scale_curve.csv", help="CSV of tau vs smoothed price curve")
parser.add_argument("--tau_list", default="36,60", help="Comma-separated tau values to compute")
parser.add_argument("--output_csv", default="output/pred_3rd.csv", help="Predictions output CSV")
parser.add_argument("--output_fig", default="output/fig_pred_3rd.png", help="Predictions figure path")
args = parser.parse_args()

# Load inputs
panel_trans = pd.read_parquet(args.panel)
# Load base panel with index (complex_id, year_month)
panel_base = pd.read_parquet("output/panel_panel.parquet")
panel_base = panel_base.reset_index()
# Combine base index with transformed features
panel_trans = panel_trans.reset_index(drop=True)
if len(panel_base) != len(panel_trans):
    raise ValueError("Row count mismatch between panel_base and panel_transformed")
panel = pd.concat([panel_base[['complex_id','year_month']], panel_trans.reset_index(drop=True)], axis=1)
# Remove existing complex_name from panel to allow meta merge
if 'complex_name' in panel.columns:
    panel.drop(columns=['complex_name'], inplace=True)
panel['complex_id'] = panel['complex_id'].astype(str).str.replace(r"\.0$", "", regex=True)
panel['complex_id'] = panel['complex_id'].astype(str)
meta = pd.read_pickle(args.meta_a)
meta['complex_id'] = meta['complex_id'].astype(str)
coef_df = pd.read_csv(args.model_coef)
scale_df = pd.read_csv(args.scale_curve)

# Prepare coefficients
coef_series = coef_df.set_index('feature')['coef']
# Prepare scale curve mapping
scale_map = scale_df.set_index('tau')['smoothed_price'].to_dict()
f0 = scale_map.get(0, 1.0)

# Process meta A for launch month
if args.launch_date_col not in meta.columns:
    raise ValueError(f"Launch date column '{args.launch_date_col}' not found in meta A")
meta['launch_date'] = pd.to_datetime(meta[args.launch_date_col], errors='coerce')
meta['launch_ym'] = meta['launch_date'].dt.to_period('M').dt.to_timestamp()

# Merge panel features at launch
df = panel.merge(meta[['complex_id','complex_name','launch_ym']], on='complex_id', how='inner')
df_launch = df[df['year_month'] == df['launch_ym']].copy()
if df_launch.empty:
    raise ValueError("No panel records found for complexes at their launch month")

# Build feature matrix
feature_cols = [c for c in coef_series.index if c != 'const']
X = df_launch[feature_cols]
X_sm = sm.add_constant(X)

# Predict base log price and price
base_ln = X_sm.dot(coef_series)
df_launch['base_price'] = np.exp(base_ln)

# Compute future tau prices
taus = [int(t) for t in args.tau_list.split(',')]
for tau in taus:
    f_tau = scale_map.get(tau, np.nan)
    df_launch[f'price_tau{tau}'] = df_launch['base_price'] * (f_tau / f0)

# Save predictions
out_cols = ['complex_id','complex_name','launch_ym','base_price'] + [f'price_tau{t}' for t in taus]
df_launch[out_cols].to_csv(args.output_csv, index=False)
print(f"▶ Predictions saved → {args.output_csv}")

# Plot mean price by tau
mean_prices = df_launch[['base_price'] + [f'price_tau{t}' for t in taus]].mean()
plt.figure(figsize=(6,4))
mean_prices.plot(kind='bar')
plt.ylabel('Predicted price per m2')
plt.title('Predicted price at launch and future tau')
plt.tight_layout()
plt.savefig(args.output_fig, dpi=150)
print(f"▶ Prediction figure saved → {args.output_fig}") 