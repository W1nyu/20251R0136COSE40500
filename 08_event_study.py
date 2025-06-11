#!/usr/bin/env python3
"""
08_event_study.py
 - Calculate event time tau = transaction_month - expected_movein_month
 - Aggregate mean price per tau and apply LOWESS smoothing
 - Output: output/scale_curve.csv, output/fig_scale_curve.png
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# CLI
parser = argparse.ArgumentParser(description="Event study: price vs tau")
parser.add_argument("--panel_feat", default="output/panel_feat.parquet", help="panel feat data path")
parser.add_argument("--meta_a", default="output/layer_A_complex_meta.pickle", help="A layer meta pickle with expected move-in date")
parser.add_argument("--frac", type=float, default=0.15, help="LOWESS smoothing fraction")
parser.add_argument("--tau_min", type=int, default=-36, help="minimum tau")
parser.add_argument("--tau_max", type=int, default=60, help="maximum tau")
parser.add_argument("--output_csv", default="output/scale_curve.csv", help="output CSV path")
parser.add_argument("--output_fig", default="output/fig_scale_curve.png", help="output figure path")
args = parser.parse_args()

# Load data
df = pd.read_parquet(args.panel_feat)
# Ensure complex_id types match
df['complex_id'] = df['complex_id'].astype(str).str.replace(r"\.0$", "", regex=True)
df['complex_id'] = df['complex_id'].astype(str)
# Load meta and ensure key types
meta = pd.read_pickle(args.meta_a)
meta['complex_id'] = meta['complex_id'].astype(str)

# Identify expected move-in date column
# Assuming meta contains '사용승인일' or similar; find first date-like col
date_cols = [c for c in meta.columns if '승인' in c or '입주' in c]
if not date_cols:
    raise ValueError("No expected move-in date column found in meta A")
exp_col = date_cols[0]

# Parse expected move-in date
meta['expected_date'] = pd.to_datetime(meta[exp_col], errors='coerce')
meta['expected_ym'] = meta['expected_date'].dt.to_period('M').dt.to_timestamp()

# Merge panel feat with expected move-in
df = df.merge(meta[['complex_id','expected_ym']], on='complex_id', how='left')

# Ensure year_month is datetime
df['year_month'] = pd.to_datetime(df['year_month'])
# Compute tau
df['tau'] = (df['year_month'].dt.year*12 + df['year_month'].dt.month) - (df['expected_ym'].dt.year*12 + df['expected_ym'].dt.month)
# Filter tau range
df = df[(df['tau'] >= args.tau_min) & (df['tau'] <= args.tau_max)]

# Aggregate raw mean price per m2 by tau
agg = df.groupby('tau')['price_per_m2'].mean().reset_index(name='raw_mean_price')
# LOWESS smoothing
tau = agg['tau'].values
raw = agg['raw_mean_price'].values
smoothed = lowess(raw, tau, frac=args.frac, return_sorted=True)
# smoothed is array of (tau, price)
agg['smoothed_price'] = np.interp(tau, smoothed[:,0], smoothed[:,1])

# Save CSV
agg.to_csv(args.output_csv, index=False)
print(f"▶ Scale curve saved → {args.output_csv}")

# Plot
plt.figure(figsize=(8,4))
plt.plot(agg['tau'], agg['raw_mean_price'], color='gray', label='Raw mean')
plt.plot(agg['tau'], agg['smoothed_price'], color='red', label='LOWESS smoothed')
plt.xlabel('Event time (tau months)')
plt.ylabel('Mean price per m2')
plt.title('Event study: price vs tau')
plt.legend()
plt.tight_layout()
plt.savefig(args.output_fig, dpi=150)
print(f"▶ Figure saved → {args.output_fig}") 