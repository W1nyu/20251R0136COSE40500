#!/usr/bin/env python3
"""
05_prepare_model.py
 - 03_build_panel 단계 출력 패널 데이터를 백업하고
   모델 학습에 불필요한 컬럼을 제거하여 저장
 - 백업: output/panel_panel_full.parquet
 - 모델용 데이터: output/panel_model.parquet
"""
import argparse
from pathlib import Path
import pandas as pd

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Prepare model dataset from panel data")
parser.add_argument("--input",  default="output/panel_panel.parquet", help="패널 데이터 경로 (03단계 출력)")
parser.add_argument("--backup", default="output/panel_panel_full.parquet", help="백업 파일 경로")
parser.add_argument("--output", default="output/panel_model.parquet", help="모델용 데이터 저장 경로")
args = parser.parse_args()

IN_PATH  = Path(args.input)
BACKUP_PATH = Path(args.backup)
OUT_PATH = Path(args.output)
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# ──────────────────────────────────────────────
# 데이터 로드 및 백업
# ──────────────────────────────────────────────
df = pd.read_parquet(IN_PATH)
df.to_parquet(BACKUP_PATH, index=False)
print(f"▶ Backup saved → {BACKUP_PATH} ({df.shape[0]} rows × {df.shape[1]} cols)")

# ──────────────────────────────────────────────
# 불필요 컬럼 드롭
# ──────────────────────────────────────────────
# ID 컬럼
drop_cols = ["no"]
# 날짜 관련 컬럼
drop_cols += ["contract_day", "contract_ym", "contract_year", "time_id"]
# 시간 더미 컬럼 (tm_*)
drop_cols += [c for c in df.columns if str(c).startswith("tm_")]

df_model = df.drop(columns=drop_cols, errors="ignore")
print(f"▶ Dropped {len(drop_cols)} columns: {drop_cols[:5]}{'...' if len(drop_cols)>5 else ''}")

# ──────────────────────────────────────────────
# 모델 데이터 저장
# ──────────────────────────────────────────────
df_model.to_parquet(OUT_PATH, index=False)
print(f"✅ Model dataset saved → {OUT_PATH} ({df_model.shape[0]} rows × {df_model.shape[1]} cols)") 