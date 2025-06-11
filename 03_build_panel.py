"""
03_build_panel.py
 - 패널 데이터(단지×월) 재구성 및 시간 고정효과 생성
 - 입력 : output/panel_feat.parquet
 - 출력 : output/panel_panel.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
from janitor import clean_names
import unicodedata

# ────────────────────────────────
# CLI 설정
# ────────────────────────────────
parser = argparse.ArgumentParser(description="Build panel dataset for hedonic modeling")
parser.add_argument(
    "--input",
    default="output/panel_feat.parquet",
    help="02 단계 출력 패널 피처 파일 경로"
)
parser.add_argument(
    "--output",
    default="output/panel_panel.parquet",
    help="생성될 패널 데이터 파일 경로"
)
parser.add_argument(
    "--add_time_dummies",
    action="store_true",
    help="월별 고정효과(dummy) 변수를 추가 생성"
)
parser.add_argument(
    "--sparse_dummies",
    action="store_true",
    help="시간 더미 생성 시 희소(sparse) 형식으로 생성"
)
parser.add_argument(
    "--time_dummy_prefix",
    default="tm",
    help="시간 더미 변수의 접두사 (기본: tm)"
)
args = parser.parse_args()

IN  = Path(args.input)
OUT = Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────
# 1. 데이터 로드 및 칼럼 정리
# ────────────────────────────────
df = pd.read_parquet(IN)
print(f"▶ Loaded panel features: {df.shape[0]} rows × {df.shape[1]} cols")
df = clean_names(df)
df.columns = [unicodedata.normalize("NFC", c.strip()) for c in df.columns]

# ────────────────────────────────
# 2. 시간 식별자 생성
# ────────────────────────────────
df["year_month"] = pd.to_datetime(df["year_month"])
# YYYYMM 형식의 정수로 변환 (예: 2021-07 → 202107)
df["time_id"] = df["year_month"].dt.year * 100 + df["year_month"].dt.month
print(f"▶ Created time_id, unique periods: {df['time_id'].nunique()}")

# ────────────────────────────────
# 3. (선택) 시점 고정효과 더미 생성
# ────────────────────────────────
if args.add_time_dummies:
    print(f"▶ Generating time dummies: prefix={args.time_dummy_prefix}, sparse={args.sparse_dummies}")
    dummies = pd.get_dummies(
        df["time_id"],
        prefix=args.time_dummy_prefix,
        dtype="int8",
        sparse=args.sparse_dummies
    )
    # Parquet 호환성: sparse 데이터는 지원되지 않으므로 dense로 변환
    if args.sparse_dummies:
        dummies = dummies.sparse.to_dense()
    print(f"▶ Added time dummies: {dummies.shape[1]} columns, new shape: {df.shape[1] + dummies.shape[1]} cols")
    df = pd.concat([df, dummies], axis=1)

# ────────────────────────────────
# 4. 인덱스 설정 및 중복 제거
# ────────────────────────────────
print("▶ Setting complex_id and year_month as multiindex and removing duplicates")
initial = len(df)
df.set_index(["complex_id", "year_month"], inplace=True)
# 동일 complex_id+year_month가 중복될 경우 첫 행만 남김
df = df[~df.index.duplicated(keep="first")]
removed = initial - len(df)
print(f"▶ Duplicate removal: {removed} rows removed, final rows: {len(df)}")

# ────────────────────────────────
# 5. 저장
# ────────────────────────────────
df.to_parquet(OUT)
print(f"✅ Panel dataset saved → {OUT} ({len(df)} rows × {df.shape[1]} cols)")
