"""
02_feature_engineer.py
  ▸ 단지-월 패널에 파생 변수/롤링 지표 추가
  ▸ 출력: artifacts/panel_feat.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from janitor import clean_names
from tqdm import tqdm
import unicodedata  # Unicode normalization for Hangul NFC

# ────────────────────────────────
# CLI
# ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",  default="output/panel_clean.parquet")
parser.add_argument("--output", default="output/panel_feat.parquet")
parser.add_argument("--min_obs", type=int, default=30,
                    help="built_age 계산용 built_year·contract_date 최소 관측수")
parser.add_argument("--sparse_dummies", action="store_true", help="use sparse format for get_dummies regional encoding")
parser.add_argument("--roll_window_supply", type=int, default=12, help="rolling window size in months for supply shock")
parser.add_argument("--roll_window_demand", type=int, default=3, help="rolling window size in months for competition rate")
parser.add_argument("--drop_threshold", type=float, default=0.05, help="threshold to drop columns with missing rate above this fraction")
parser.add_argument("--min_clip", type=float, default=1.0, help="minimum clip value for price_per_m2 before log")
args = parser.parse_args()

IN  = Path(args.input)
OUT = Path(args.output)
OUT.parent.mkdir(exist_ok=True, parents=True)

# ────────────────────────────────
# 1. 데이터 로드 & 기본 정리
# ────────────────────────────────
df = pd.read_parquet(IN)
# 불필요 ID 컬럼 제거 (모델 학습에 사용하지 않음)
for col in ['no','본번','부번']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
# snake_case 및 컬럼명 strip 후 Hangul NFC 정규화
df = clean_names(df)
df.columns = [unicodedata.normalize('NFC', c.strip()) for c in df.columns]

# 컬럼명 매핑: 공급·수요 및 경쟁률 컬럼명 통일
if "supply" in df.columns:
    df.rename(columns={"supply": "unsold_units"}, inplace=True)
if "일반공급_경쟁률" in df.columns:
    df["comp_rate"] = df["일반공급_경쟁률"]

# ────────────────────────────────
# 날짜 누락 보강 & 파싱
# 1) contract_ym, contract_day 기반으로 contract_date 보강
if "contract_ym" in df.columns and "contract_day" in df.columns:
    ym_str = df["contract_ym"].fillna(0).astype(int).astype(str).str.zfill(6)
    day_str = df["contract_day"].fillna(1).astype(int).astype(str).str.zfill(2)
    df["contract_date"] = df.get("contract_date").fillna(
        pd.to_datetime(ym_str + day_str, format="%Y%m%d", errors="coerce"))
# 2) contract_date 기반 year_month 생성
if "contract_date" in df.columns:
    df["year_month"] = pd.to_datetime(df["contract_date"]).dt.to_period("M").dt.to_timestamp()
# 3) contract_year 추출
df["contract_year"] = df["year_month"].dt.year

# built_year 파생: 사용승인일 컬럼이 있으면 연도만 추출하여 built_year로 설정
if "built_year" not in df.columns and "사용승인일" in df.columns:
    df["built_year"] = pd.to_datetime(df["사용승인일"]).dt.year

# ────────────────────────────────
# 2. 파생 가격 변수
# ────────────────────────────────
df["price_per_m2"] = df["price"] / df["area_m2"]
df["ln_price"]     = np.log(df["price_per_m2"].clip(lower=args.min_clip))

# ────────────────────────────────
# 3. 건축 연차(built_age)
# ────────────────────────────────
if {"built_year","contract_year"}.issubset(df.columns):
    df["built_age"] = df["contract_year"] - df["built_year"]
    # built_year 결측 치환(단지‐평균)
    mean_by_cx = df.groupby("complex_id")["built_year"].transform("mean")
    df.loc[df["built_year"].isna(), "built_year"] = mean_by_cx
    df["built_age"] = df["contract_year"] - df["built_year"]

# ────────────────────────────────
# 4. 지역 더미 (시군구 고정효과용)
# ────────────────────────────────
if args.sparse_dummies:
    dummies = pd.get_dummies(df["시군구명"], prefix="reg", dtype="int8", sparse=True)
else:
    dummies = pd.get_dummies(df["시군구명"], prefix="reg", dtype="int8")
df = pd.concat([df, dummies], axis=1)

# ────────────────────────────────
# 5. 공급·수요 롤링 지표
# ────────────────────────────────
df.sort_values(["시군구명","year_month"], inplace=True)

# 공급 쇼크: 미분양(un‐sold) 12개월 누적
if "unsold_units" in df.columns:
    df[f"unsold_units_{args.roll_window_supply}m"] = (
        df.groupby("시군구명")["unsold_units"]
          .transform(lambda s: s.fillna(0).rolling(window=args.roll_window_supply, min_periods=1).sum())
    )

# 청약 과열: 경쟁률 3개월 MA
if "comp_rate" in df.columns:
    df[f"comp_rate_ma{args.roll_window_demand}"] = (
        df.groupby("시군구명")["comp_rate"]
          .transform(lambda s: s.ffill().rolling(window=args.roll_window_demand, min_periods=1).mean())
    )

# ────────────────────────────────
# 6. 결측·이상치 최종 점검
# ────────────────────────────────
num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

# 결측률 5% 이상 변수 → 제거
missing_rate = df.isna().mean()
print("▶ Missing rate per column (top 10):", missing_rate.sort_values(ascending=False).head(10).to_dict())
_initial_drop = missing_rate[missing_rate > args.drop_threshold].index.tolist()
# key 컬럼 보호: multiindex와 주요 파생 변수를 위해 제외
protected = {"complex_id","year_month","price","area_m2","contract_date","contract_year"}
drop_cols = [c for c in _initial_drop if c not in protected]
if drop_cols:
    print(f"⚠️  Removing high-NA columns (>{args.drop_threshold:%}):", drop_cols)
    df.drop(columns=drop_cols, inplace=True)

# 잔여 결측 → 단지/지역 단위 평균 대체 (drop된 컬럼 제외)
for col in [c for c in num_cols if c in df.columns]:
    if df[col].isna().any():
        df[col] = df.groupby("complex_id")[col].transform(
            lambda s: s.fillna(s.mean()))
        df[col] = df.groupby("시군구명")[col].transform(
            lambda s: s.fillna(s.mean()))

# ────────────────────────────────
# 7. 저장
# ────────────────────────────────
df.to_parquet(OUT, index=False)
print(f"✅  Feature set saved → {OUT}")
