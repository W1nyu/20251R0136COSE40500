"""
01_clean_merge.py
 - A,B,C,D,E 레이어를 읽어 전처리·병합
 - 결과: artifacts/panel_clean.parquet
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from rapidfuzz import fuzz, process          # fuzzy 단지명 매칭
from janitor import clean_names              # snake_case 컬럼
from tqdm import tqdm
import logging

# ────────────────────────────────
# CLI
# ────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",  default="output", help="00 단계 출력 폴더")
parser.add_argument("--output_dir", default="output", help="병합 결과 저장 폴더")
parser.add_argument("--crosswalk",  default="data/국토교통부_전국 법정동_20250415.csv",
                    help="법정동↔시군구↔좌표 매핑 테이블 (CSV 또는 Parquet)" )
parser.add_argument("--test", action="store_true", help="테스트 모드: 데이터 일부 샘플만 처리")

# 상세 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

args = parser.parse_args()

IN  = Path(args.input_dir)
OUT = Path(args.output_dir)
OUT.mkdir(exist_ok=True, parents=True)

# ────────────────────────────────
# 날짜 전처리 헬퍼 정의 (레이어 로드 이전에 삽입)
# ────────────────────────────────
def _prep_date(df, col):
    df[col] = pd.to_datetime(df[col])
    df["year_month"] = df[col].dt.to_period("M").dt.to_timestamp()
    return df

# ────────────────────────────────
# 1. 레이어 로드
# ────────────────────────────────
print("▶ loading layers …")
# A 레이어 로드 로깅
A = pd.read_pickle(IN / "layer_A_complex_meta.pickle")    # 단지 메타
logging.info("Loaded A layer: %d rows × %d cols", A.shape[0], A.shape[1])
B = pd.read_pickle(IN / "layer_B_transactions.pickle")     # 실거래
logging.info("Loaded B layer: %d rows × %d cols", B.shape[0], B.shape[1])
print("▶ Pre-merge B columns:", B.columns.tolist())
# 테스트 모드: raw A/B 레이어 샘플링 (중요: C/D/E 샘플링은 각 로드 직후 수행)
if args.test:
    print("▶ Test mode: 샘플링 raw A/B 레이어")
    A = A.head(100)
    B = B.head(1000)
# 단지명 컬럼 리네이밍 (complex_name 통일)
for src in ["단지명","complex","단지"]:
    if src in B.columns and "complex_name" not in B.columns:
        B.rename(columns={src: "complex_name"}, inplace=True)
        print(f"▶ Renamed column {src} to complex_name")
        break
# B 레이어 날짜/칼럼 전처리 (raw 거래 CSV 포함)
if "계약년월" in B.columns and "계약일" in B.columns:
    # raw transaction CSV: 칼럼 rename 및 datetime/price parsing
    B = B.rename(columns={
        '단지명': 'complex_name',
        '계약년월': 'contract_ym',
        '계약일': 'contract_day',
        '거래금액(만원)': 'price',
        '전용면적(㎡)': 'area_m2',
        '시군구': 'region_full'
    })
    # 계약일 생성
    # 계약년월, 계약일 float에 .0 제거 후 정수->문자열로 변환하여 정확한 날짜 파싱
    ym = B['contract_ym'].fillna(0).astype(int).astype(str)
    day = B['contract_day'].fillna(0).astype(int).astype(str).str.zfill(2)
    B['contract_date'] = pd.to_datetime(
        ym + day,
        format='%Y%m%d', errors='coerce'
    )
    # 문자열→숫자(만원 단위), NaN은 0 처리 후 원 단위 환산
    B['price'] = pd.to_numeric(B['price'].str.replace(',', '', regex=False), errors='coerce').fillna(0).astype(int) * 10000
    # area_m2 숫자
    B['area_m2'] = B['area_m2'].astype(float)
    # period 처리
    B = _prep_date(B, 'contract_date')
    # complex_id 초기화
    B['complex_id'] = np.nan
else:
    if 'contract_date' in B.columns:
        B = _prep_date(B, 'contract_date')
    else:
        print("▶ Warning: B 레이어에 'contract_date' 컬럼이 없어 날짜 전처리를 건너뜁니다.")
        B['year_month'] = pd.NaT
# 지수·거시 (C 레이어): wide→long & year_month 생성
C_wide = pd.read_pickle(IN / "layer_C_macro_index.pickle")
logging.info("Loaded C_wide: %d rows × %d cols", C_wide.shape[0], C_wide.shape[1])
# 테스트 모드: raw C_wide 레이어 샘플링
if args.test:
    print("▶ Test mode: 샘플링 raw C_wide 레이어")
    C_wide = C_wide.head(100)
date_pattern = re.compile(r"^\d{4}(?:\.\d+)?(?:/2)?(?:년)?$")
date_cols = [c for c in C_wide.columns if date_pattern.match(c)]
id_vars = [c for c in C_wide.columns if c not in date_cols]
C = C_wide.melt(id_vars=id_vars, value_vars=date_cols,
                var_name="year_month_raw", value_name="macro_index")
def parse_ym(s):
    s2 = s.replace("/2", "").replace("년", "")
    return pd.to_datetime(s2, format="%Y.%m", errors="coerce")
C["year_month"] = C["year_month_raw"].apply(parse_ym)
C = C.drop(columns="year_month_raw")
# C 레이어: 병합 키용 시군구명 컬럼 생성
if "행정구역별" in C.columns:
    C["시군구명"] = C["행정구역별"]
logging.info("Transformed C to long: %d rows × %d cols", C.shape[0], C.shape[1])
D = pd.read_pickle(IN / "layer_D_supply.pickle")     # 공급
logging.info("Loaded D layer: %d rows × %d cols", D.shape[0], D.shape[1])
# 테스트 모드: raw D 레이어 샘플링
if args.test:
    print("▶ Test mode: 샘플링 raw D 레이어")
    D = D.head(100)
# D 레이어: 불필요 메타컬럼 제거 후 wide->long 변환 및 year_month 생성
# id_vars는 '시군구'만 사용
date_pattern_D = re.compile(r"^\d{4}년\s*\d{1,2}월$")
date_cols_D = [c for c in D.columns if date_pattern_D.match(c)]
# 필요한 컬럼만 유지
D = D[['시군구'] + date_cols_D]
# date_cols를 datetime으로 일괄 변환 매핑 생성
mapping_D = {col: pd.to_datetime(col.replace('년','').replace('월',''), format='%Y %m', errors='coerce')
             for col in date_cols_D}
logging.info("D unpivot 준비: date columns=%d, keep id_vars=['시군구']", len(date_cols_D))
# unpivot 실행
D = D.melt(id_vars=['시군구'], value_vars=date_cols_D,
           var_name='year_month_raw', value_name='supply')
# 매핑을 통해 vectorized datetime 생성
D['year_month'] = D['year_month_raw'].map(mapping_D)
D.drop(columns=['year_month_raw'], inplace=True)
# id_vars '시군구'를 '시군구명'으로 복사
D = D.rename(columns={'시군구':'시군구명'})
logging.info("D 최적화 unpivot 완료: %d rows × %d cols", D.shape[0], D.shape[1])
E = pd.read_pickle(IN / "layer_E_competition.pickle")# 청약 경쟁률
logging.info("Loaded E layer: %d rows × %d cols", E.shape[0], E.shape[1])
# 테스트 모드: raw E 레이어 샘플링
if args.test:
    print("▶ Test mode: 샘플링 raw E 레이어")
    E = E.head(100)
# E 레이어: 병합 키용 시군구명 및 연월->year_month 변환
if "지역" in E.columns:
    E["시군구명"] = E["지역"]
if "연월" in E.columns:
    E["year_month"] = pd.to_datetime(E["연월"].astype(str), format="%Y.%m", errors="coerce")
cw_path = Path(args.crosswalk)
if cw_path.suffix.lower() == ".csv":
    # CSV crosswalk 로드 및 칼럼 정제
    XW = pd.read_csv(
        cw_path,
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
        on_bad_lines="skip",
        engine="python"
    ).rename(columns=lambda c: c.strip())
    # 필요한 칼럼 매핑
    rename_map = {
        "법정동코드": "법정동코드",
        "법정동명":  "읍면동명",
        "시도코드":  "시도코드",
        "시군구코드": "시군구코드",
        "시도명":   "시도명",
        "시군구명": "시군구명",
        "폐지여부": "폐지여부",
        "생성일자": "유효시작일",
    }
    XW = XW[[k for k in rename_map.keys() if k in XW.columns]].rename(columns=rename_map)
else:
    XW = pd.read_parquet(args.crosswalk)
    # Parquet crosswalk (예: 이미 변환된 파일)

# B 레이어 empty guard
if B.empty:
    print("▶ Warning: B 레이어가 비어 있어 패널 병합을 생략하고 결과를 빈 패널로 저장합니다.")
    import sys
    panel_empty = pd.DataFrame()
    panel_empty.to_parquet(OUT/"panel_clean.parquet", index=False)
    sys.exit(0)

# ────────────────────────────────
# 2. 공통 컬럼 전처리
# ────────────────────────────────
# snake_case & strip
for df in (A, B, C, D, E):
    df = clean_names(df)                # janitor
    df.columns = [c.strip() for c in df.columns]

# ────────────────────────────────
# 3. 공간 매핑: 거래 B → 단지 A
# ────────────────────────────────
try:
    logging.info("Starting spatial join trades → complex")
    # 3-1. 우선 complex_id가 있는 행은 그대로 매핑
    mapped = B[B["complex_id"].notna()].copy()
    # 3-2. complex_id 없는 행 → 정규화 기반 exact 매핑 후 필요시 fuzzy 매칭
    nomap = B[B["complex_id"].isna()].copy()
    if len(nomap):
        # normalize 함수 정의 (소문자, 공백·특수문자 제거 및 '아파트' 제거)
        def normalize(x):
            if pd.notna(x):
                s = x.lower()
                s = re.sub(r"\(.*?\)", "", s)
                s = s.replace("아파트", "")
                s = re.sub(r"[\s\-\.\(\)]", "", s)
                return s
            return ""
        # 1) A 레이어에서 normalized 이름 리스트 및 매핑 생성
        A_norm = A["complex_name"].astype(str).apply(normalize)
        name_dict_norm = dict(zip(A_norm, A["complex_id"]))
        # 2) nomap 정규화 및 exact 매핑
        nomap["complex_norm"] = nomap["complex_name"].astype(str).apply(normalize)
        nomap["complex_id"] = nomap["complex_norm"].map(name_dict_norm)
        # 2-5) manual crosswalk mapping for unmapped entries
        manual_path = IN / "complex_manual_crosswalk.csv"
        if manual_path.exists():
            manual = pd.read_csv(manual_path, dtype=str)
            manual_map = dict(zip(manual["complex_name"], manual["complex_id"]))
            before_manual = nomap["complex_id"].isna().sum()
            nomap["complex_id"] = nomap["complex_id"].fillna(nomap["complex_name"].map(manual_map))
            after_manual = nomap["complex_id"].isna().sum()
            logging.info("Applied manual complex crosswalk: %d -> %d unmapped", before_manual, after_manual)
        # 3) exact 매핑 후 결측인 normalized 이름만 fuzzy 매칭
        unmapped_norm = nomap.loc[nomap["complex_id"].isna(), "complex_norm"].dropna().unique()
        if len(unmapped_norm):
            # block 기반 후보군 축소
            block_map = {}
            for norm_key, cid in name_dict_norm.items():
                blk = norm_key[:2]
                block_map.setdefault(blk, []).append(norm_key)
            matches = {}
            for norm_val in tqdm(unmapped_norm, desc="fuzzy matching by block"):
                blk = norm_val[:2]
                choices = block_map.get(blk, list(name_dict_norm.keys()))
                threshold = 90 if len(norm_val) > 4 else 85
                match = process.extractOne(norm_val, choices, scorer=fuzz.ratio)
                matches[norm_val] = name_dict_norm.get(match[0]) if match and match[1] >= threshold else np.nan
            # 결과 반영
            mask = nomap["complex_id"].isna()
            nomap.loc[mask, "complex_id"] = nomap.loc[mask, "complex_norm"].map(matches)
            # 저장되지 않은 매핑 필요 항목 목록 저장
            unmapped_list = nomap.loc[nomap["complex_id"].isna(), "complex_name"].unique()
            if len(unmapped_list):
                pd.DataFrame(unmapped_list, columns=["complex_name"]).to_csv(OUT/"unmapped_complex_names.csv", index=False)
                logging.info("Saved %d unmapped complex names for manual mapping to %s", len(unmapped_list), OUT/"unmapped_complex_names.csv")
        nomap.drop(columns=["complex_norm"], inplace=True)
        B = pd.concat([mapped, nomap]).reset_index(drop=True)
    # 4. 거래 B ↔ Cross-walk merge 대신 B 원본 시군구(region_full)를 시군구명으로 사용
    # B['region_full'] 컬럼이 '시군구' 정보를 담고 있으므로 이를 시군구명으로 복사
    B["시군구명"] = B.get("region_full", B.get("시군구", None))
    # 시도명 정보가 필요하다면, crosswalk 대신 지정하거나 None으로 채우기
    B["시도명"] = None
    # merge 직전 complex_id 타입 일관화를 위해 문자열 변환
    A["complex_id"] = A["complex_id"].astype(str)
    B["complex_id"] = B["complex_id"].astype(str)
    # 5. 이상치・결측 처리 (순서 생략)
    logging.info("Merging layers: A, C, D, E into panel")
    # A 레이어 merge
    panel = B.merge(A, on="complex_id", how="left", suffixes=("","_cx"))
    logging.info("After merging A: panel %d rows × %d cols", panel.shape[0], panel.shape[1])
    # C 레이어 merge
    panel = panel.merge(C, on=["시군구명","year_month"], how="left")
    logging.info("After merging C: panel %d rows × %d cols", panel.shape[0], panel.shape[1])
    # D 레이어 merge (키 존재 시)
    if "시군구명" in D.columns and "year_month" in D.columns:
        panel = panel.merge(D, on=["시군구명","year_month"], how="left", suffixes=("","_sup"))
        logging.info("After merging D: panel %d rows × %d cols", panel.shape[0], panel.shape[1])
    else:
        print("▶ Warning: D 레이어 merge skipped (keys not present)")
    # E 레이어 merge (키 존재 시)
    if "시군구명" in E.columns and "year_month" in E.columns:
        panel = panel.merge(E, on=["시군구명","year_month"], how="left", suffixes=("","_cmp"))
        logging.info("After merging E: panel %d rows × %d cols", panel.shape[0], panel.shape[1])
    else:
        print("▶ Warning: E 레이어 merge skipped (keys not present)")
except Exception as e:
    print(f"▶ Warning: mapping/merge 단계 생략({e})")
    panel = B

# ────────────────────────────────
# 7. 기본 QC 로그
# ────────────────────────────────
qc = {
    "rows_total": len(panel),
    "missing_complex": int(panel["complex_id"].isna().sum()),
    "missing_price":   int(panel["price"].isna().sum()),
    "merge_null_rate": round(panel.isna().mean().mean(), 4)
}
(OUT / "qc_clean_merge.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False))
print("QC summary:", qc)

# ────────────────────────────────
# 8. 저장
# ────────────────────────────────
panel.to_parquet(OUT / "panel_clean.parquet", index=False)
print(f"✅ saved to {OUT/'panel_clean.parquet'}")
