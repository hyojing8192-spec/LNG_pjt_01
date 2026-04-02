"""
economics_engine.py  (F3)
=========================
주요 함수:
  - get_season()             : 월 → 계절 문자열
  - get_load_type()          : 날짜·시간 → 부하 유형
  - get_elec_price()         : 날짜·시간 → 수전단가 (원/kWh)
  - calc_replace_cost()      : 대체단가 계산
  - calc_bep()               : BEP 계산
  - calc_economics()         : 경제성 차이 + 경제성(억원)
  - get_best_mode()          : 시간별 최적 운전모드 선정
  - build_hourly_table()     : 24시간 경제성 테이블 생성
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date, datetime
from config import (
    ELEC_RATES, LEGAL_HOLIDAYS, SPRING_FALL_DISCOUNT_HOURS,
    OVERHEAD_COST,
    MODE_LABELS,
)

# 연료량 → MMBtu 환산 (사용자 엑셀 정의와 동일)
NM3_PER_TON = 1293.0      # Nm³/ton
MMBTU_PER_TON = 52.0      # MMBtu/ton


# ──────────────────────────────────────────────────────────────
# F3.1  수전단가 결정
# ──────────────────────────────────────────────────────────────

def get_season(month: int) -> str:
    """월 번호 → 계절 키 반환."""
    if month in (6, 7, 8):
        return "summer"
    elif month in (3, 4, 5, 9, 10):
        return "spring_fall"
    return "winter"


def _is_legal_holiday(d: date) -> bool:
    """일요일 또는 법정·대체공휴일 여부 (임시공휴일 제외)."""
    return d in LEGAL_HOLIDAYS or d.weekday() == 6


def _is_saturday(d: date) -> bool:
    return d.weekday() == 5


def get_load_type(d: date, hour: int) -> str:
    """
    날짜·시간 → '경부하' / '중간부하' / '최대부하'.

    규칙:
    - 일요일·공휴일 → 전 시간 경부하
    - 토요일 → 08~22시 중간부하, 나머지 경부하
    - 겨울 평일 → 최대: 09~12, 16~19 / 중간: 08~09, 12~16, 19~22
    - 봄가을·여름 평일 → 최대: 15~21 / 중간: 08~15, 21~22
    """
    month = d.month
    if _is_legal_holiday(d):
        return "경부하"
    if _is_saturday(d):
        return "중간부하" if 8 <= hour < 22 else "경부하"

    season = get_season(month)
    if hour < 8 or hour >= 22:
        return "경부하"

    if season == "winter":
        if hour < 9 or (12 <= hour < 16) or (19 <= hour < 22):
            return "중간부하"
        return "최대부하"
    else:
        if hour < 15 or hour >= 21:
            return "중간부하"
        return "최대부하"


def get_elec_price(d: date, hour: int) -> float:
    """
    날짜·시간 → 실제 수전단가 (원/kWh).
    봄가을 주말 11~13시 50% 특별할인 적용.
    """
    season    = get_season(d.month)
    load_type = get_load_type(d, hour)
    rate      = ELEC_RATES[season][load_type]

    # 봄가을 특별할인
    is_weekend = _is_legal_holiday(d) or _is_saturday(d)
    if season == "spring_fall" and is_weekend and hour in SPRING_FALL_DISCOUNT_HOURS:
        rate = rate * 0.5

    return round(rate, 4)


# ──────────────────────────────────────────────────────────────
# F3.2  대체단가 계산
# ──────────────────────────────────────────────────────────────

def calc_replace_cost(
    mode: str,
    elec_price: float,
    smp: float,
    net_load_kw: float,
    lng_price: float,
    lng_heat: float,
    efficiency: float,
    exchange_rate: float,
) -> float:
    """
    대체단가 계산 (원/kWh).

    엑셀 공식 기준:
    - 1기:  (min(287.5, 순부하/1000) * 수전단가 + max(0, 287.5 - 순부하/1000) * SMP) / 287.5
    - 2기:  (max(0, min(575, 순부하/1000)-287.5) * 수전단가 + (287.5 - max(0,min(575,순부하/1000)-287.5)) * SMP) / 287.5
    - low2gi: 수전단가 (그대로)
    """
    net_mw = net_load_kw / 1_000.0

    if mode == "1gi":
        mix = min(287.5, net_mw) * elec_price + max(0, 287.5 - net_mw) * smp
        return mix / 287.5

    elif mode == "2gi":
        above = max(0.0, min(575.0, net_mw) - 287.5)
        mix   = above * elec_price + (287.5 - above) * smp
        return mix / 287.5

    else:   # low2gi
        return elec_price


# ──────────────────────────────────────────────────────────────
# F3.3  BEP 계산
# ──────────────────────────────────────────────────────────────

def calc_bep(
    mode: str,
    replace_cost: float,
    lng_heat: float,
    efficiency: float,
    exchange_rate: float,
) -> float | None:
    """
    BEP(Break-Even Point) 계산 ($/MMBtu).

    공식: (대체단가 / 효율) * 열량 * (1293 Nm³/ton) / (52 MMBtu/ton) / 환율 - 0.8

    [저부하(low2gi) 효율 처리]
    엑셀 원본 수식(R열)은 효율을 1.7로 하드코딩했으나,
    실제 예측 시에는 ML 모델이 학습한 efficiency 예측값을 그대로 사용한다.
    ML 모델 미존재(학습 데이터 부족) 시에만 config.LOW2GI_EFF_FALLBACK(≈1.591)을 폴백으로 사용.

    효율이 0이면 None 반환.
    """
    from config import LOW2GI_EFF_FALLBACK

    eff = efficiency

    # 저부하 모드: ML 예측 효율이 0이거나 미수신이면 실측 평균값으로 폴백
    if mode == "low2gi" and (eff == 0 or eff is None):
        eff = LOW2GI_EFF_FALLBACK

    if eff == 0 or eff is None:
        return None

    bep = (replace_cost / eff) * lng_heat * NM3_PER_TON / MMBTU_PER_TON / exchange_rate - OVERHEAD_COST
    return round(bep, 4)


# ──────────────────────────────────────────────────────────────
# F3.4  경제성 판단
# ──────────────────────────────────────────────────────────────

def calc_economics(
    lng_price: float,
    bep: float | None,
    lng_gen_kw: float,
    efficiency: float,
    lng_heat: float,
    exchange_rate: float,
) -> dict:
    """
    경제성(억원) — 사용자 정의식:
      (BEP - LNG가격) [$/MMBtu]
      × 효율(Mcal/kWh) × 발전량(kW) / 열량(Mcal/Nm³)
      / (1293 Nm³/ton) × (52 MMBtu/ton)
      × 환율(원/$) / 10^8(원/억원)

    연료량: 발전량(kW)×1h를 kWh로 두고, 효율·열량으로 Nm³ → ton → MMBtu.

    경제성차이(원/kWh):
      위 식으로 구한 해당 1시간 총액(원)을 LNG발전량(kW)으로 나눈 값.
      (BEP−LNG) 마진을 전력 kWh당으로 환산한 지표.

    Returns:
        econ_diff: 원/kWh, econ_bil: 해당 1시간 경제성(억원), viable: (BEP−LNG)×MMBtu>0
    """
    if (
        bep is None
        or efficiency is None
        or efficiency <= 0
        or lng_heat <= 0
        or lng_gen_kw <= 0
        or not np.isfinite(lng_gen_kw)
    ):
        return {"econ_diff": 0.0, "econ_bil": 0.0, "viable": False}

    vol_nm3 = (lng_gen_kw * efficiency) / lng_heat
    mmbtu = (vol_nm3 / NM3_PER_TON) * MMBTU_PER_TON
    usd_h = (bep - lng_price) * mmbtu
    won_h = usd_h * exchange_rate
    econ_bil = won_h / 1e8
    econ_diff = won_h / lng_gen_kw

    return {
        "econ_diff": round(float(econ_diff), 4),
        "econ_bil": round(float(econ_bil), 6),
        "viable": float(usd_h) > 0,
    }


# ──────────────────────────────────────────────────────────────
# F3.5  최적 운전모드 선정
# ──────────────────────────────────────────────────────────────

def get_best_mode(mode_results: dict[str, dict]) -> str:
    """
    모드별 경제성 딕셔너리에서 최적 운전모드 반환.

    모든 모드가 경제성 < 0이면 'off' 반환.
    """
    best_mode = "off"
    best_diff = 0.0   # 임계값: 0보다 커야 가동

    for mode, res in mode_results.items():
        if res.get("econ_diff", 0) > best_diff:
            best_diff = res["econ_diff"]
            best_mode = mode

    return best_mode


# ──────────────────────────────────────────────────────────────
# 24시간 경제성 테이블 생성
# ──────────────────────────────────────────────────────────────

def _lng_gen_for_hour(
    hour: int,
    lng_gen_series: list[float | None] | None,
    net_load_kw: float,
) -> float:
    """해당 시각 실측 LNG발전량(kW); 없으면 순부하 대용."""
    if lng_gen_series is not None and hour < len(lng_gen_series):
        v = lng_gen_series[hour]
        if v is not None:
            try:
                x = float(v)
                if x > 0 and np.isfinite(x):
                    return x
            except (TypeError, ValueError):
                pass
    return float(net_load_kw)


def build_hourly_table(
    target_date: date,
    smp_series: list[float],        # 시간별 SMP 24개
    lng_price: float,
    lng_heat: float,
    exchange_rate: float,
    pred_results: dict,             # {mode: {hour: {export, import_kw, efficiency}}}
    net_load_kw: float = 280_000.0, # 기본 순부하 (발전량 미입력 시)
    lng_gen_series: list[float | None] | None = None,
) -> pd.DataFrame:
    """
    24시간 경제성 분석 테이블 생성.

    Returns:
        DataFrame with columns:
          시간, SMP, 수전단가, 운전모드, 대체단가, BEP,
          경제성차이(원/kWh), 경제성(억·해당1시간), 최적모드
    """
    from config import MODES

    records = []
    for hour in range(24):
        smp        = smp_series[hour] if hour < len(smp_series) else 0.0
        elec_price = get_elec_price(target_date, hour)
        lng_gen_h  = _lng_gen_for_hour(hour, lng_gen_series, net_load_kw)
        row_base   = {
            "시간":     f"{hour:02d}:00",
            "SMP(원/kWh)":  smp,
            "수전단가(원/kWh)": elec_price,
        }

        mode_results: dict[str, dict] = {}

        for mode in MODES:
            preds = pred_results.get(mode, {}).get(hour, {})
            eff   = preds.get("efficiency", 0.0)

            # 저부하: ML 예측 효율 우선, 0이면 실측 평균 폴백 (1.7 하드코딩 아님)
            if mode == "low2gi" and eff == 0.0:
                from config import LOW2GI_EFF_FALLBACK
                eff = LOW2GI_EFF_FALLBACK

            # 1기/2기: 예측값 없으면 모드별 기본값
            if eff == 0.0:
                eff = 1.595 if mode == "1gi" else 1.575

            nl    = net_load_kw

            rc    = calc_replace_cost(mode, elec_price, smp, nl, lng_price, lng_heat, eff, exchange_rate)
            bep   = calc_bep(mode, rc, lng_heat, eff, exchange_rate)
            econ  = calc_economics(lng_price, bep, lng_gen_h, eff, lng_heat, exchange_rate)

            mode_results[mode] = {**econ, "replace_cost": rc, "bep": bep}

            row_base[f"대체단가_{MODE_LABELS[mode]}"] = round(rc, 2)
            row_base[f"BEP_{MODE_LABELS[mode]}"] = (
                round(bep, 3) if bep is not None else np.nan
            )
            row_base[f"경제성차이_{MODE_LABELS[mode]}"] = round(econ["econ_diff"], 2)
            row_base[f"경제성(억)_{MODE_LABELS[mode]}"] = round(econ["econ_bil"], 3)

        best = get_best_mode(mode_results)
        row_base["최적모드"] = MODE_LABELS.get(best, best)

        records.append(row_base)

    return pd.DataFrame(records)
