"""
anomaly_detector.py  (F4)
=========================
주요 함수:
  - detect_smp_anomalies()       : SMP 이상구간 탐지 (F4.1)
  - detect_econ_change()         : 경제성 급변 구간 탐지 (F4.2)
  - annotate_anomalies_on_fig()  : Plotly 차트에 이상구간 강조 표시
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import (
    SMP_ZERO_THRESHOLD, SMP_HIGH_THRESHOLD,
    ECON_CHANGE_THRESHOLD, COLOR_ANOMALY,
)


# ──────────────────────────────────────────────────────────────
# F4.1  SMP 이상구간 탐지
# ──────────────────────────────────────────────────────────────

def detect_smp_anomalies(
    df: pd.DataFrame,
    zero_threshold: float = SMP_ZERO_THRESHOLD,
    high_threshold: float = SMP_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """
    SMP 이상구간 탐지.

    조건:
      - SMP ≤ zero_threshold → 유형: "SMP 제로"
      - SMP ≥ high_threshold → 유형: "SMP 과대"

    Args:
        df:              'datetime', 'smp' 컬럼 포함 DataFrame
        zero_threshold:  제로 기준 (기본 0)
        high_threshold:  과대 기준 (기본 170, 슬라이더로 조정 가능)

    Returns:
        이상 행만 추출한 DataFrame (컬럼: datetime, smp, anomaly_type)
    """
    if "smp" not in df.columns:
        return pd.DataFrame(columns=["datetime", "smp", "anomaly_type"])

    mask_zero = df["smp"] <= zero_threshold
    mask_high = df["smp"] >= high_threshold

    results = []
    for _, row in df[mask_zero | mask_high].iterrows():
        atype = "SMP 제로" if row["smp"] <= zero_threshold else "SMP 과대"
        results.append({
            "datetime":     row["datetime"],
            "smp":          row["smp"],
            "anomaly_type": atype,
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────
# F4.2  경제성 급변 구간 탐지
# ──────────────────────────────────────────────────────────────

def detect_econ_change(
    df: pd.DataFrame,
    econ_col: str = "econ_diff_2gi",
    threshold: float = ECON_CHANGE_THRESHOLD,
) -> pd.DataFrame:
    """
    전 시간 대비 경제성 변화율이 임계값을 초과하는 구간 탐지.

    Args:
        df:        'datetime', econ_col 컬럼 포함 DataFrame
        econ_col:  경제성 컬럼명
        threshold: 변화량 임계값 (원/kWh)

    Returns:
        급변 행 DataFrame (datetime, econ_val, delta, direction)
    """
    if econ_col not in df.columns:
        return pd.DataFrame()

    df = df.copy().sort_values("datetime")
    df["_delta"] = df[econ_col].diff().abs()

    mask = df["_delta"] >= threshold
    result = df[mask][["datetime", econ_col, "_delta"]].copy()
    result.columns = ["datetime", "econ_val", "delta"]
    result["direction"] = df.loc[mask, econ_col].diff().apply(
        lambda x: "↑ 급등" if x > 0 else "↓ 급락"
    )
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# Plotly 차트 - SMP 이상구간 시각화
# ──────────────────────────────────────────────────────────────

def build_smp_chart(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    high_threshold: float = SMP_HIGH_THRESHOLD,
) -> go.Figure:
    """
    SMP 시계열 라인차트 + 이상구간 빨간 마커 오버레이.

    Args:
        df:        'datetime', 'smp' 컬럼 포함 DataFrame
        anomalies: detect_smp_anomalies() 반환값

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # 기본 SMP 라인
    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df["smp"],
        mode="lines",
        name="SMP (원/kWh)",
        line=dict(color="#4A90D9", width=1.5),
    ))

    # 이상치 마커
    if not anomalies.empty:
        zero_df = anomalies[anomalies["anomaly_type"] == "SMP 제로"]
        high_df = anomalies[anomalies["anomaly_type"] == "SMP 과대"]

        if not zero_df.empty:
            fig.add_trace(go.Scatter(
                x=zero_df["datetime"], y=zero_df["smp"],
                mode="markers",
                name="SMP 제로",
                marker=dict(color=COLOR_ANOMALY, size=8, symbol="circle"),
            ))

        if not high_df.empty:
            fig.add_trace(go.Scatter(
                x=high_df["datetime"], y=high_df["smp"],
                mode="markers",
                name="SMP 과대(≥170)",
                marker=dict(color="#FF8C00", size=8, symbol="triangle-up"),
            ))

    # 임계선
    fig.add_hline(
        y=high_threshold, line_dash="dash",
        line_color="#FF8C00", opacity=0.6,
        annotation_text=f"과대 기준 {high_threshold}원",
    )
    fig.add_hline(
        y=0, line_dash="dot",
        line_color=COLOR_ANOMALY, opacity=0.6,
        annotation_text="SMP=0",
    )

    fig.update_layout(
        title="SMP 이상구간 탐지",
        xaxis_title="시간",
        yaxis_title="SMP (원/kWh)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=420,
    )
    return fig


def build_econ_change_chart(
    df: pd.DataFrame,
    change_df: pd.DataFrame,
    econ_col: str = "econ_diff_2gi",
) -> go.Figure:
    """
    경제성 시계열 + 급변 구간 빨간 세로선 강조.

    Args:
        df:        원본 데이터 (datetime + econ_col)
        change_df: detect_econ_change() 반환값

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"], y=df[econ_col],
        mode="lines",
        name="경제성 차이 (원/kWh)",
        line=dict(color="#7ED321", width=1.5),
    ))

    # 제로 기준선
    fig.add_hline(y=0, line_dash="dash", line_color="#999", opacity=0.5)

    # 급변 지점
    if not change_df.empty:
        fig.add_trace(go.Scatter(
            x=change_df["datetime"],
            y=change_df["econ_val"],
            mode="markers",
            name=f"급변 (|Δ|≥{ECON_CHANGE_THRESHOLD})",
            marker=dict(color=COLOR_ANOMALY, size=10, symbol="x"),
            hovertemplate="시간: %{x}<br>경제성: %{y:.2f}<extra></extra>",
        ))

    fig.update_layout(
        title="경제성 급변 구간 탐지",
        xaxis_title="시간",
        yaxis_title="경제성 차이 (원/kWh)",
        hovermode="x unified",
        height=420,
    )
    return fig
