"""
app.py
======
LNG 발전 경제성 분석 Streamlit 대시보드 진입점.

탭 구성:
  1. ML 예측   (F2)
  2. 경제성 분석 (F3)
  3. 이상치 탐지 (F4)

실행:
  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 모듈 경로에 추가
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modules"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime, timedelta

from config import (
    DATA_PATH, MODES, MODE_LABELS, MODE_COLORS,
    COLOR_PROFIT, COLOR_LOSS, COLOR_WARNING,
)


def _data_mtime() -> float:
    """`data/데이터.csv`가 바뀌면 캐시가 무효화되도록 수정 시각을 사용."""
    p = Path(DATA_PATH)
    return float(p.stat().st_mtime) if p.is_file() else 0.0
from modules.ml_predictor import (
    load_data, build_features, load_models, retrain,
    predict_day, classify_mode,
)
from modules.economics_engine import (
    get_elec_price, build_hourly_table,
)
from modules.anomaly_detector import (
    detect_smp_anomalies, detect_econ_change,
    build_smp_chart, build_econ_change_chart,
)


def _hourly_optional_series(day_df: pd.DataFrame, col: str) -> list | None:
    """선택 날짜 데이터프레임에서 0~23시 값 리스트 (해당 시각 없으면 None)."""
    if day_df.empty or col not in day_df.columns:
        return None
    out: list = []
    for h in range(24):
        rows = day_df[day_df["datetime"].dt.hour == h]
        if rows.empty:
            out.append(None)
        else:
            v = rows[col].iloc[0]
            out.append(float(v) if pd.notna(v) else None)
    return out

# ──────────────────────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LNG 발전 경제성 대시보드",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 메인 폰트 */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+KR:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans KR', sans-serif; }
    code, .stCode { font-family: 'IBM Plex Mono', monospace; }

    /* 헤더 */
    .main-header {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
        color: #e0e8ff;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #4A90D9;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 0.85rem; color: #8899bb; }

    /* 메트릭 카드 */
    .metric-card {
        background: #f8faff;
        border: 1px solid #e0e8ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.78rem; color: #666; font-weight: 600; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 600; margin-top: 0.2rem; }
    .metric-card.profit .value { color: #00C49F; }
    .metric-card.loss   .value { color: #FF4C4C; }
    .metric-card.neutral .value { color: #4A90D9; }

    /* 이상치 배지 */
    .anomaly-badge {
        background: #FF4C4C22;
        color: #FF4C4C;
        border: 1px solid #FF4C4C55;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .ok-badge {
        background: #00C49F22;
        color: #00C49F;
        border: 1px solid #00C49F55;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #f0f4ff;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] { background: #4A90D9; color: white; }

    /* 사이드바 */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] * { color: #c9d1d9; }
    [data-testid="stSidebar"] h2 { color: #58a6ff; font-size: 1rem; }
    [data-testid="stSidebar"] hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 캐싱: 데이터 로드
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="데이터 로딩 중...")
def cached_load_data(_mtime: float) -> pd.DataFrame:
    """CSV 로드 + 피처 엔지니어링. `_mtime`은 data 폴더의 CSV가 갱신되면 자동 반영."""
    df = load_data(DATA_PATH)
    return build_features(df)


@st.cache_resource(show_spinner="모델 로딩 중...")
def cached_load_models(_mtime: float, df: pd.DataFrame):
    """모델 로드 (없으면 자동 학습). 데이터 파일이 바뀌면 캐시가 새로 잡힘."""
    return load_models(df)


# ──────────────────────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ LNG 발전 경제성 분석 대시보드</h1>
  <p>ML 예측 · 룰베이스 경제성 계산 · 이상치 탐지 통합 플랫폼</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# 사이드바 입력 패널
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 운전 파라미터")
    st.markdown("---")

    lng_price     = st.number_input("LNG 가격 ($/MMBtu)",  min_value=1.0,  max_value=30.0, value=13.5,  step=0.1)
    exchange_rate = st.number_input("환율 (원/$)",         min_value=900.0, max_value=2000.0, value=1447.0, step=1.0)
    lng_heat      = st.number_input("LNG 열량 (Mcal/Nm³)", min_value=8.0,  max_value=11.0, value=9.10,  step=0.01)

    st.markdown("---")
    st.markdown("## 📅 분석 날짜")

    try:
        df_tmp = cached_load_data(_data_mtime())
        min_date = df_tmp["datetime"].min().date()
        max_date = df_tmp["datetime"].max().date()
    except Exception:
        min_date = date(2025, 1, 1)
        max_date = date(2026, 2, 28)

    target_date = st.date_input(
        "분석 날짜 선택",
        value=min_date + timedelta(days=90),
        min_value=min_date,
        max_value=max_date,
    )

    st.markdown("---")
    st.markdown("## 📊 데이터 현황")
    try:
        df_info = cached_load_data(_data_mtime())
        st.metric("총 데이터 수", f"{len(df_info):,}시간")
        st.metric("기간", f"{min_date} ~ {max_date}")
    except Exception:
        st.warning("데이터 로드 실패")


# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────
try:
    _mt = _data_mtime()
    df = cached_load_data(_mt)
    models, metrics = cached_load_models(_mt, df)
    data_ok = True
except FileNotFoundError:
    st.error(f"⚠️ 데이터 파일을 찾을 수 없습니다: `{DATA_PATH}`")
    st.info("프로젝트 루트에 `data/데이터.csv` 파일을 배치 후 재실행하세요.")
    st.stop()
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────
# 탭 구성
# ──────────────────────────────────────────────────────────────
tab_ml, tab_econ, tab_anomaly = st.tabs([
    "🤖 ML 예측",
    "💰 경제성 분석",
    "🔍 이상치 탐지",
])


# ══════════════════════════════════════════════════════════════
# TAB 1: ML 예측 (F2)
# ══════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("운전모드별 설비특성 ML 예측")

    col_retrain, col_info = st.columns([1, 3])

    with col_retrain:
        if st.button("🔄 모델 재학습", type="primary", use_container_width=True):
            with st.spinner("재학습 중... (수분 소요)"):
                try:
                    metrics = retrain(df)
                    cached_load_models.clear()
                    st.success("재학습 완료!")
                    st.rerun()
                except Exception as e:
                    st.error(f"재학습 실패: {e}")

    with col_info:
        st.caption(f"분석 날짜: **{target_date}** · LNG {lng_price}$/MMBtu · 환율 {exchange_rate:,.0f}원/$ · 열량 {lng_heat} Mcal/Nm³")

    # ── 모델 성능 지표 ──
    st.markdown("#### 📈 모델 성능 (train · CV · 시계열 test)")
    sp = metrics.get("_split") if metrics else None
    if sp:
        st.caption(
            f"시계열 분할: train **{sp.get('n_train', '?')}**행 "
            f"(`{sp.get('train_datetime_start', '')}` ~ `{sp.get('train_datetime_end', '')}`) · "
            f"test **{sp.get('n_test', '?')}**행 (최종 평가만, `{sp.get('test_datetime_start', '')}` ~ "
            f"`{sp.get('test_datetime_end', '')}`) · test 비율 **{sp.get('test_fraction', '')}**"
        )
    if metrics:
        perf_rows = []
        for mode in MODES:
            for target, mval in metrics.get(mode, {}).items():
                target_labels = {"export": "역송량(kW)", "import": "수전량(kW)", "efficiency": "효율(Mcal/kWh)"}
                perf_rows.append({
                    "운전모드": MODE_LABELS[mode],
                    "예측 대상": target_labels.get(target, target),
                    "MAE (train)": mval.get("mae", "-"),
                    "R² (train)": mval.get("r2", "-"),
                    "R² (CV)": mval.get("r2_cv", "-"),
                    "MAE (test)": mval.get("mae_test", "-"),
                    "R² (test)": mval.get("r2_test", "-"),
                    "n train": mval.get("n_samples", "-"),
                    "n test": mval.get("n_samples_test", "-"),
                })
        if perf_rows:
            perf_df = pd.DataFrame(perf_rows)

            def color_r2(val):
                try:
                    v = float(val)
                    if v >= 0.8:  return "color: #00C49F; font-weight:600"
                    elif v >= 0.5: return "color: #F5A623"
                    else:          return "color: #FF4C4C"
                except: return ""

            r2_cols = [c for c in perf_df.columns if "R²" in c]
            styled = perf_df.style.applymap(color_r2, subset=r2_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("모델을 먼저 학습하세요.")
    else:
        st.warning("모델 성능 정보가 없습니다. '재학습' 버튼을 눌러주세요.")

    st.divider()

    # ── 24시간 예측 결과 ──
    st.markdown(f"#### 📋 {target_date} 운전모드별 24시간 예측")

    # 해당 날짜 SMP 가져오기
    day_df = df[df["datetime"].dt.date == target_date]
    smp_series = day_df["smp"].fillna(0).tolist() if not day_df.empty else [0.0] * 24
    lng_gen_h = _hourly_optional_series(day_df, "lng_gen")
    net_load_h = _hourly_optional_series(day_df, "net_load")

    try:
        pred_results = predict_day(
            models, target_date, smp_series,
            lng_price, lng_heat, exchange_rate,
            get_elec_price,
            lng_gen_series=lng_gen_h,
            net_load_series=net_load_h,
        )

        for mode in MODES:
            with st.expander(f"**{MODE_LABELS[mode]}** 예측 결과", expanded=(mode == "2gi")):
                pred_rows = []
                for hour in range(24):
                    p = pred_results[mode][hour]
                    pred_rows.append({
                        "시간":         f"{hour:02d}:00",
                        "역송량 (kW)":  f"{p.get('export', 0):,.0f}",
                        "수전량 (kW)":  f"{p.get('import', 0):,.0f}",
                        "효율 (Mcal/kWh)": f"{p.get('efficiency', 0):.4f}",
                    })
                pred_df = pd.DataFrame(pred_rows)
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"예측 오류: {e}. 모델 재학습을 시도해보세요.")

    # ── 모드별 분포 차트 ──
    st.markdown("#### 📊 학습 데이터 운전모드 분포")
    mode_counts = df["mode"].value_counts().reset_index()
    mode_counts.columns = ["mode", "count"]
    mode_counts["label"] = mode_counts["mode"].map(MODE_LABELS)
    mode_counts["color"] = mode_counts["mode"].map(MODE_COLORS)

    fig_pie = px.pie(
        mode_counts, values="count", names="label",
        color="mode", color_discrete_map=MODE_COLORS,
        hole=0.4,
    )
    fig_pie.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: 경제성 분석 (F3)
# ══════════════════════════════════════════════════════════════
with tab_econ:
    st.subheader(f"📅 {target_date} 24시간 경제성 분석")
    st.caption(
        "**경제성(억원)** = (BEP−LNG가) $/MMBtu × 효율×발전량/열량 ÷1293(Nm³/ton) ×52(MMBtu/ton) × 환율 ÷10⁸"
    )

    # 해당 날짜 데이터
    day_df = df[df["datetime"].dt.date == target_date].copy()
    if day_df.empty:
        st.warning(f"선택 날짜({target_date}) 데이터가 없습니다.")
    else:
        smp_series = day_df["smp"].fillna(0).tolist()
        lng_gen_h = _hourly_optional_series(day_df, "lng_gen")
        net_load_h = _hourly_optional_series(day_df, "net_load")

        # 예측값 생성 (또는 캐시 재사용)
        try:
            pred_results = predict_day(
                models, target_date, smp_series,
                lng_price, lng_heat, exchange_rate,
                get_elec_price,
                lng_gen_series=lng_gen_h,
                net_load_series=net_load_h,
            )
        except Exception:
            pred_results = {m: {h: {"export": 0, "import": 0, "efficiency": 1.58} for h in range(24)} for m in MODES}

        # 경제성 테이블 생성 (시간별 LNG발전량은 실측 시계열 사용)
        econ_table = build_hourly_table(
            target_date, smp_series,
            lng_price, lng_heat, exchange_rate,
            pred_results,
            lng_gen_series=lng_gen_h,
        )

        # ── 요약 KPI ──
        best_mode_col  = f"경제성차이_{MODE_LABELS['2gi']}"
        econ_bil_col   = f"경제성(억)_{MODE_LABELS['2gi']}"
        total_profit_h = (econ_table[best_mode_col] > 0).sum() if best_mode_col in econ_table.columns else 0
        avg_econ       = econ_table[best_mode_col].mean() if best_mode_col in econ_table.columns else 0
        total_bil      = econ_table[econ_bil_col].sum() if econ_bil_col in econ_table.columns else 0

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            cls = "profit" if total_profit_h > 12 else "loss"
            st.markdown(f"""<div class="metric-card {cls}">
                <div class="label">경제성 양수 시간</div>
                <div class="value">{total_profit_h}h</div></div>""", unsafe_allow_html=True)
        with kpi2:
            cls = "profit" if avg_econ > 0 else "loss"
            st.markdown(f"""<div class="metric-card {cls}">
                <div class="label">평균 경제성 (원/kWh)</div>
                <div class="value">{avg_econ:.1f}</div></div>""", unsafe_allow_html=True)
        with kpi3:
            cls = "profit" if total_bil > 0 else "loss"
            st.markdown(f"""<div class="metric-card {cls}">
                <div class="label">일일 경제성 합계 (억원)</div>
                <div class="value">{total_bil:.3f}</div></div>""", unsafe_allow_html=True)
        with kpi4:
            opt_cnt = (econ_table["최적모드"] == MODE_LABELS["2gi"]).sum()
            st.markdown(f"""<div class="metric-card neutral">
                <div class="label">{MODE_LABELS['2gi']} 최적 시간</div>
                <div class="value">{opt_cnt}h</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── 경제성 차트 (full부하 기준) ──
        if best_mode_col in econ_table.columns:
            econ_table["_color"] = econ_table[best_mode_col].apply(
                lambda x: COLOR_PROFIT if x > 0 else COLOR_LOSS
            )
            fig_econ = go.Figure()
            fig_econ.add_trace(go.Bar(
                x=econ_table["시간"],
                y=econ_table[best_mode_col],
                marker_color=econ_table["_color"],
                name=f"경제성 차이 ({MODE_LABELS['2gi']} 기준)",
                hovertemplate="<b>%{x}</b><br>경제성: %{y:.2f} 원/kWh<extra></extra>",
            ))
            fig_econ.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.5)
            fig_econ.update_layout(
                title=f"{target_date} 시간별 경제성차이 ({MODE_LABELS['2gi']}, BEP−LNG·연료 MMBtu 기준 원/kWh)",
                xaxis_title="시간", yaxis_title="경제성 (원/kWh)",
                height=380, showlegend=False,
            )
            st.plotly_chart(fig_econ, use_container_width=True)

        # ── 최적 운전모드 Gantt 스타일 ──
        mode_color_map = {v: MODE_COLORS.get(k, "#999") for k, v in MODE_LABELS.items()}
        fig_mode = go.Figure()
        for _, row in econ_table.iterrows():
            hour_num = int(row["시간"].split(":")[0])
            color = mode_color_map.get(row["최적모드"], "#999")
            fig_mode.add_trace(go.Bar(
                x=[row["시간"]], y=[1],
                marker_color=color,
                name=row["최적모드"],
                showlegend=False,
                hovertemplate=f"<b>{row['시간']}</b><br>최적모드: {row['최적모드']}<extra></extra>",
            ))

        # 범례
        for label, color in mode_color_map.items():
            fig_mode.add_trace(go.Bar(x=[None], y=[None], name=label, marker_color=color))

        fig_mode.update_layout(
            title="시간별 최적 운전모드",
            barmode="stack",
            xaxis_title="시간", yaxis=dict(showticklabels=False),
            height=180,
            legend=dict(orientation="h", yanchor="bottom", y=1.05),
        )
        st.plotly_chart(fig_mode, use_container_width=True)

        fx1, fx2 = st.columns(2)
        with fx1:
            st.metric("적용 LNG 가격 (사이드바)", f"{lng_price:.2f} $/MMBtu")
        with fx2:
            st.metric("적용 환율", f"{exchange_rate:,.0f} 원/$")

        # ── 상세 테이블 ──
        with st.expander("📋 24시간 상세 경제성 테이블", expanded=False):
            display_cols = ["시간", "SMP(원/kWh)", "수전단가(원/kWh)", "최적모드"]
            for mode in MODES:
                label = MODE_LABELS[mode]
                for suffix in ["대체단가", "BEP", "경제성(억)"]:
                    col = f"BEP_{label}" if suffix == "BEP" else f"{suffix}_{label}"
                    if col in econ_table.columns:
                        display_cols.append(col)

            sub = econ_table[[c for c in display_cols if c in econ_table.columns]]

            def highlight_bil(val):
                try:
                    v = float(val)
                    if v > 0:   return "background-color: #e6fff5; color: #007a5e"
                    elif v < 0: return "background-color: #fff0f0; color: #a00"
                except: pass
                return ""

            num_cols = list(sub.select_dtypes(include=["number"]).columns)
            bil_cols = [c for c in num_cols if "경제성(억)" in c]
            other_num = [c for c in num_cols if c not in bil_cols]
            styled = sub.style
            if bil_cols:
                styled = styled.format("{:.3f}", subset=bil_cols, na_rep="")
            if other_num:
                styled = styled.format("{:.1f}", subset=other_num, na_rep="")
            if bil_cols:
                styled = styled.applymap(highlight_bil, subset=bil_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: 이상치 탐지 (F4)
# ══════════════════════════════════════════════════════════════
with tab_anomaly:
    st.subheader("🔍 이상치 탐지")

    # 기간 필터
    col_from, col_to = st.columns(2)
    with col_from:
        anomaly_from = st.date_input("분석 시작", value=df["datetime"].min().date(), key="anom_from")
    with col_to:
        anomaly_to   = st.date_input("분석 종료", value=df["datetime"].max().date(), key="anom_to")

    anom_df = df[(df["datetime"].dt.date >= anomaly_from) & (df["datetime"].dt.date <= anomaly_to)].copy()

    if anom_df.empty:
        st.warning("선택 기간에 데이터가 없습니다.")
    else:
        # ── F4.1 SMP 이상구간 ──
        st.markdown("### F4.1  SMP 이상구간 탐지")

        smp_anomalies = detect_smp_anomalies(anom_df)

        cnt_zero = (smp_anomalies["anomaly_type"] == "SMP 제로").sum() if not smp_anomalies.empty else 0
        cnt_high = (smp_anomalies["anomaly_type"] == "SMP 과대").sum() if not smp_anomalies.empty else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            badge = f'<span class="anomaly-badge">⚠ {len(smp_anomalies)}건</span>' if len(smp_anomalies) > 0 else '<span class="ok-badge">✓ 정상</span>'
            st.markdown(f"**전체 이상 감지** {badge}", unsafe_allow_html=True)
        with c2:
            st.metric("SMP 제로 구간", f"{cnt_zero}시간")
        with c3:
            st.metric("SMP 과대 구간 (≥170)", f"{cnt_high}시간")

        # SMP 차트
        fig_smp = build_smp_chart(anom_df, smp_anomalies)
        st.plotly_chart(fig_smp, use_container_width=True)

        # 이상 시간대 리스트
        if not smp_anomalies.empty:
            with st.expander(f"📋 이상 시간대 상세 목록 ({len(smp_anomalies)}건)", expanded=False):
                st.dataframe(
                    smp_anomalies.style.apply(
                        lambda col: ["background-color:#fff0f0" if v == "SMP 제로" else "background-color:#fff8e6" for v in col]
                        if col.name == "anomaly_type" else [""] * len(col),
                    ),
                    use_container_width=True, hide_index=True,
                )

        st.divider()

        # ── F4.2 경제성 급변 구간 ──
        st.markdown("### F4.2  경제성 급변 구간 탐지")

        econ_col_options = {
            "full부하": "econ_diff_2gi",
            "1기":        "econ_diff_1gi",
            "2기 저부하": "econ_diff_low",
        }
        selected_mode_label = st.selectbox("운전모드 선택", list(econ_col_options.keys()))
        selected_col        = econ_col_options[selected_mode_label]

        threshold = st.slider("급변 임계값 (원/kWh)", min_value=10, max_value=200, value=50, step=5)

        if selected_col in anom_df.columns:
            change_df = detect_econ_change(anom_df, selected_col, threshold)

            c1, c2 = st.columns(2)
            with c1:
                badge = f'<span class="anomaly-badge">⚠ {len(change_df)}건</span>' if len(change_df) > 0 else '<span class="ok-badge">✓ 없음</span>'
                st.markdown(f"**급변 구간** {badge}", unsafe_allow_html=True)
            with c2:
                if not change_df.empty:
                    st.metric("최대 변화폭", f"{change_df['delta'].max():.1f} 원/kWh")

            fig_change = build_econ_change_chart(anom_df, change_df, selected_col)
            st.plotly_chart(fig_change, use_container_width=True)

            if not change_df.empty:
                with st.expander("📋 급변 구간 상세 목록", expanded=False):
                    st.dataframe(change_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"경제성 컬럼 '{selected_col}'이 데이터에 없습니다. 컬럼명을 확인하세요.")

        st.divider()

        # ── 종합 이상치 분포 히트맵 ──
        st.markdown("### 📊 월별 이상치 분포")
        if not smp_anomalies.empty:
            smp_anomalies["month"] = pd.to_datetime(smp_anomalies["datetime"]).dt.month
            smp_anomalies["day"]   = pd.to_datetime(smp_anomalies["datetime"]).dt.day

            pivot = smp_anomalies.groupby(["month", "anomaly_type"]).size().reset_index(name="count")
            fig_bar = px.bar(
                pivot, x="month", y="count", color="anomaly_type",
                color_discrete_map={"SMP 제로": "#FF4C4C", "SMP 과대": "#FFA500"},
                labels={"month": "월", "count": "이상 건수", "anomaly_type": "유형"},
                title="월별 SMP 이상 건수",
            )
            fig_bar.update_layout(height=300)
            st.plotly_chart(fig_bar, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# 푸터
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.78rem;'>"
    "LNG 발전 경제성 대시보드 · Python 3.10+ · Streamlit · XGBoost · Plotly"
    "</div>",
    unsafe_allow_html=True,
)
