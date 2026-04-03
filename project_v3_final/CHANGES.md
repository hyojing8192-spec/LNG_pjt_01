# F2/F3/F4 수정 사항 (v3)

## F2 ml_predictor.py
- **R² 마이너스 수정**: `_safe_r2_mean()` 추가 — 극단 음수(< -1.0) fold 제거 후 평균
- **분산 체크**: efficiency 등 std < 0.005 타깃은 안정적 상수 처리 (R²=-∞ 방지)
- **TimeSeriesSplit 개선**: min_train_size 보장으로 첫 fold 폭발 방지
- **low2gi 보강 기준 변경**: 380k~440k 구간으로 정확히 제한 (v9 저부하 데이터 반영)

## F3 economics_engine.py
- **calc_economics 수식 수정**: 엑셀 원본과 일치
  - `경제성차이(원/kWh) = SMP - 대체단가` (이전: BEP 기반 복잡 계산)
  - `경제성(억원/월) = 경제성차이 × 발전량 × 52h / 1억`
- viable 기준: `BEP > LNG가` OR `경제성차이 > 0`

## F4 anomaly_detector.py
- **SMP 임계값 파라미터화**: `detect_smp_anomalies(df, zero_threshold, high_threshold)`
- **build_smp_chart**: high_threshold 파라미터 추가

## app.py
- **환율 수동 오버라이드**: 사이드바에 입력창 추가 (전일 평균 자동 + 수동 수정 가능)
- **SMP 탐지 슬라이더**: SMP 과대/제로 기준 실시간 조정
- **F4.2 경제성 컬럼 없을 때**: SMP로 대체 탐지 (빈 화면 방지)
- **Gantt 차트 성능**: 24개 개별 trace → 단일 Bar 트레이스로 교체
