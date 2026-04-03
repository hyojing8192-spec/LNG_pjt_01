"""data_01.csv에서 빈 셀 '-' 패턴을 0으로 치환 (음수 -2.1 등은 유지)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data_01.csv"
# 원본이 Excel 등에 열려 있으면 덮어쓰기 실패 → 결과는 별도 파일로 저장
OUT = HERE / "data_01.csv"


def main() -> None:
    t = SRC.read_text(encoding="utf-8-sig")
    rep = 0
    while ", - ," in t:
        t = t.replace(", - ,", ", 0 ,")
        rep += 1
    for a, b in ((",-,", ",0,"), (",- ", ",0 "), (" -,", " 0,")):
        c = t.count(a)
        if c:
            t = t.replace(a, b)
            rep += c
    try:
        OUT.write_text(t, encoding="utf-8-sig")
        print(f"OK: 저장 {OUT} (', - ,' 치환 루프 {rep}회 + 기타 패턴)")
    except PermissionError:
        alt = HERE / "data_01_fixed.csv"
        alt.write_text(t, encoding="utf-8-sig")
        print(f"원본 잠금 → {alt} 로 저장했습니다. Excel을 닫은 뒤 이름을 바꿔 쓰세요.")


if __name__ == "__main__":
    main()
