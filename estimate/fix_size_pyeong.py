"""
fix_size_pyeong.py — size_pyeong이 0인 JSON에서 request_body_text로 보완

실행:
    python fix_size_pyeong.py          # dry-run (변경 없이 확인만)
    python fix_size_pyeong.py --apply  # 실제 저장
"""

import re
import json
import argparse
import pathlib

DATA_DIR = "./estimate_data"

# 평수 추출 패턴 (우선순위 순)
PATTERNS = [
    r"(\d+)\s*평형",                         # "25평형"
    r"(\d+)\s*평\s*(?:아파트|형|짜리|대)",   # "30평 아파트"
    r"전용\s*(\d+(?:\.\d+)?)\s*평",          # "전용 24.5평"
    r"(\d+)\s*[Pp][Yy]\b",                   # "28PY", "26py"
    r"(\d+)\s*평",                            # "30평" (가장 일반적)
]

def extract_size(text: str) -> int:
    """텍스트에서 평수 추출. 못 찾으면 0 반환."""
    if not text:
        return 0
    for pattern in PATTERNS:
        m = re.search(pattern, text)
        if m:
            val = float(m.group(1))
            # 비현실적인 값 제외 (10평 미만이거나 100평 초과)
            if 10 <= val <= 100:
                return int(val)
    return 0


def process_all(apply: bool):
    data_root = pathlib.Path(DATA_DIR)
    fixed = skipped = already_ok = 0

    for region_dir in sorted(data_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue
            json_path = article_dir / f"{article_dir.name}.json"
            if not json_path.exists():
                continue

            data = json.loads(json_path.read_text(encoding="utf-8"))
            current = data.get("size_pyeong") or 0

            if current > 0:
                already_ok += 1
                continue

            # request_body_text → body_text 순으로 평수 추출 시도
            size = extract_size(data.get("request_body_text", ""))
            if not size:
                size = extract_size(data.get("body_text", ""))
            if not size:
                size = extract_size(data.get("post_title", ""))

            if not size:
                print(f"  [FAIL] {article_dir.name}: 평수 추출 불가")
                skipped += 1
                continue

            print(f"  [FIX]  {article_dir.name}: 0 → {size}평")

            if apply:
                data["size_pyeong"] = size
                json_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            fixed += 1

    print(f"\n결과: 보완 {fixed}건 / 추출실패 {skipped}건 / 이미정상 {already_ok}건")
    if not apply and fixed > 0:
        print("--apply 플래그로 실행하면 실제 저장됩니다.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="실제 저장 (기본: dry-run)")
    args = parser.parse_args()
    process_all(apply=args.apply)


if __name__ == "__main__":
    main()
