"""
eval_auto.py — 골든셋 없이 즉시 실행 가능한 자동 평가

실행:
    python eval_auto.py

평가 항목:
    ① parsed_estimate 보유율  — 전체 수집 대비 파싱 성공 비율
    ② size_pyeong 미추출률    — 평수가 0 또는 없는 비율
    ③ 내부합산 일치율         — line_items 합계 vs total_cost (±20% 이내)
    ④ 중복집계 의심률         — 항목합계 > total_cost × 1.3 (대분류+소분류 중복 패턴)
    ⑤ 지역별 / 평수 구간별 분포

목표치:
    내부합산 일치율 80% 이상
    중복집계 의심률 0% (프롬프트 수정 후 재파싱으로 해결)
    size_pyeong 미추출률 10% 이하
"""

import json
import pathlib
from collections import Counter

DATA_DIR              = "./estimate_data"
CONSISTENCY_TOLERANCE = 0.20  # ±20% 이내면 일치
DUPLICATE_THRESHOLD   = 1.30  # 항목합계가 총금액의 1.3배 이상이면 중복 의심


# ══════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════

def load_all_json(data_root: pathlib.Path) -> list[dict]:
    records = []
    for region_dir in sorted(data_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue
            json_path = article_dir / f"{article_dir.name}.json"
            if not json_path.exists():
                continue
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                records.append(data)
            except Exception as e:
                print(f"  [WARN] JSON 로드 실패: {json_path} — {e}")
    return records


# ══════════════════════════════════════════════════════
# ① parsed_estimate 보유율
# ══════════════════════════════════════════════════════

def check_coverage(records: list[dict]) -> dict:
    total  = len(records)
    parsed = sum(1 for r in records if r.get("parsed_estimate"))
    return {
        "전체":   total,
        "파싱완료": parsed,
        "미파싱":  total - parsed,
        "보유율":  f"{parsed / total * 100:.1f}%" if total else "-",
    }


# ══════════════════════════════════════════════════════
# ② size_pyeong 미추출률
# ══════════════════════════════════════════════════════

def check_size_pyeong(records: list[dict]) -> dict:
    total   = len(records)
    missing = sum(1 for r in records if not r.get("size_pyeong"))
    return {
        "전체":          total,
        "미추출(0또는없음)": missing,
        "미추출률":       f"{missing / total * 100:.1f}%" if total else "-",
    }


# ══════════════════════════════════════════════════════
# ③ 내부합산 일치율  /  ④ 중복집계 의심률
# ══════════════════════════════════════════════════════

def check_internal_consistency(records: list[dict]) -> dict:
    results = []

    for r in records:
        pe = r.get("parsed_estimate")
        if not pe:
            continue

        total_cost = pe.get("total_cost") or 0
        article_id = r.get("article_id", "?")

        if not total_cost:
            results.append({"article_id": article_id, "status": "total_cost_없음"})
            continue

        line_sum = sum(
            item.get("amount") or 0
            for item in pe.get("line_items", [])
            if isinstance(item.get("amount"), (int, float))
        )

        ratio      = line_sum / total_cost
        error_rate = abs(line_sum - total_cost) / total_cost

        if ratio >= DUPLICATE_THRESHOLD:
            status = "중복집계_의심"
        elif error_rate <= CONSISTENCY_TOLERANCE:
            status = "일치"
        else:
            status = "불일치"

        results.append({
            "article_id": article_id,
            "total_cost": total_cost,
            "line_sum":   line_sum,
            "오차율":      round(error_rate * 100, 1),
            "status":     status,
        })

    counter = Counter(r["status"] for r in results)
    valid   = len(results)

    return {
        "검사대상":       valid,
        "일치":          counter.get("일치", 0),
        "불일치":         counter.get("불일치", 0),
        "중복집계_의심":   counter.get("중복집계_의심", 0),
        "total_cost_없음": counter.get("total_cost_없음", 0),
        "내부합산_일치율": f"{counter.get('일치', 0) / valid * 100:.1f}%" if valid else "-",
        "중복집계_의심률": f"{counter.get('중복집계_의심', 0) / valid * 100:.1f}%" if valid else "-",
        "상세":          results,
    }


# ══════════════════════════════════════════════════════
# ⑤ 데이터 분포
# ══════════════════════════════════════════════════════

def check_distribution(records: list[dict]) -> dict:
    region_counter = Counter(r.get("region", "미분류") for r in records)

    size_bins: dict[str, int] = {
        "~20평":   0,
        "21~30평": 0,
        "31~40평": 0,
        "41평~":   0,
        "미추출":   0,
    }
    for r in records:
        s = r.get("size_pyeong") or 0
        if s == 0:
            size_bins["미추출"] += 1
        elif s <= 20:
            size_bins["~20평"] += 1
        elif s <= 30:
            size_bins["21~30평"] += 1
        elif s <= 40:
            size_bins["31~40평"] += 1
        else:
            size_bins["41평~"] += 1

    return {
        "지역별":    dict(region_counter.most_common()),
        "평수구간별": size_bins,
    }


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None,
                        help="특정 article_id만 검사 (예: --id 877555)")
    parser.add_argument("--show-all", action="store_true",
                        help="불일치 목록 전체 출력")
    args = parser.parse_args()

    data_root = pathlib.Path(DATA_DIR)
    if not data_root.exists():
        print(f"[ERR] 폴더 없음: {DATA_DIR}")
        return

    records = load_all_json(data_root)

    if args.id:
        records = [r for r in records if r.get("article_id") == args.id]
        if not records:
            print(f"[ERR] article_id '{args.id}' 를 찾을 수 없음")
            return

    print(f"총 {len(records)}개 JSON 로드")

    # ① 보유율
    print_section("① parsed_estimate 보유율")
    cov = check_coverage(records)
    for k, v in cov.items():
        print(f"  {k}: {v}")

    # ② size_pyeong
    print_section("② size_pyeong 미추출률")
    sz = check_size_pyeong(records)
    for k, v in sz.items():
        print(f"  {k}: {v}")

    # ③④ 내부합산 + 중복집계
    print_section("③ 내부합산 일치율  /  ④ 중복집계 의심률")
    ic = check_internal_consistency(records)
    for k, v in ic.items():
        if k == "상세":
            continue
        print(f"  {k}: {v}")

    problems = [r for r in ic["상세"] if r["status"] != "일치"]
    dup_problems = [r for r in problems if r["status"] == "중복집계_의심"]
    if dup_problems:
        print(f"\n  [중복집계 의심 전체 목록]")
        for p in dup_problems:
            print(f"    {p['article_id']}: {p['status']}"
                  f" | 총금액 {p.get('total_cost', '-'):,}"
                  f" / 항목합계 {p.get('line_sum', '-'):,}"
                  f" / 오차 {p.get('오차율', '-')}%")
    if problems:
        limit = len(problems) if args.show_all else 10
        label = "전체" if args.show_all else "상위 10건"
        print(f"\n  [불일치 목록] ({label})")
        for p in [r for r in problems if r["status"] == "불일치"][:limit]:
            print(f"    {p['article_id']}: {p['status']}"
                  f" | 총금액 {p.get('total_cost', '-'):,}"
                  f" / 항목합계 {p.get('line_sum', '-'):,}"
                  f" / 오차 {p.get('오차율', '-')}%")
        mismatch = [r for r in problems if r["status"] == "불일치"]
        if not args.show_all and len(mismatch) > 10:
            print(f"    ... 외 {len(mismatch) - 10}건  (전체 보려면 --show-all)")

    # ⑤ 분포
    print_section("⑤ 데이터 분포")
    dist = check_distribution(records)
    print("  지역별:")
    for region, cnt in dist["지역별"].items():
        bar = "█" * cnt
        print(f"    {region:6s}: {cnt:4d}건  {bar}")
    print("  평수 구간별:")
    for bin_name, cnt in dist["평수구간별"].items():
        bar = "█" * cnt
        print(f"    {bin_name:8s}: {cnt:4d}건  {bar}")

    # 목표치 달성 여부 요약
    print_section("목표치 달성 여부")
    ic_rate  = ic.get("내부합산_일치율", "0%").rstrip("%")
    dup_rate = ic.get("중복집계_의심률", "100%").rstrip("%")
    sz_rate  = sz.get("미추출률", "100%").rstrip("%")

    def check(label, value_str, target, higher_is_better=True):
        try:
            val = float(value_str)
            ok  = val >= target if higher_is_better else val <= target
            mark = "✅" if ok else "❌"
        except Exception:
            mark = "?"
            val  = "?"
        print(f"  {mark} {label}: {value_str}% (목표: {'≥' if higher_is_better else '≤'}{target}%)")

    check("내부합산 일치율", ic_rate,  80.0, higher_is_better=True)
    check("중복집계 의심률", dup_rate,  0.0, higher_is_better=False)
    check("size_pyeong 미추출률", sz_rate, 10.0, higher_is_better=False)

    print()


if __name__ == "__main__":
    main()
