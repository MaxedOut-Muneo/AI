"""
eval_golden.py — PHASE 2: golden_vision.json 기반 파싱 정확도 평가

실행:
    python eval_golden.py
    python eval_golden.py --verbose   # 건별 상세 출력

입력:
    eval_data/golden_vision.json   — 팀원이 수작업 작성한 정답 레이블
    estimate_data/                 — 크롤링 + 파싱 결과 JSON

출력 지표:
    ① 총금액 오차율          abs(실제 - 파싱) / 실제 × 100  → 목표 평균 ±5%
    ② 카테고리별 소계 오차율  카테고리 소계 오차 평균        → 목표 평균 ±10%
    ③ 카테고리 누락률         이미지에 있는데 파싱에 없는 카테고리 비율

카테고리 매칭 전략 (순서대로 시도, 매칭된 파싱 항목은 재사용 불가):
    1단계: 이름 완전 일치
    2단계: 이름 부분 포함 (골든명 ⊂ 파싱명 또는 반대)
    3단계: 금액 ±10% 이내인 파싱 항목 중 금액 차이가 가장 작은 것
    → 매칭 실패 시 누락으로 판정
"""

import json
import pathlib
import argparse
from collections import defaultdict

GOLDEN_FILE = "./eval_data/golden_vision.json"
DATA_DIR    = "./estimate_data"

COST_ERROR_TARGET     = 5.0   # 총금액 오차율 목표 (%)
CATEGORY_ERROR_TARGET = 10.0  # 카테고리 소계 오차율 목표 (%)
AMOUNT_MATCH_TOLERANCE = 0.10  # 금액 기준 매칭 허용 오차 (3단계)


# ══════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════

def load_golden(path: str) -> list[dict]:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[ERR] 골든셋 파일 없음: {path}")
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [d for d in data if d.get("article_id")]


def find_json(data_root: pathlib.Path, article_id: str) -> pathlib.Path | None:
    for region_dir in data_root.iterdir():
        if not region_dir.is_dir():
            continue
        candidate = region_dir / article_id / f"{article_id}.json"
        if candidate.exists():
            return candidate
    return None


# ══════════════════════════════════════════════════════
# 파싱 결과에서 카테고리별 소계 계산
# ══════════════════════════════════════════════════════

def calc_parsed_category_totals(line_items: list[dict]) -> dict[str, int]:
    """line_items를 category로 묶어 소계 계산."""
    totals = defaultdict(int)
    for item in line_items:
        cat = item.get("category", "").strip()
        amt = item.get("amount") or 0
        if cat and cat != "0":  # category가 "0"인 오파싱 항목 제외
            totals[cat] += int(amt)
    return dict(totals)


# ══════════════════════════════════════════════════════
# 카테고리 매칭 (3단계, 매칭 항목 소비 처리)
# ══════════════════════════════════════════════════════

def match_categories(golden_cats: dict[str, int],
                     parsed_cats: dict[str, int]) -> dict[str, dict]:
    """
    골든셋 카테고리 ↔ 파싱 카테고리를 3단계로 매칭.
    매칭된 파싱 항목은 used_parsed에 등록해 재사용 방지 (다대일 충돌 차단).

    반환값: {골든_카테고리명: {"parsed_key": str|None, "parsed_amt": int|None,
                               "match_stage": int|None, "missing": bool}}
    """
    used_parsed = set()
    result = {}

    # ── 1단계: 이름 완전 일치 ──────────────────────────
    for g_name in golden_cats:
        if g_name in parsed_cats and g_name not in used_parsed:
            result[g_name] = {
                "parsed_key":   g_name,
                "parsed_amt":   parsed_cats[g_name],
                "match_stage":  1,
                "missing":      False,
            }
            used_parsed.add(g_name)

    # ── 2단계: 이름 부분 포함 ──────────────────────────
    for g_name in golden_cats:
        if g_name in result:
            continue
        g_norm = g_name.replace(" ", "")
        for p_name, p_amt in parsed_cats.items():
            if p_name in used_parsed:
                continue
            p_norm = p_name.replace(" ", "")
            if g_norm in p_norm or p_norm in g_norm:
                result[g_name] = {
                    "parsed_key":   p_name,
                    "parsed_amt":   p_amt,
                    "match_stage":  2,
                    "missing":      False,
                }
                used_parsed.add(p_name)
                break

    # ── 3단계: 금액 ±10% 이내, 차이 최소인 것 ──────────
    for g_name, g_amt in golden_cats.items():
        if g_name in result:
            continue
        if not g_amt:
            continue

        candidates = [
            (p_name, p_amt)
            for p_name, p_amt in parsed_cats.items()
            if p_name not in used_parsed
            and abs(p_amt - g_amt) / g_amt <= AMOUNT_MATCH_TOLERANCE
        ]
        if candidates:
            # 금액 차이가 가장 작은 것 선택
            best_name, best_amt = min(candidates, key=lambda x: abs(x[1] - g_amt))
            result[g_name] = {
                "parsed_key":   best_name,
                "parsed_amt":   best_amt,
                "match_stage":  3,
                "missing":      False,
            }
            used_parsed.add(best_name)

    # ── 매칭 실패 → 누락 ───────────────────────────────
    for g_name in golden_cats:
        if g_name not in result:
            result[g_name] = {
                "parsed_key":   None,
                "parsed_amt":   None,
                "match_stage":  None,
                "missing":      True,
            }

    return result


# ══════════════════════════════════════════════════════
# 건별 평가
# ══════════════════════════════════════════════════════

def evaluate_one(golden: dict, data_root: pathlib.Path, verbose: bool) -> dict | None:
    aid = golden["article_id"]
    json_path = find_json(data_root, aid)

    if json_path is None:
        print(f"  [SKIP] {aid}: JSON 파일 없음")
        return None

    record = json.loads(json_path.read_text(encoding="utf-8"))
    pe = record.get("parsed_estimate")

    if not pe:
        print(f"  [SKIP] {aid}: parsed_estimate 없음 (미파싱)")
        return None

    result = {"article_id": aid}

    # ─── ① 총금액 오차율 ─────────────────────────────
    actual_cost = golden.get("실제총금액") or 0
    parsed_cost = pe.get("total_cost") or 0

    if actual_cost and parsed_cost:
        cost_error = abs(actual_cost - parsed_cost) / actual_cost * 100
        result["actual_cost"]  = actual_cost
        result["parsed_cost"]  = parsed_cost
        result["cost_error_%"] = round(cost_error, 1)
    else:
        result["actual_cost"]  = actual_cost
        result["parsed_cost"]  = parsed_cost
        result["cost_error_%"] = None

    # ─── ② 카테고리별 소계 오차율 ────────────────────
    golden_cats = golden.get("카테고리별_소계") or {}
    parsed_cats = calc_parsed_category_totals(pe.get("line_items", []))
    matched     = match_categories(golden_cats, parsed_cats)

    cat_results  = {}
    missing_cats = []

    for g_name, g_amt in golden_cats.items():
        m = matched[g_name]
        if m["missing"]:
            missing_cats.append(g_name)
            cat_results[g_name] = {
                "actual": g_amt, "parsed": None,
                "error_%": None, "missing": True, "match_stage": None,
                "parsed_key": None,
            }
        elif g_amt > 0:
            err = abs(g_amt - m["parsed_amt"]) / g_amt * 100
            cat_results[g_name] = {
                "actual": g_amt, "parsed": m["parsed_amt"],
                "error_%": round(err, 1), "missing": False,
                "match_stage": m["match_stage"], "parsed_key": m["parsed_key"],
            }
        else:
            cat_results[g_name] = {
                "actual": g_amt, "parsed": m["parsed_amt"],
                "error_%": 0.0, "missing": False,
                "match_stage": m["match_stage"], "parsed_key": m["parsed_key"],
            }

    measurable_errors = [v["error_%"] for v in cat_results.values() if v["error_%"] is not None]
    avg_cat_error     = sum(measurable_errors) / len(measurable_errors) if measurable_errors else None
    missing_rate      = len(missing_cats) / len(golden_cats) * 100 if golden_cats else None

    result["cat_results"]     = cat_results
    result["missing_cats"]    = missing_cats
    result["avg_cat_error_%"] = round(avg_cat_error, 1) if avg_cat_error is not None else None
    result["missing_rate_%"]  = round(missing_rate, 1) if missing_rate is not None else None

    # ─── 상세 출력 ────────────────────────────────────
    if verbose:
        cost_str = (f"{result['cost_error_%']}%" if result["cost_error_%"] is not None else "측정불가")
        cat_str  = (f"{avg_cat_error:.1f}%" if avg_cat_error is not None else "측정불가")
        print(f"\n  [{aid}]")
        print(f"    총금액:        실제 {actual_cost:,} / 파싱 {parsed_cost:,} → 오차 {cost_str}")
        print(f"    카테고리 소계: 평균 오차 {cat_str} | 누락 {missing_cats or '없음'}")
        if golden_cats:
            stage_label = {1: "완전일치", 2: "부분일치", 3: "금액매칭"}
            for cat, v in cat_results.items():
                if v["missing"]:
                    print(f"      ❌ {cat}: 실제 {v['actual']:,} → 파싱 누락")
                else:
                    mark  = "✅" if v["error_%"] <= CATEGORY_ERROR_TARGET else "⚠️"
                    stage = stage_label.get(v["match_stage"], "?")
                    alias = f" ← '{v['parsed_key']}'" if v["parsed_key"] != cat else ""
                    print(f"      {mark} {cat}{alias} [{stage}]: "
                          f"실제 {v['actual']:,} / 파싱 {v['parsed']:,} → {v['error_%']}%")

    return result


# ══════════════════════════════════════════════════════
# 전체 집계
# ══════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="건별 카테고리 소계 상세 출력")
    args = parser.parse_args()

    golden_list = load_golden(GOLDEN_FILE)
    if not golden_list:
        print("[ERR] 골든셋이 비어 있습니다.")
        print(f"  → {GOLDEN_FILE} 에 정답 레이블을 작성해주세요.")
        print("  → eval_data/README.md 참고")
        return

    print(f"골든셋 {len(golden_list)}개 로드")

    data_root = pathlib.Path(DATA_DIR)
    results   = []

    if args.verbose:
        print_section("건별 상세 결과")

    for golden in golden_list:
        r = evaluate_one(golden, data_root, args.verbose)
        if r:
            results.append(r)

    if not results:
        print("[ERR] 평가 가능한 항목이 없습니다.")
        return

    # ── ① 총금액 오차율 집계 ─────────────────────────
    print_section("① 총금액 오차율")
    cost_errors = [r["cost_error_%"] for r in results if r["cost_error_%"] is not None]

    if cost_errors:
        avg_error = sum(cost_errors) / len(cost_errors)
        within_5  = sum(1 for e in cost_errors if e <= 5.0)
        within_10 = sum(1 for e in cost_errors if e <= 10.0)
        mark = "✅" if avg_error <= COST_ERROR_TARGET else "❌"
        print(f"  측정 건수:   {len(cost_errors)}건")
        print(f"  ±5%  이내:  {within_5}/{len(cost_errors)}건")
        print(f"  ±10% 이내:  {within_10}/{len(cost_errors)}건")
        print(f"\n  {mark} 평균 오차율: {avg_error:.1f}% (목표 ≤{COST_ERROR_TARGET}%)")

        worst = sorted(
            [r for r in results if r["cost_error_%"] is not None],
            key=lambda r: r["cost_error_%"], reverse=True
        )
        if worst:
            print(f"\n  [오차 상위 5건]")
            for r in worst[:5]:
                print(f"    {r['article_id']}: 실제 {r['actual_cost']:,} / 파싱 {r['parsed_cost']:,} → {r['cost_error_%']}%")
    else:
        print("  측정 가능한 건수 없음")

    # ── ② 카테고리 소계 오차율 집계 ─────────────────
    print_section("② 카테고리별 소계 오차율")
    cat_errors   = [r["avg_cat_error_%"] for r in results if r["avg_cat_error_%"] is not None]
    missing_rates = [r["missing_rate_%"] for r in results if r["missing_rate_%"] is not None]

    if cat_errors:
        avg_cat = sum(cat_errors) / len(cat_errors)
        mark = "✅" if avg_cat <= CATEGORY_ERROR_TARGET else "❌"
        print(f"  측정 건수:       {len(cat_errors)}건")
        print(f"  {mark} 평균 소계 오차율: {avg_cat:.1f}% (목표 ≤{CATEGORY_ERROR_TARGET}%)")

        if missing_rates:
            avg_miss = sum(missing_rates) / len(missing_rates)
            miss_mark = "✅" if avg_miss == 0 else "❌"
            print(f"  {miss_mark} 평균 카테고리 누락률: {avg_miss:.1f}%")

        # 소계 오차 큰 건
        worst_cat = sorted(
            [r for r in results if r["avg_cat_error_%"] is not None],
            key=lambda r: r["avg_cat_error_%"], reverse=True
        )
        if worst_cat[:3]:
            print(f"\n  [소계 오차 상위 3건]")
            for r in worst_cat[:3]:
                miss = r["missing_cats"]
                print(f"    {r['article_id']}: 평균 {r['avg_cat_error_%']}%"
                      + (f" | 누락: {miss}" if miss else ""))
    else:
        print("  카테고리별_소계가 없음 — golden_vision.json에 추가 필요")

    # ── 전체 요약 ────────────────────────────────────
    print_section("전체 요약")
    print(f"  골든셋 총:  {len(golden_list)}개")
    print(f"  평가 완료:  {len(results)}개")
    skipped = len(golden_list) - len(results)
    if skipped:
        print(f"  스킵:       {skipped}개 (미파싱 또는 JSON 없음)")
    print()


if __name__ == "__main__":
    main()
