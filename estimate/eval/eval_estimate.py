"""
eval_estimate.py — 가견적 정확도 평가

실행:
    python eval_estimate.py

평가 지표:
    범위_포함률: actual_cost가 (최소, 최대) 구간 안에 드는 비율  (목표 ≥ 0.70)
    MAPE:        |actual - 중간| / actual * 100 평균              (목표 ≤ 20%)

사전 조건:
    python build_estimate_testset.py --apply  ← hold-out 먼저 실행
"""

import sys
import json
import pathlib
import statistics

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from estimate_engine import EstimateEngine

COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
TESTSET_FILE    = (
    pathlib.Path(__file__).parent.parent / "docs" / "eval_data" / "estimate_testset.json"
)


def check_leakage(testset_ids: list[str]) -> list[str]:
    """테스트 케이스 중 ChromaDB에 남아있는 항목 목록 반환."""
    try:
        from chroma_client import get_chroma_client
        from chromadb.utils import embedding_functions
        ef     = embedding_functions.SentenceTransformerEmbeddingFunction(EMBED_MODEL)
        client = get_chroma_client()
        col    = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
        existing = set(col.get(include=[])["ids"])
        return [aid for aid in testset_ids if aid in existing]
    except Exception:
        return []


def evaluate(testset: list[dict]) -> dict:
    engine   = EstimateEngine()
    in_range = 0
    ape_list = []
    detail   = []
    errors   = 0

    for item in testset:
        article_id = item["article_id"]
        actual     = item["actual_total_cost"]
        user_input = item["user_input"]

        try:
            result = engine.generate(user_input)
        except Exception as e:
            errors += 1
            detail.append({"article_id": article_id, "actual": actual, "error": str(e)})
            continue

        if "error" in result:
            errors += 1
            detail.append({"article_id": article_id, "actual": actual, "error": result["error"]})
            continue

        r   = result["총_견적_범위"]
        lo  = r["최소"]
        hi  = r["최대"]
        mid = r["중간"]

        hit = lo <= actual <= hi
        ape = abs(actual - mid) / actual * 100

        if hit:
            in_range += 1
        ape_list.append(ape)

        detail.append({
            "article_id": article_id,
            "actual":     actual,
            "estimate":   {"최소": lo, "중간": mid, "최대": hi},
            "in_range":   hit,
            "ape_%":      round(ape, 1),
        })

    valid         = len(testset) - errors
    coverage_rate = round(in_range / valid, 4) if valid else 0.0
    mape          = round(statistics.mean(ape_list), 1) if ape_list else None

    return {
        "testset_count":  len(testset),
        "valid_count":    valid,
        "error_count":    errors,
        "in_range_count": in_range,
        "coverage_rate":  coverage_rate,
        "mape_%":         mape,
        "detail":         detail,
    }


def run_estimate_eval() -> dict:
    """eval_all.py에서 호출하는 진입점."""
    if not TESTSET_FILE.exists():
        return {
            "status":  "skipped",
            "reason":  "estimate_testset.json 없음 — build_estimate_testset.py --apply 먼저 실행",
            "metrics": {},
            "goals":   {},
        }

    testset = json.loads(TESTSET_FILE.read_text(encoding="utf-8"))
    if not testset:
        return {
            "status":  "skipped",
            "reason":  "테스트셋이 비어 있음",
            "metrics": {},
            "goals":   {},
        }

    # leakage 감지
    leaked = check_leakage([t["article_id"] for t in testset])
    if leaked:
        return {
            "status":  "skipped",
            "reason":  (
                f"테스트 케이스 {len(leaked)}개가 ChromaDB에 남아있어 정확한 평가 불가 "
                f"(data leakage). build_estimate_testset.py --apply 재실행 필요."
            ),
            "metrics": {},
            "goals":   {},
        }

    result        = evaluate(testset)
    coverage_rate = result["coverage_rate"]
    mape          = result["mape_%"]

    metrics = {
        "testset_count":  result["testset_count"],
        "valid_count":    result["valid_count"],
        "error_count":    result["error_count"],
        "in_range_count": result["in_range_count"],
        "coverage_rate":  coverage_rate,
        "mape_%":         mape,
    }
    goals = {"coverage_rate ≥ 0.70": coverage_rate >= 0.70}
    if mape is not None:
        goals["mape ≤ 20%"] = mape <= 20.0

    return {
        "status":  "ok",
        "metrics": metrics,
        "goals":   goals,
        "detail":  result["detail"],
    }


def main():
    if not TESTSET_FILE.exists():
        print(f"[ERR] 테스트셋 없음: {TESTSET_FILE}")
        print("  → python build_estimate_testset.py --apply 먼저 실행")
        return

    testset = json.loads(TESTSET_FILE.read_text(encoding="utf-8"))
    print(f"테스트셋 {len(testset)}건 로드\n")

    # leakage 감지
    print("leakage 확인 중...")
    leaked = check_leakage([t["article_id"] for t in testset])
    if leaked:
        print(f"[ERR] 테스트 케이스 {len(leaked)}개가 아직 ChromaDB에 있음 → 결과를 신뢰할 수 없음")
        print(f"  → python build_estimate_testset.py --apply 로 hold-out 먼저 실행")
        print(f"  leakage 대상: {leaked[:5]}{'...' if len(leaked) > 5 else ''}")
        return
    print("leakage 없음 — 정상 평가 가능\n")

    result        = evaluate(testset)
    coverage_rate = result["coverage_rate"]
    mape          = result["mape_%"]

    print("=" * 55)
    print("  PHASE 4 — 가견적 정확도 평가 결과")
    print("=" * 55)
    print(f"  테스트셋:    {result['testset_count']}건  "
          f"(유효 {result['valid_count']}건 / 오류 {result['error_count']}건)")
    print(f"  범위 포함률: {coverage_rate:.1%}  "
          f"({result['in_range_count']}/{result['valid_count']})  (목표 ≥ 70%)")
    if mape is not None:
        print(f"  MAPE:        {mape:.1f}%  (목표 ≤ 20%)")
    print()
    print(f"  {'✅' if coverage_rate >= 0.70 else '❌'} coverage_rate ≥ 0.70")
    if mape is not None:
        print(f"  {'✅' if mape <= 20.0 else '❌'} MAPE ≤ 20%")

    misses = [d for d in result["detail"] if "in_range" in d and not d["in_range"]]
    if misses:
        print(f"\n  [범위 밖 사례] {len(misses)}건")
        for d in misses[:5]:
            lo  = d["estimate"]["최소"] // 10000
            mid = d["estimate"]["중간"] // 10000
            hi  = d["estimate"]["최대"] // 10000
            act = d["actual"] // 10000
            print(f"    {d['article_id']}: 실제 {act}만 | "
                  f"추정 {lo}~{hi}만 (중간 {mid}만) | 오차 {d['ape_%']}%")
        if len(misses) > 5:
            print(f"    ... 외 {len(misses) - 5}건")

    print(f"\n평가 완료 후 복원: python build_estimate_testset.py --restore")
    print()


if __name__ == "__main__":
    main()
