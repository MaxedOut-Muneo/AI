"""
eval_rag.py — RAG 검색 품질 평가 (Hit@K / MRR / 평수 편차)

실행:
    python eval_rag.py
    python eval_rag.py --top 3

사전 조건:
    split_rag_testset.py --apply 를 먼저 실행해서 rag_testset.json 생성 필요
    (테스트셋 항목이 ChromaDB에 포함된 상태에서 평가)

평가 방식 (관련성 기반):
    정답 기준: 동일 지역 + 평수 ±7평 이내 + 공종 2개 이상 겹침 → 관련 사례
    쿼리 원본 article은 검색 결과에서 제외 (자기 자신 매칭 방지)
    Hit@K: 상위 K개 중 관련 사례가 1건 이상이면 적중
    MRR:   첫 번째 관련 사례가 등장한 순위의 역수 평균

출력 지표:
    Hit@K  — 상위 K개 결과 안에 관련 사례가 있는 비율 (목표: 0.70 이상)
    MRR    — 관련 사례가 처음 등장하는 순위의 역수 평균 (목표: 0.60 이상)
    평균 평수 편차 — 쿼리 평수 vs 검색 결과 평수 차이
"""

import sys
import json
import argparse
import pathlib
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from chroma_client import get_chroma_client

COLLECTION_NAME  = "estimates"
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
TESTSET_FILE     = "./eval_data/rag_testset.json"
DEFAULT_TOP_K    = 5

# 관련성 판정 기준
WORK_TYPES       = ["has_창호", "has_도배", "has_타일", "has_가구",
                    "has_욕실", "has_바닥", "has_전기", "has_조명"]
SIZE_RANGE       = 7   # 평수 허용 오차 (±평)
MIN_WORK_OVERLAP = 2   # 최소 공종 겹침 수


def is_relevant(q_meta: dict, r_meta: dict) -> bool:
    """쿼리와 검색 결과가 관련 사례인지 판정.

    조건 (모두 충족):
      1. 동일 지역
      2. 평수 ±SIZE_RANGE 이내
      3. 공종 MIN_WORK_OVERLAP개 이상 겹침
         (쿼리 공종이 2개 미만이면 겹침 조건 완화)
    """
    # 동일 지역
    if q_meta.get("region") != r_meta.get("region"):
        return False

    # 평수 범위
    q_size = int(q_meta.get("size_pyeong") or 0)
    r_size = int(r_meta.get("size_pyeong") or 0)
    if q_size > 0 and r_size > 0 and abs(q_size - r_size) > SIZE_RANGE:
        return False

    # 공종 겹침 (쿼리 공종 수에 따라 threshold 완화)
    q_works   = {w for w in WORK_TYPES if q_meta.get(w) == "true"}
    r_works   = {w for w in WORK_TYPES if r_meta.get(w) == "true"}
    threshold = min(MIN_WORK_OVERLAP, len(q_works))
    if threshold == 0:
        return True
    return len(q_works & r_works) >= threshold


def evaluate(test_queries: list[dict], top_k: int) -> dict:
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client     = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    hits             = 0
    reciprocal_ranks = []
    size_deviations  = []
    detail_rows      = []

    for q in test_queries:
        article_id = q["메타"]["article_id"]

        # ChromaDB에서 쿼리 원본 메타 조회 (has_* 필드 포함)
        q_data = collection.get(ids=[article_id], include=["metadatas"])
        q_meta = q_data["metadatas"][0] if q_data["metadatas"] else q["메타"]

        # top_k+1 검색 후 자기 자신 제외
        results = collection.query(
            query_texts=[q["query"]],
            n_results=min(top_k + 1, collection.count()),
            include=["metadatas"],
        )
        returned_meta = [
            m for m in results["metadatas"][0]
            if m.get("article_id") != article_id
        ][:top_k]

        # 관련성 판정
        relevant_flags = [is_relevant(q_meta, m) for m in returned_meta]
        hit = any(relevant_flags)
        if hit:
            hits += 1

        # MRR — 첫 번째 관련 사례의 순위
        rr = 0.0
        for rank, flag in enumerate(relevant_flags, 1):
            if flag:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # 평수 편차
        query_size = int(q_meta.get("size_pyeong") or 0)
        if query_size:
            for m in returned_meta:
                result_size = int(m.get("size_pyeong") or 0)
                if result_size:
                    size_deviations.append(abs(query_size - result_size))

        relevant_ids = [m["article_id"] for m, f in zip(returned_meta, relevant_flags) if f]
        detail_rows.append({
            "query_id":     q["query_id"],
            "query":        q["query"],
            "hit":          hit,
            "rr":           round(rr, 4),
            "관련_사례_수":  sum(relevant_flags),
            "첫_관련_순위": (relevant_flags.index(True) + 1) if hit else "-",
            "관련_ids":     relevant_ids[:3],
        })

    n       = len(test_queries)
    hit_atk = hits / n
    mrr     = sum(reciprocal_ranks) / n
    avg_dev = sum(size_deviations) / len(size_deviations) if size_deviations else None

    return {
        f"Hit@{top_k}":   round(hit_atk, 4),
        "MRR":            round(mrr, 4),
        "평균_평수_편차":   round(avg_dev, 1) if avg_dev is not None else "-",
        "테스트_쿼리_수":  n,
        "상세":           detail_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_K,
                        help=f"상위 K개 검색 결과 (기본: {DEFAULT_TOP_K})")
    args = parser.parse_args()

    testset_path = pathlib.Path(TESTSET_FILE)
    if not testset_path.exists():
        print(f"[ERR] 테스트셋 파일 없음: {TESTSET_FILE}")
        print("  먼저 실행: python split_rag_testset.py --apply")
        return

    test_queries = json.loads(testset_path.read_text(encoding="utf-8"))
    print(f"테스트 쿼리 {len(test_queries)}개 로드\n")

    result = evaluate(test_queries, top_k=args.top)

    print("=" * 55)
    print("  RAG 검색 품질 평가 결과")
    print("=" * 55)
    for k, v in result.items():
        if k == "상세":
            continue
        print(f"  {k}: {v}")

    # 목표치 달성 여부
    print("\n  목표치 달성 여부:")
    top_k = args.top
    hit   = result.get(f"Hit@{top_k}", 0)
    mrr   = result.get("MRR", 0)
    print(f"  {'✅' if hit >= 0.70 else '❌'} Hit@{top_k}: {hit} (목표: ≥0.70)")
    print(f"  {'✅' if mrr >= 0.60 else '❌'} MRR:    {mrr} (목표: ≥0.60)")

    # 미적중 사례 출력
    misses = [r for r in result["상세"] if not r["hit"]]
    if misses:
        print(f"\n  [미적중 사례] {len(misses)}건")
        for r in misses[:5]:
            print(f"    {r['query_id']}: {r['query']}")
        if len(misses) > 5:
            print(f"    ... 외 {len(misses) - 5}건")

    # 상세 결과 출력
    print(f"\n  [상세]")
    for r in result["상세"]:
        mark = "✅" if r["hit"] else "❌"
        print(f"    {mark} {r['query_id']} | 관련 {r['관련_사례_수']}건 | "
              f"첫 순위 {r['첫_관련_순위']} | RR={r['rr']}")

    print()


if __name__ == "__main__":
    main()
