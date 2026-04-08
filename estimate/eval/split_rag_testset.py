"""
split_rag_testset.py — RAG 평가용 hold-out 테스트셋 분리

실행:
    python split_rag_testset.py              # dry-run (변경 없이 선택 목록만 확인)
    python split_rag_testset.py --apply      # ChromaDB에서 제거 + rag_testset.json 저장

문제 배경:
    build_rag.py를 이미 실행했다면 전체 데이터가 ChromaDB에 들어가 있음.
    이 상태에서 Hit@K / MRR을 측정하면 DB 안의 데이터로 DB를 검색하는 것과 같아
    지표가 부풀려짐 (data leakage).

해결:
    1. ChromaDB에서 랜덤 N개 항목을 선택해 DB에서 제거
    2. rag_testset.json으로 저장 (쿼리 템플릿 자동 생성)
    3. eval_rag.py에서 이 파일을 입력으로 Hit@K / MRR 계산

주의:
    --apply는 되돌릴 수 없음. dry-run으로 먼저 확인하세요.
"""

import sys
import json
import random
import argparse
import pathlib
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from chroma_client import get_chroma_client

COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
TESTSET_SIZE    = 20
TESTSET_OUTPUT  = "./eval_data/rag_testset.json"
SEED            = 42


def build_query(meta: dict) -> str:
    """메타데이터에서 RAG 검색 쿼리 텍스트 생성."""
    size   = meta.get("size_pyeong") or "?"
    region = meta.get("region", "")

    has_map = {
        "창호": meta.get("has_창호") == "true",
        "도배": meta.get("has_도배") == "true",
        "타일": meta.get("has_타일") == "true",
        "가구": meta.get("has_가구") == "true",
        "욕실": meta.get("has_욕실") == "true",
        "바닥": meta.get("has_바닥") == "true",
        "전기": meta.get("has_전기") == "true",
        "조명": meta.get("has_조명") == "true",
    }
    works = [k for k, v in has_map.items() if v]

    parts = [f"{size}평"]
    if region:
        parts.append(region)
    if works:
        parts.append(" ".join(works))
    parts.append("리모델링")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="ChromaDB에서 실제 제거 및 testset 저장 (기본: dry-run)")
    parser.add_argument("--size", type=int, default=TESTSET_SIZE,
                        help=f"테스트셋 크기 (기본: {TESTSET_SIZE})")
    args = parser.parse_args()

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client     = get_chroma_client()
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    total = collection.count()
    print(f"ChromaDB 전체 항목: {total}개")

    if total < args.size + 10:
        print(f"[WARN] DB 항목이 {args.size + 10}개 미만 — 테스트셋 분리 시 학습 데이터가 너무 적어짐")

    if total < args.size:
        print(f"[ERR] DB 항목({total})이 테스트셋 크기({args.size})보다 적음")
        return

    # 전체 항목 조회
    all_data = collection.get(include=["metadatas"])
    all_ids  = all_data["ids"]
    all_meta = all_data["metadatas"]

    # 랜덤 샘플링 (재현 가능하도록 seed 고정)
    random.seed(SEED)
    sampled_idx = random.sample(range(len(all_ids)), args.size)

    testset = []
    for idx in sampled_idx:
        aid  = all_ids[idx]
        meta = all_meta[idx]
        testset.append({
            "query_id":          f"q_{aid}",
            "query":             build_query(meta),
            "정답_article_ids":   [aid],
            "메타": {
                "article_id":  aid,
                "region":      meta.get("region", ""),
                "size_pyeong": meta.get("size_pyeong", 0),
            },
        })

    # dry-run 출력
    print(f"\n선택된 테스트셋 {args.size}개:")
    for t in testset:
        m = t["메타"]
        print(f"  {m['article_id']:8s} | {m['region']:4s} {m['size_pyeong']}평 | {t['query']}")

    if not args.apply:
        print(f"\n[dry-run] 변경 없음.")
        print(f"  위 목록이 맞으면 --apply 플래그로 다시 실행하세요.")
        print(f"  --apply 시: ChromaDB에서 {args.size}개 제거 + {TESTSET_OUTPUT} 저장")
        return

    # ── 실제 적용 ──────────────────────────────────────
    remove_ids = [t["정답_article_ids"][0] for t in testset]
    collection.delete(ids=remove_ids)
    print(f"\nChromaDB에서 {len(remove_ids)}개 제거. 남은 항목: {collection.count()}개")

    pathlib.Path(TESTSET_OUTPUT).write_text(
        json.dumps(testset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"테스트셋 저장: {TESTSET_OUTPUT}")
    print(f"\n다음 단계: python eval_rag.py")


if __name__ == "__main__":
    main()
