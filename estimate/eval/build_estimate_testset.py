"""
build_estimate_testset.py — 골든셋 기반 가견적 평가 테스트셋 생성

실행:
    python build_estimate_testset.py              # dry-run (변경 없이 목록만 확인)
    python build_estimate_testset.py --apply      # ChromaDB에서 제거 + testset 저장
    python build_estimate_testset.py --restore    # 제거했던 케이스 ChromaDB에 복원

동작:
    golden_vision.json(실제 총금액 레이블)에 있는 article_id를
    ChromaDB에서 조회해 user_input을 재구성하고 hold-out 후 저장.
    테스트 케이스가 ChromaDB에 없으면 generate() 검색 시 자기 자신을 참조하지 않으므로
    data leakage 없는 순수 성능 측정이 가능.
"""

import sys
import json
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from chroma_client import get_chroma_client
from chromadb.utils import embedding_functions

COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
GOLDEN_FILE     = pathlib.Path(__file__).parent.parent / "docs" / "eval_data" / "golden_vision.json"
OUTPUT_FILE     = pathlib.Path(__file__).parent.parent / "docs" / "eval_data" / "estimate_testset.json"

REGION_TO_USER = {
    "서울": "서울",
    "경기": "수도권",
    "인천": "수도권",
}

HAS_TO_공종 = {
    "has_도배": "도배",
    "has_바닥": "마루",
    "has_욕실": "욕실",
    "has_가구": "주방",
    "has_전기": "전기/조명",
    "has_창호": "창호",
}


def meta_to_user_input(meta: dict) -> dict:
    지역_raw = meta.get("region", "기타")
    지역 = REGION_TO_USER.get(지역_raw, "지방")

    # has_* 기반 공종 + 전체 리모델링 기본 공종(목공/도장/철거)
    기본_공종 = ["철거", "목공", "도장"]
    has_공종 = [
        공종명
        for has_key, 공종명 in HAS_TO_공종.items()
        if meta.get(has_key) == "true"
    ]
    공종 = 기본_공종 + (has_공종 if has_공종 else ["도배"])

    # cost_per_pyeong으로 자재등급 추론
    # 실제 사례의 평당 단가를 기반으로 사용자가 원하는 자재 등급을 역추정
    cpp = int(meta.get("cost_per_pyeong") or 0)
    if cpp > 0 and cpp < 1_500_000:    # 150만원/평 미만 → 일반
        자재등급 = "일반"
    elif cpp > 2_500_000:               # 250만원/평 초과 → 고급
        자재등급 = "고급"
    else:
        자재등급 = "중급"

    return {
        "공종":       공종,
        "시공범위":   "전체",
        "공간유형":   "아파트",
        "평수":       int(meta.get("size_pyeong") or 20),
        "방개수":     3,
        "지역":       지역,
        "건물연식":   "10~20년",
        "자재등급":   자재등급,
        "철거여부":   "있음",
        "층수":       5,
        "엘리베이터": "있음",
        "트럭접근":   "가능",
        "거주중공사": "공실",
        "공사시기":   "미정",
    }


def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return get_chroma_client().get_collection(name=COLLECTION_NAME, embedding_function=ef)


def cmd_build(apply: bool):
    if not GOLDEN_FILE.exists():
        print(f"[ERR] 골든셋 파일 없음: {GOLDEN_FILE}")
        return

    golden_list = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    golden_ids  = [str(g["article_id"]) for g in golden_list]
    golden_cost = {str(g["article_id"]): g["실제총금액"] for g in golden_list}

    print(f"골든셋 항목: {len(golden_list)}개")

    collection = get_collection()
    # 골든셋 article_id를 ChromaDB에서 일괄 조회
    result  = collection.get(ids=golden_ids, include=["metadatas", "documents"])
    found_ids = result["ids"]
    meta_map  = {aid: meta for aid, meta in zip(result["ids"], result["metadatas"])}
    doc_map   = {aid: doc  for aid, doc  in zip(result["ids"], result["documents"])}

    missing = [aid for aid in golden_ids if aid not in meta_map]
    if missing:
        print(f"[WARN] ChromaDB에 없는 항목 {len(missing)}개 (건너뜀): {missing}")

    testset = []
    for aid in found_ids:
        meta = meta_map[aid]
        testset.append({
            "article_id":        aid,
            "actual_total_cost": golden_cost[aid],
            "user_input":        meta_to_user_input(meta),
            "_chroma_document":  doc_map[aid],
            "_chroma_metadata":  meta,
        })

    # 목록 출력
    print(f"\n테스트셋 {len(testset)}개:")
    for t in testset:
        m    = t["_chroma_metadata"]
        공종  = ", ".join(t["user_input"]["공종"])
        cost = t["actual_total_cost"] // 10000
        print(f"  {t['article_id']:8s} | {m.get('region','?'):4s} "
              f"{m.get('size_pyeong','?')}평 | {공종} | 실제 {cost}만원")

    if not apply:
        print(f"\n[dry-run] 변경 없음.")
        print(f"  위 목록이 맞으면 --apply 플래그로 다시 실행하세요.")
        print(f"  --apply 시: ChromaDB에서 {len(testset)}개 삭제 + {OUTPUT_FILE} 저장")
        return

    # ChromaDB에서 삭제 (hold-out)
    collection.delete(ids=[t["article_id"] for t in testset])
    print(f"\nChromaDB에서 {len(testset)}개 삭제. 남은 항목: {collection.count()}개")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"테스트셋 저장: {OUTPUT_FILE}")
    print(f"\n다음 단계: python eval_estimate.py")
    print(f"평가 완료 후 복원: python build_estimate_testset.py --restore")


def cmd_restore():
    if not OUTPUT_FILE.exists():
        print(f"[ERR] 테스트셋 파일 없음: {OUTPUT_FILE}")
        return

    testset = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
    if not testset:
        print("[ERR] 테스트셋이 비어 있음")
        return

    collection = get_collection()
    existing   = set(collection.get(include=[])["ids"])
    to_restore = [t for t in testset if t["article_id"] not in existing]

    if not to_restore:
        print(f"[INFO] {len(testset)}개 모두 이미 ChromaDB에 있음 — 복원 불필요")
        return

    collection.upsert(
        ids       =[t["article_id"]       for t in to_restore],
        documents =[t["_chroma_document"] for t in to_restore],
        metadatas =[t["_chroma_metadata"] for t in to_restore],
    )
    print(f"[OK] {len(to_restore)}개 복원 완료. 현재 DB 항목: {collection.count()}개")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply",   action="store_true",
                        help="ChromaDB에서 실제 삭제 + testset 저장 (기본: dry-run)")
    parser.add_argument("--restore", action="store_true",
                        help="삭제된 케이스를 ChromaDB에 복원")
    args = parser.parse_args()

    if args.restore:
        cmd_restore()
    else:
        cmd_build(apply=args.apply)


if __name__ == "__main__":
    main()
