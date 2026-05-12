"""
build_rag.py — 파싱된 견적 데이터를 ChromaDB 벡터 DB에 삽입

실행:
    python build_rag.py

동작:
    - estimate_data/{지역}/{article_id}/{article_id}.json 순회
    - parsed_estimate 필드가 있는 항목만 ChromaDB에 삽입
    - 임베딩 모델: paraphrase-multilingual-MiniLM-L12-v2 (한국어 지원)
    - 컬렉션명: estimates
    - article_id 기준 중복 삽입 방지

결과:
    Chroma Cloud의 estimates 컬렉션에 데이터 삽입
"""

import json
import pathlib
import chromadb
from chromadb.utils import embedding_functions
from chroma_client import get_chroma_client

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════

DATA_DIR        = "./estimate_data"
COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"

# has_* 판단 키워드
HAS_KEYWORDS: dict[str, list[str]] = {
    "has_창호": ["창호", "샷시", "새시", "현관문", "도어", "발코니창"],
    "has_도배": ["도배", "벽지", "합지", "실크"],
    "has_타일": ["타일", "도기", "줄눈"],
    "has_가구": ["가구", "붙박이", "신발장", "싱크대", "수납장"],
    "has_욕실": ["욕실", "화장실", "욕조", "세면대", "변기"],
    "has_바닥": ["바닥", "마루", "장판", "데코타일", "강마루", "강화마루"],
    "has_전기": ["전기", "배선", "콘센트", "인덕션", "분전반"],
    "has_조명": ["조명", "다운라이트", "간접등", "LED", "등기구"],
}


# ══════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════

def build_check_text(data: dict) -> str:
    """has_* 판단용 텍스트: request_body_text + line_items 카테고리·항목명 합산."""
    parts = [(data.get("request_body_text") or "")]
    for item in (data.get("parsed_estimate") or {}).get("line_items", []):
        if item.get("category"):
            parts.append(item["category"])
        if item.get("item"):
            parts.append(item["item"])
    return " ".join(parts)


def check_has_keywords(text: str) -> dict[str, str]:
    """텍스트에서 키워드 존재 여부 판단. ChromaDB는 bool 미지원 → "true"/"false"."""
    return {
        key: "true" if any(kw in text for kw in keywords) else "false"
        for key, keywords in HAS_KEYWORDS.items()
    }


_HAS_TO_WORK = {
    "has_창호": "창호", "has_도배": "도배", "has_타일": "타일",
    "has_가구": "가구", "has_욕실": "욕실", "has_바닥": "바닥",
    "has_전기": "전기", "has_조명": "조명",
}


def build_document(data: dict) -> str:
    """임베딩 대상 텍스트 구성. 지역·공종 정보를 명시적으로 포함."""
    size         = data.get("size_pyeong", "?")
    region       = data.get("region", "")
    request_text = (data.get("request_body_text") or "").strip()

    check_text = build_check_text(data)
    has_fields = check_has_keywords(check_text)
    works      = [name for key, name in _HAS_TO_WORK.items() if has_fields.get(key) == "true"]

    header = " ".join(filter(None, [
        f"{size}평",
        region,
        " ".join(works),
        "리모델링",
    ]))
    if request_text:
        return f"{header}\n요청공사: {request_text}"
    return header


# 정규화 카테고리 → DB cost_* 키 매핑
CATEGORY_NORM: dict[str, str] = {
    "철거공사": "철거", "철거": "철거",
    "착공공사": "철거",                         # 착공 단계 철거 포함
    "목공사": "목공", "목공": "목공",
    "도배공사": "도배", "도배": "도배",
    "바닥공사": "바닥", "바닥": "바닥", "마루공사": "바닥",
    "바닥장공사": "바닥",                        # OCR 변형 (바닥장→바닥)
    "마루": "바닥",
    "타일공사": "타일", "타일": "타일",
    "욕실공사": "욕실", "설비공사": "설비", "수전/위생공사": "설비",
    "전기공사": "전기", "전기": "전기", "조명공사": "전기",
    "가구공사": "가구", "주방가구공사": "가구", "싱크공사": "가구",
    "창호공사": "창호", "샷시공사": "창호",
    "현호공사": "창호",                          # OCR 오인식 (현호→창호)
    "도어공사": "창호",                          # 현관문·방문 도어 포함
    "샷시": "창호", "새시": "창호", "새시공사": "창호",
    "필름공사": "필름", "시트공사": "필름",      # 필름·시트는 동일 공종
    "도장공사": "도장",
    "폐기물처리": "철거",
}


def build_category_costs(line_items: list[dict], total_cost: int) -> dict[str, int]:
    """
    line_items에서 정규화 카테고리별 금액 합산.
    대분류/소분류 중복 집계를 피하기 위해
    line_items 합계가 total_cost 1.3배 이상이면 total_cost 비율로 재산정.
    """
    raw: dict[str, int] = {}
    for item in line_items:
        cat_raw = item.get("category", "")
        norm    = CATEGORY_NORM.get(cat_raw)
        if not norm:
            continue
        amount = int(item.get("amount") or 0)
        if amount > 0:
            raw[norm] = raw.get(norm, 0) + amount

    line_sum = sum(raw.values())
    # 중복 집계 감지 → 비율로 재산정
    if total_cost > 0 and line_sum > total_cost * 1.3:
        ratio = total_cost / line_sum
        raw = {k: int(v * ratio) for k, v in raw.items()}

    return {f"cost_{k}": v for k, v in raw.items()}


def build_metadata(data: dict) -> dict:
    """ChromaDB metadata 딕셔너리 구성."""
    article_id   = str(data.get("article_id", ""))
    size_pyeong  = int(data.get("size_pyeong") or 0)
    region       = data.get("region", "기타")
    request_text = (data.get("request_body_text") or "")

    parsed      = data.get("parsed_estimate") or {}
    total_cost  = int(parsed.get("total_cost") or 0)
    line_items  = parsed.get("line_items", [])
    cost_per_pyeong = int(total_cost / size_pyeong) if size_pyeong > 0 else 0

    local_images = data.get("local_images") or []
    image_path   = local_images[0] if local_images else ""

    check_text      = build_check_text(data)
    has_fields      = check_has_keywords(check_text)
    category_costs  = build_category_costs(line_items, total_cost)

    return {
        "article_id":      article_id,
        "region":          region,
        "size_pyeong":     size_pyeong,
        "total_cost":      total_cost,
        "cost_per_pyeong": cost_per_pyeong,
        **has_fields,
        **category_costs,          # cost_철거, cost_도배, cost_타일, ...
        "image_path":      image_path,
    }


# ══════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════

def load_articles() -> list[dict]:
    """estimate_data/{지역}/{article_id}.json 전부 로드."""
    articles = []
    data_root = pathlib.Path(DATA_DIR)
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
                articles.append(data)
            except Exception as e:
                print(f"  [WARN]  {json_path} 로드 실패: {e}")
    return articles


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main():
    # ChromaDB 클라이언트 + 컬렉션
    chroma_client = get_chroma_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    # 기존 ID 목록 (중복 방지)
    existing = collection.get(include=[])
    existing_ids = set(existing["ids"])
    print(f"[INFO] 기존 DB 항목: {len(existing_ids)}개\n")

    articles = load_articles()
    print(f"[DIR] JSON 파일 로드: {len(articles)}개\n")

    added = skipped_no_parse = 0

    for data in articles:
        article_id = str(data.get("article_id", ""))
        if not article_id:
            continue

        # parsed_estimate 없으면 skip
        if not data.get("parsed_estimate"):
            print(f"  [SKIP]  파싱 데이터 없음: {article_id}")
            skipped_no_parse += 1
            continue

        document = build_document(data)
        metadata = build_metadata(data)

        collection.upsert(
            ids=[article_id],
            documents=[document],
            metadatas=[metadata],
        )

        size   = metadata["size_pyeong"]
        total  = metadata["total_cost"]
        items  = len(data["parsed_estimate"].get("line_items", []))
        action = "갱신" if article_id in existing_ids else "추가"
        print(f"  [OK] {action}: {article_id} | {size}평 | {total:,}원 | 항목 {items}개")
        added += 1

    print()
    print("=" * 50)
    print(f"[OK] 완료: {added}개 upsert (신규+갱신)")
    print(f"[SKIP]  파싱 없음 skip: {skipped_no_parse}개")
    print(f"[DIR] DB: Chroma Cloud ({COLLECTION_NAME})")
    print("=" * 50)


if __name__ == "__main__":
    main()
