from db import get_sync_collection_from_db


def search_estimate_cases_for_reference(question: str) -> str:
    """
    견적서 사례를 MongoDB에서 직접 조회해서
    '견적서에 자주 포함되는 항목' 등을 추출하는 함수
    """

    col = get_sync_collection_from_db("estimate_db", "estimate_cases")

    # parsed_estimate가 있는 데이터 일부만 가져오기
    docs = list(col.find(
        {"parsed_estimate": {"$exists": True}},
        {"parsed_estimate.line_items": 1}
    ).limit(30))

    if not docs:
        return "견적서 사례 데이터가 없습니다."

    category_count = {}
    sample_items = []

    for doc in docs:
        parsed = doc.get("parsed_estimate", {})
        items = parsed.get("line_items", [])

        for item in items:
            category = item.get("category") or item.get("description", "")

            if not category:
                continue

            category_count[category] = category_count.get(category, 0) + 1

            if len(sample_items) < 10:
                sample_items.append(category)

    # 많이 등장한 항목 상위 7개
    top_items = sorted(category_count.items(), key=lambda x: x[1], reverse=True)[:7]

    result = "견적서 사례에서 자주 포함되는 항목:\n"

    for name, count in top_items:
        result += f"- {name}\n"

    if sample_items:
        result += "\n예시 항목:\n"
        for item in sample_items[:5]:
            result += f"- {item}\n"

    return result