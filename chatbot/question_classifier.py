def classify_question(question: str) -> dict:
    q = question.lower()

    estimate_keywords = [
        "견적서", "견적 항목", "견적서 양식", "내역서", "누락",
        "포함 항목", "별도 비용", "공사 범위", "자재명", "규격"
    ]

    legal_keywords = [
        "계약", "계약서", "위약금", "계약해제", "해제",
        "추가공사", "추가 비용", "공사변경", "잔금", "계약금",
        "중도금", "지연배상", "공사지연"
    ]

    defect_keywords = [
        "하자", "누수", "균열", "파손", "들뜸", "탈락",
        "보수", "무상수리", "as", "곰팡이", "결로", "오염",
        "타일", "창호", "배관", "바닥재"
    ]

    interior_keywords = [
        "인테리어", "리모델링", "도배", "장판", "마루", "타일",
        "욕실", "주방", "창호", "싱크대", "철거", "시공", "공사",
        "벽지", "실크", "합지", "몰딩", "필름", "도장"
    ]

    use_estimate_cases = any(k in q for k in estimate_keywords)
    use_legal_docs = any(k in q for k in legal_keywords)
    use_defect_docs = any(k in q for k in defect_keywords)

    is_interior = (
        use_estimate_cases
        or use_legal_docs
        or use_defect_docs
        or any(k in q for k in interior_keywords)
    )

    return {
        "use_estimate_cases": use_estimate_cases,
        "use_legal_docs": use_legal_docs,
        "use_defect_docs": use_defect_docs,
        "is_interior": is_interior
    }