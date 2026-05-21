"""
estimate_engine.py — 사용자 입력(STEP 1~3) → RAG 기반 가견적 생성

사용법:
    python estimate_engine.py          # 샘플 입력으로 테스트
    python estimate_engine.py --input user_input.json

또는 import:
    from estimate_engine import EstimateEngine
    engine = EstimateEngine()
    result = engine.generate(user_input)

user_input 구조:
    {
      "공종":       ["도배", "마루", "욕실"],
      # 도배/장판/마루/주방/욕실/전기·조명/목공/도장/설비/창호/필름/마감·공과잡비
      "시공범위":   "전체",                    # 전체/부분 (기본값: 부분)
      "공간유형":   "아파트",
      "평수":       30,
      "방개수":     3,
      "지역":       "서울",                    # 서울/수도권/지방

      "건물연식":   "20년이상",                # 신축(3년이하)/10년이하/10~20년/20년이상
      "자재등급":   "중급",                    # 일반/중급/고급
      "철거여부":   "있음",                    # 있음/없음/모름
      "층수":       10,
      "엘리베이터": "있음",                    # 있음/없음
      "트럭접근":   "가능",                    # 가능/불가/모름
      "거주중공사": "공실",                    # 거주중/공실
      "공사시기":   "1~3개월",               # 1개월이내/1~3개월/3개월이후/미정

      "도배": {"범위": "전체", "도배지종류": "실크벽지", "초배포함": "있음"},
      "마루": {"자재종류": "강마루", "범위": "전체", "철거여부": "있음"},
      "욕실": {"개수": 2, "크기": "중형", "도기교체": "있음", "방수포함": "있음",
               "욕조샤워부스": "없음", "타일등급": "중급"},
      "주방": {"싱크대교체": "있음", "형태": "I자형", "길이": "2~3m"},
    }
"""

import json
import argparse
import math
import pathlib
import re
import statistics
from collections import defaultdict

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════

COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K              = 15   # 유사 사례 최대 조회 수
SIZE_RANGE         = 7    # 평수 ±7평 필터
MAX_SPEC_ITEMS     = 12   # 공종별 명세 최대 항목 수 (ancillary 제외)
SPEC_RATIO         = 0.15 # 비정규화 항목 등장 비율 threshold (전체 사례 수 × 비율)
SCOPE_COVERAGE_MIN = 0.40 # 요청 공종 비용 합계 / 사례 총 비용 최소 비율 (전체 시공용)

# 치수 표기 기호(×/✕/*/+) 정규화: "540*540" → "540×540"
_DIM_SEP_RE = re.compile(r'(\d+)[×✕\*\+](\d+)')

# 사용자 지역 → DB region 매핑
REGION_MAP = {
    "서울":  ["서울"],
    "수도권": ["경기", "인천"],
    "지방":  ["부산", "대구", "울산", "광주", "대전", "세종",
               "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "기타"],
}

# 사용자 공종 → DB has_* 플래그
# 목공/도장/설비는 has_* 플래그 없음 — 필터 없이 비용만 추출
공종_TO_HAS = {
    "도배":       "has_도배",
    "장판":       "has_바닥",
    "마루":       "has_바닥",
    "욕실":       "has_욕실",
    "주방":       "has_가구",
    "가구":       "has_가구",   # 붙박이장·신발장 등
    "전기/조명":  "has_전기",
    "창호":       "has_창호",
}

# 사용자 공종 → DB cost_* 키
# 욕실: cost_설비 포함 (욕실 단독 선택 시) — 설비 독립 선택 시 중복 주의
공종_TO_COST = {
    "도배":       ["cost_도배"],
    "장판":       ["cost_바닥"],
    "마루":       ["cost_바닥"],
    "욕실":       ["cost_욕실", "cost_설비", "cost_타일"],
    "주방":       ["cost_가구"],
    "가구":       ["cost_가구"],   # 붙박이장·신발장 — calc_공종_factors에서 비율 분배
    "철거":       ["cost_철거"],
    "전기/조명":  ["cost_전기"],
    "목공":       ["cost_목공"],
    "도장":       ["cost_도장"],
    "설비":       ["cost_설비"],
    "창호":       ["cost_창호"],
    "필름":       ["cost_필름"],
    # 마감/공과잡비: DB 키 없음 — calc_factors에서 비율로 산정
}

# 사용자 공종 → 파싱 데이터 category 이름 매핑
공종_TO_CATEGORY = {
    "도배":       ["도배공사"],
    "마루":       ["바닥공사"],
    "장판":       ["바닥공사"],
    "욕실":       ["타일공사", "수전공사", "도기공사"],
    "주방":       ["가구공사"],
    "가구":       ["가구공사"],   # 붙박이장·신발장 위주 — EXCLUDE_DESC로 명세 분리
    "전기/조명":  ["전기공사", "조명공사"],
    "목공":       ["목공사", "목공"],
    "도장":       ["도장공사"],
    "설비":       ["설비공사", "수전/위생공사"],
    "철거":       ["철거공사"],
    "창호":       ["창호공사"],
    "필름":       ["필름공사"],
}

# 공종별 제외 description
# 마루↔장판 혼재 방지 / 주방↔가구 line_items 명세 분리
공종_EXCLUDE_DESC: dict[str, set] = {
    "마루":  {"장판"},
    "장판":  {"강마루"},
    "주방":  {"붙박이장", "신발장", "수납장", "현관장", "키큰장"},  # 주방 = 싱크대·후드 위주
    "가구":  {"싱크대", "냉장고장", "후드", "주방수전"},            # 가구 = 붙박이·수납 위주
    "욕실":  {"거실바닥타일"},                                      # 욕실 공사와 무관한 타일 제외
}

# description 정규화 테이블: category → [(포함 키워드 목록, 대표 이름)]
# 매칭 순서가 우선순위 — 위에 있을수록 먼저 매칭
NORM_MAP = {
    "도배공사": [
        (["LX", "KCC", "자연애", "장판", "강마루", "마루"],  None),  # 바닥재 오기재 제외
        (["초배"],                          "초배지"),
        (["실크"],                          "실크벽지"),
        (["합지"],                          "합지벽지"),
        (["퍼티", "바탕면처리", "벽지제거"], "바탕면처리·퍼티"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "바닥공사": [
        (["강마루", "강화마루", "원목마루", "마루재", "합판마루"], "강마루"),
        (["자연애", "장판", "LX", "KCC"],    "장판"),
        (["합판"],                           "합판"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "타일공사": [
        (["주방벽타일", "주방타일"],          None),       # 타일공사 내 주방 관련 오기재 제외
        (["현관"],                           "현관타일"),
        (["발코니"],                         "발코니타일"),
        (["욕실 벽", "욕실벽"],              "욕실벽타일"),
        (["욕실 바닥", "욕실바닥"],          "욕실바닥타일"),
        (["바닥타일"],                       "거실바닥타일"),
        (["코너비트", "코너"],               "코너비트"),
        (["줄눈"],                           "줄눈"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "수전공사": [
        (["세면기수전", "세면대수전"],        "세면기수전"),
        (["샤워기"],                         "샤워기"),
        (["슬라이딩장"],                     "슬라이딩장"),
        (["환풍기"],                         "욕실환풍기"),
        (["휴지걸이", "수건걸이", "액세서리", "유리코너"], "욕실액세서리"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "도기공사": [
        (["세면도기", "세면기", "반다리"],   "세면기"),
        (["양변도기", "양변기"],             "양변기"),
        (["자바라"],                         "자바라트랩"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비", "운반비"],               "운송비"),
    ],
    "가구공사": [
        (["싱크대", "싱크", "사재싱크"],     "싱크대"),
        (["냉장고장", "냉장고 장"],          "냉장고장"),
        (["붙박이", "붙박이장"],             "붙박이장"),
        (["신발장"],                         "신발장"),
        (["수납장"],                         "수납장"),
        (["수전", "원홀"],                   "주방수전"),
        (["현관장"],                         "현관장"),
        (["키큰장"],                         "키큰장"),
        (["후드"],                           "후드"),
        (["부자재"],                         "부자재"),
        (["인건비", "시공인건비", "안건비"], "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "철거공사": [
        (["폐기물", "건축폐기물"],           "폐기물처리"),
        (["사다리차", "장비대"],             "사다리차"),
        (["마루철거"],                       "마루철거"),
        (["타일철거"],                       "타일철거"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "전기공사": [
        (["분전반"],                         "분전반"),
        (["인덕션"],                         "인덕션"),
        (["콘센트"],                         "콘센트"),
        (["스위치"],                         "스위치"),
        (["배선", "배관"],                   "배선/배관"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "조명공사": [
        (["다운라이트"],                     "다운라이트"),
        (["간접등", "간접조명"],             "간접조명"),
        (["LED"],                            "LED등"),
        (["등기구"],                         "등기구"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "목공사": [
        (["걸레받이"],                       "걸레받이"),
        (["문선"],                           "문선"),
        (["문틀"],                           "문틀"),
        (["천장"],                           "천장목공"),
        (["파티션"],                         "파티션"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "도장공사": [
        (["탄성", "단성"],                   "탄성코트"),
        (["세라믹"],                         "세라믹코트"),
        (["페인트", "페인"],                 "페인트"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "창호공사": [
        (["현관문"],                         "현관문"),
        (["발코니창", "발코니"],             "발코니창"),
        (["샷시", "새시"],                   "샷시/새시"),
        (["방화문"],                         "방화문"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "필름공사": [
        (["문짝", "문 필름"],                "문짝필름"),
        (["몰딩"],                           "몰딩필름"),
        (["샷시", "새시"],                   "샷시필름"),
        (["가구"],                           "가구필름"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "설비공사": [
        (["배관"],                           "배관"),
        (["보일러"],                         "보일러"),
        (["난방"],                           "난방배관"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
}

# 사용자에게 보여줄 필요 없는 내부 항목 (식대, 주차비 등)
_SKIP_KEYWORDS = ["식대", "주차", "통행료"]

# ── 조정 계수 ──────────────────────────────────────────
자재_FACTOR      = {"일반": 0.85, "중급": 1.0, "고급": 1.25}
자재등급_TO_GRADE = {"일반": "일반", "중급": "중급", "고급": "고급"}
연식_FACTOR = {
    "신축(3년이하)": 0.80,
    "10년이하":      0.90,
    "10~20년":       1.00,
    "20년이상":      1.15,
}
거주_FACTOR  = {"거주중": 1.10, "공실": 1.00}
지역_FACTOR  = {"서울": 1.12, "수도권": 1.05, "지방": 1.00}
시기_FACTOR  = {
    "1개월이내": 1.05,
    "1~3개월":  1.00,
    "3개월이후": 0.95,
    "미정":      1.00,
}
# 철거 추가 단가 (원/평) — 기존 철거비가 없는 경우 보정
철거_평당 = {"있음": 25000, "없음": 0, "모름": 12000}

# 양중비 (엘리베이터 없을 때만 층수 × 계수)
양중_층당 = 150000  # 원/층 (사다리차 중간단가 30만 × 2~3회 / 5층 기준)

# 트럭 접근 불가 시 운반비 할증 보정계수
트럭_FACTOR: dict[str, float] = {
    "가능":          1.00,
    "불가(골목·지하)": 1.07,   # 골목·지하 등 접근 제한 → +7%
    "모름":          1.03,   # 불확실 → 보수적 +3%
}

# 마감/공과잡비: DB cost 키 없음 → 총 공사비의 일정 비율로 산정
마감_비율 = 0.03    # 3% (보양·TV방수·엘리베이터보양·항균관리 등 포함)

# 전체 리모델링 버퍼 (total_costs 기반 경로)
# cost_* 합산은 실제 총금액의 약 69%만 커버 → total_costs를 직접 사용
# actual이 견적서보다 10~20% 높은 케이스를 최댓값 버퍼로 커버
전체_버퍼_LO = 1.00   # 최솟값: total_costs 그대로
전체_버퍼_HI = 1.25   # 최댓값: 25% 안전마진

# 전체 리모델링 기본 공정 목록 (시공범위="전체" 시 자동 포함 기준)
전체_기본공정 = [
    "철거", "설비", "전기/조명", "목공", "도배",
    "마루", "타일", "욕실", "주방", "도장", "마감/공과잡비",
]

# ── P2 보정 상수 ───────────────────────────────────────

# 도배 범위별 전체 평수 대비 면적 비율
도배_범위_비율: dict[str, float] = {
    "전체": 1.00,
    "거실": 0.35,
    "침실": 0.40,   # 3개방 기준 — 방 개수로 보정됨
    "주방": 0.10,
}

# 방 개수별 침실 면적 비율 (3개방 0.40 기준, 방 하나당 ≈0.13)
방별_침실_비율: dict[int, float] = {1: 0.13, 2: 0.27, 3: 0.40, 4: 0.53}

# 도배지 종류별 단가 보정계수 (실크벽지 기준 1.0)
도배지_FACTOR: dict[str, float] = {
    "실크벽지":  1.00,
    "합지벽지":  0.75,
    "천연벽지":  1.40,
}

# 방 개수별 마루·장판 면적 보정계수 (3개방 기준 1.0)
방_마루_비율: dict[int, float] = {1: 0.70, 2: 0.85, 3: 1.00, 4: 1.15}

import sys
import os
import chromadb
from chromadb.utils import embedding_functions
from chroma_client import get_chroma_client

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db import get_sync_collection

# ══════════════════════════════════════════════════════
# EstimateEngine
# ══════════════════════════════════════════════════════

class EstimateEngine:

    def __init__(self):
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client = get_chroma_client()
        self.collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
        self._cases_col = get_sync_collection("estimate_cases")

    @staticmethod
    def _normalize_desc(category: str, desc: str) -> tuple:
        """description을 NORM_MAP 기준으로 정규화.

        반환: (normalized_desc | None, was_normalized: bool)
          - None: 집계 제외 (skip 키워드 또는 NORM_MAP에서 None으로 지정)
          - was_normalized=True : NORM_MAP에 매핑됨 → min_cases=1 허용
          - was_normalized=False: 매핑 없이 원본 반환 → min_cases=2 유지
        """
        for kw in _SKIP_KEYWORDS:
            if kw in desc:
                return None, False
        rules = NORM_MAP.get(category, [])
        for keywords, normalized in rules:
            if any(kw in desc for kw in keywords):
                return normalized, (normalized is not None)
        return desc, False  # 매핑 없으면 원본 유지, 비정규화 표시

    @staticmethod
    def _normalize_spec_desc(desc: str) -> str:
        """치수 구분 기호(×/✕/*/+)를 × 로 통일해 중복 항목 방지."""
        return _DIM_SEP_RE.sub(r'\1×\2', desc)

    @staticmethod
    def _filter_by_scope_coverage(cases: list[dict], 공종들: list[str],
                                   min_ratio: float = SCOPE_COVERAGE_MIN) -> list[dict]:
        """요청 공종 비용 합계 / 사례 총 비용 비율이 낮은 케이스 제거.

        전체 시공에서 참고 사례가 요청 공종 외 다수 공종을 포함해
        total_cost가 과대 산정되는 문제를 방지한다.
        """
        def _coverage(case: dict) -> float:
            tc = int(case.get("total_cost") or 0)
            if tc <= 0:
                return 0.0
            trade_sum = sum(
                int(case.get(k) or 0)
                for g in 공종들
                for k in 공종_TO_COST.get(g, [])
            )
            return trade_sum / tc

        filtered = [c for c in cases if _coverage(c) >= min_ratio]
        return filtered if len(filtered) >= 3 else cases

    def collect_line_items(self, cases, 공종들):
        """유사 사례 JSON에서 공종별 line_items를 집계해 명세 반환.

        반환 구조:
        {
          "도배공사": [
            {
              "description": "실크벽지",
              "amount_range": {"최소": ..., "중간": ..., "최대": ...},
              "등장_사례_수": 18
            },
            ...
          ],
          ...
        }
        """
        # 집계할 category 목록 결정
        target_categories: list[str] = []
        for 공종 in 공종들:
            target_categories.extend(공종_TO_CATEGORY.get(공종, []))

        if not target_categories:
            return {}

        # MongoDB에서 article_id 배치 조회 후 line_items 집계
        # amounts[category][normalized_desc] = [amount, ...]
        amounts = defaultdict(lambda: defaultdict(list))

        article_ids = [str(c.get("article_id", "")) for c in cases if c.get("article_id")]
        docs = {
            str(d["article_id"]): d
            for d in self._cases_col.find(
                {"article_id": {"$in": article_ids}},
                {"parsed_estimate": 1, "article_id": 1, "_id": 0},
            )
        }

        for case in cases:
            aid = str(case.get("article_id", ""))
            data = docs.get(aid)
            if not data:
                continue
            pe = data.get("parsed_estimate")
            if not pe:
                continue
            for item in pe.get("line_items", []):
                cat = item.get("category", "")
                if cat not in target_categories:
                    continue
                amt = int(item.get("amount") or 0)
                if amt <= 0:
                    continue
                desc = self._normalize_spec_desc(item.get("description", ""))
                normalized, was_norm = self._normalize_desc(cat, desc)
                if normalized is None:
                    continue
                amounts[cat][(normalized, was_norm)].append(amt)

        # 공종별로 집계 결과 생성
        result = {}
        for 공종 in 공종들:
            cats = 공종_TO_CATEGORY.get(공종, [])
            excluded = 공종_EXCLUDE_DESC.get(공종, set())

            # 서브 카테고리(타일공사·수전공사·도기공사 등)를 공종 단위로 합산
            # → 동일 normalized description 중복 방지
            # (desc, was_normalized) 키로 병합
            merged: dict[tuple, list] = defaultdict(list)
            for cat in cats:
                for (desc, was_norm), amt_list in amounts.get(cat, {}).items():
                    if desc in excluded:
                        continue
                    merged[(desc, was_norm)].extend(amt_list)

            # 비율 기반 threshold: 전체 사례 수의 SPEC_RATIO 이상 등장해야 포함
            ratio_min = max(2, math.ceil(len(cases) * SPEC_RATIO))

            items_for_공종: list[dict] = []
            for (desc, was_norm), amt_list in merged.items():
                # NORM_MAP 매핑 항목: 2건 이상 / 비매핑 raw 항목: 비율 기반 threshold
                min_cases = 2 if was_norm else ratio_min
                if len(amt_list) < min_cases:
                    continue
                trimmed = sorted(amt_list)
                if len(trimmed) > 4:
                    cut = len(trimmed) // 5
                    trimmed = trimmed[cut:-cut]
                items_for_공종.append({
                    "description": desc,
                    "amount_range": {
                        "최소": min(trimmed),
                        "중간": int(statistics.median(trimmed)),
                        "최대": max(trimmed),
                    },
                    "등장_사례_수": len(amt_list),
                })

            # 등장 사례 수 내림차순 정렬, 부대비용은 하단으로
            ancillary = {"인건비", "운송비", "부자재"}
            items_for_공종.sort(
                key=lambda x: (x["description"] in ancillary, -x["등장_사례_수"])
            )

            # 공종별 명세 항목 수 cap: 비부대비용 MAX_SPEC_ITEMS개 + 부대비용 전체
            non_anc = [x for x in items_for_공종 if x["description"] not in ancillary]
            anc     = [x for x in items_for_공종 if x["description"] in ancillary]
            items_for_공종 = non_anc[:MAX_SPEC_ITEMS] + anc

            if items_for_공종:
                result[공종] = items_for_공종

        return result

    # ── 1. 텍스트 쿼리 생성 ──────────────────────────────

    def build_query(self, inp: dict) -> str:
        parts = [
            f"{inp.get('평수', '?')}평",
            inp.get("지역", ""),
            inp.get("공간유형", "아파트"),
        ]

        # 시공범위를 쿼리에 반영해 유사 사례 매칭 정확도 향상
        if inp.get("시공범위") == "전체":
            parts.append("전체리모델링")
        else:
            parts += inp.get("공종", [])

        if inp.get("건물연식") in ("20년이상", "10~20년"):
            parts.append("구축")
        if inp.get("자재등급") == "고급":
            parts.append("고급자재")

        마루 = inp.get("마루", {})
        if 마루.get("자재종류"):
            parts.append(마루["자재종류"])

        도배 = inp.get("도배", {})
        if 도배.get("도배지종류"):
            parts.append(도배["도배지종류"])

        parts.append("리모델링")
        return " ".join(p for p in parts if p)

    # ── 2. 메타데이터 필터 생성 ──────────────────────────

    def build_filters(self, inp: dict):
        평수    = int(inp.get("평수") or 0)
        지역들  = REGION_MAP.get(inp.get("지역", "서울"), ["서울"])
        공종들  = inp.get("공종", [])

        conditions = []

        if 평수:
            conditions.append({"size_pyeong": {"$gte": 평수 - SIZE_RANGE}})
            conditions.append({"size_pyeong": {"$lte": 평수 + SIZE_RANGE}})

        for 공종 in 공종들:
            flag = 공종_TO_HAS.get(공종)
            if flag:
                conditions.append({flag: {"$eq": "true"}})

        if len(지역들) == 1:
            conditions.append({"region": {"$eq": 지역들[0]}})
        elif len(지역들) > 1:
            conditions.append({"region": {"$in": 지역들}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── 3. ChromaDB 조회 (점진적 필터 완화) ──────────────────

    def _build_filter(self, 평수: int, 지역들: list, 공종들: list,
                      use_size=True, use_region=True, use_has=True,
                      use_grade=False, grade: str = None):
        """네 그룹(평수/지역/has_*/자재등급) 조합으로 ChromaDB 필터 생성.
        ChromaDB Cloud where 조건 한도(8개)를 초과하지 않도록 has_* 슬롯을 동적으로 제한.
        """
        MAX_WHERE = 8
        conds = []
        if use_size and 평수:
            conds.append({"size_pyeong": {"$gte": 평수 - SIZE_RANGE}})
            conds.append({"size_pyeong": {"$lte": 평수 + SIZE_RANGE}})
        if use_region and 지역들:
            if len(지역들) == 1:
                conds.append({"region": {"$eq": 지역들[0]}})
            else:
                conds.append({"region": {"$in": 지역들}})
        if use_grade and grade:
            conds.append({"material_grade": {"$eq": grade}})
        if use_has:
            seen = set()
            remaining = MAX_WHERE - len(conds)  # 남은 슬롯만큼만 has_* 적용
            for 공종 in 공종들:
                if len(seen) >= remaining:
                    break
                flag = 공종_TO_HAS.get(공종)
                if flag and flag not in seen:
                    conds.append({flag: {"$eq": "true"}})
                    seen.add(flag)
        if not conds:
            return None
        if len(conds) == 1:
            return conds[0]
        return {"$and": conds}

    def retrieve_cases(self, query: str, inp: dict):
        """
        점진적 폴백으로 유사 사례 검색.
          Stage 1: 평수 + 지역 + has_* + 자재등급  (전체 조건)
          Stage 2: 평수 + 지역 + has_*             (등급 완화)
          Stage 3: 평수 + has_*                    (지역 완화)
          Stage 4: has_*만                         (평수 완화)
          Stage 5: 필터 없음                        (최후 수단)
        """
        n       = min(TOP_K, self.collection.count())
        평수    = int(inp.get("평수") or 0)
        지역들  = REGION_MAP.get(inp.get("지역", "서울"), ["서울"])
        공종들  = inp.get("공종", [])
        grade   = 자재등급_TO_GRADE.get(inp.get("자재등급", "중급"), "중급")

        def _query(where=None):
            kw = dict(query_texts=[query], n_results=n, include=["metadatas"])
            if where:
                kw["where"] = where
            return self.collection.query(**kw)["metadatas"][0]

        for use_size, use_region, use_has, use_grade in [
            (True,  True,  True,  True),   # Stage 1: 전체 조건 + 등급
            (True,  True,  True,  False),  # Stage 2: 등급 완화
            (True,  False, True,  False),  # Stage 3: 지역 완화
            (False, False, True,  False),  # Stage 4: 평수 완화
            (False, False, False, False),  # Stage 5: 필터 없음
        ]:
            where = self._build_filter(평수, 지역들, 공종들,
                                       use_size=use_size,
                                       use_region=use_region,
                                       use_has=use_has,
                                       use_grade=use_grade,
                                       grade=grade)
            try:
                cases = _query(where)
                if len(cases) >= 3:
                    return cases
            except Exception:
                pass

        return _query(None)

    # ── 4. 사례에서 비용 추출 ────────────────────────────

    def extract_costs(self, cases: list[dict], 공종들: list[str]) -> dict:
        total_costs: list[int] = []
        cat_costs: dict[str, list[int]] = defaultdict(list)

        # 욕실+설비 동시 선택 시 욕실의 cost_설비 제거 (설비가 이미 독립 집계)
        욕실_keys = list(공종_TO_COST.get("욕실", []))
        if "욕실" in 공종들 and "설비" in 공종들:
            욕실_keys = [k for k in 욕실_keys if k != "cost_설비"]

        for case in cases:
            tc = int(case.get("total_cost") or 0)
            if tc > 0:
                total_costs.append(tc)

            # cost_* 필드 직접 읽기 (build_rag.py에서 저장한 값)
            for 공종 in 공종들 + ["철거"]:
                keys = 욕실_keys if 공종 == "욕실" else 공종_TO_COST.get(공종, [])
                for cost_key in keys:
                    val = int(case.get(cost_key) or 0)
                    if val > 0:
                        cat_costs[공종].append(val)

        return total_costs, cat_costs

    # ── 5. 조정 계수 계산 ────────────────────────────────

    def calc_factors(self, inp: dict):
        factor = 1.0
        notes  = []

        f = 자재_FACTOR.get(inp.get("자재등급", "중급"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"자재등급 {inp.get('자재등급')} ({f-1:+.0%})")

        f = 연식_FACTOR.get(inp.get("건물연식", "10~20년"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"건물연식 {inp.get('건물연식')} ({f-1:+.0%})")

        f = 거주_FACTOR.get(inp.get("거주중공사", "공실"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"거주 중 공사 ({f-1:+.0%})")

        f = 시기_FACTOR.get(inp.get("공사시기", "미정"), 1.0)
        if f != 1.0:
            factor *= f
            label = "성수기 할증" if f > 1.0 else "비수기 할인"
            notes.append(f"{label} ({f-1:+.1%})")

        # 지역 계수 미적용: RAG 필터가 이미 같은 지역 사례를 가져오므로 이중 적용 방지

        # 트럭 접근 불가 할증
        f = 트럭_FACTOR.get(inp.get("트럭접근", "가능"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"트럭 접근 {inp.get('트럭접근')} ({f-1:+.0%})")

        # 양중비 (엘리베이터 없음)
        양중 = 0
        if inp.get("엘리베이터") == "없음":
            층수 = int(inp.get("층수") or 1)
            양중 = 층수 * 양중_층당
            notes.append(f"사다리차 양중비 +{양중:,}원 ({층수}층)")

        # 철거 추가비: "철거"가 공종 목록에 있으면 RAG에서 직접 추출하므로 flat-rate 미적용
        평수 = int(inp.get("평수") or 0)
        if "철거" not in inp.get("공종", []):
            철거추가 = 철거_평당.get(inp.get("철거여부", "모름"), 0) * 평수
            if 철거추가:
                notes.append(f"철거비 보정 +{철거추가:,}원")
        else:
            철거추가 = 0

        # 마감/공과잡비: 총 공사비의 3% 비율로 나중에 별도 산정
        마감비율_적용 = "마감/공과잡비" in inp.get("공종", [])
        if 마감비율_적용:
            notes.append(f"마감/공과잡비 포함 (총 공사비의 {마감_비율:.0%})")

        return factor, notes, 양중 + 철거추가, 마감비율_적용

    # ── 6. 공종별 개별 보정계수 계산 ────────────────────────

    @staticmethod
    def calc_공종_factors(inp: dict) -> dict[str, tuple[float, list]]:
        """도배 범위·도배지 종류·방 개수 등 공종별 개별 보정계수 계산.

        반환: { 공종명: (factor, [note, ...]) }
        factor가 1.0인 공종은 포함하지 않음.
        """
        result: dict[str, tuple[float, list]] = {}
        공종들  = inp.get("공종", [])
        방개수  = min(int(inp.get("방개수") or 3), 4)   # 4개 이상은 4로 처리

        # ── 도배 ────────────────────────────────────────
        if "도배" in 공종들:
            도배_inp = inp.get("도배", {})
            f = 1.0
            notes: list[str] = []

            # 범위 보정 — "전체"가 아닌 경우에만 적용
            범위_raw = 도배_inp.get("범위", "전체")
            범위_list = 범위_raw if isinstance(범위_raw, list) else [범위_raw]

            if "전체" not in 범위_list:
                ratio = 0.0
                for r in 범위_list:
                    if r == "침실":
                        # 방 개수로 침실 면적 비율 보정
                        ratio += 방별_침실_비율.get(방개수, 0.40)
                    else:
                        ratio += 도배_범위_비율.get(r, 0.0)
                ratio = max(0.05, min(ratio, 1.0))  # 5~100% 클램프
                f *= ratio
                notes.append(f"도배 범위 {'·'.join(범위_list)} (면적 {ratio:.0%})")

            # 도배지 종류 보정
            도배지 = 도배_inp.get("도배지종류", "실크벽지")
            f_지   = 도배지_FACTOR.get(도배지, 1.0)
            if f_지 != 1.0:
                f *= f_지
                notes.append(f"도배지 {도배지} ({f_지-1:+.0%})")

            if f != 1.0:
                result["도배"] = (f, notes)

        # ── 마루 / 장판 (방 개수 보정) ──────────────────
        for 공종 in ("마루", "장판"):
            if 공종 in 공종들:
                f_마루 = 방_마루_비율.get(방개수, 1.0)
                if f_마루 != 1.0:
                    result[공종] = (
                        f_마루,
                        [f"방 {방개수}개 기준 바닥 면적 보정 ({f_마루-1:+.0%})"],
                    )

        # ── 주방 / 가구 cost_가구 비율 분배 ─────────────
        # 둘 다 cost_가구를 공유하므로 동시 선택 시 중복 방지
        주방_선택 = "주방" in 공종들
        가구_선택 = "가구" in 공종들

        if 주방_선택 and 가구_선택:
            # 전체 가구비를 주방 60% / 붙박이·수납 40%로 분배
            result["주방"] = (0.60, ["주방·가구 동시 선택 — 주방(싱크대 등) 60% 배분"])
            result["가구"] = (0.40, ["주방·가구 동시 선택 — 가구(붙박이 등) 40% 배분"])
        elif 가구_선택:
            # 가구만 선택: 붙박이·신발장은 전체 가구비의 약 40%
            result["가구"] = (0.40, ["가구(붙박이·신발장) 단독 선택 (~40% 적용)"])
        # 주방 단독 선택: cost_가구 전체 사용 → 보정 없음

        return result

    # ── 7. 공종별 단가 범위 계산 ─────────────────────────

    @staticmethod
    def cost_range(values, top_trim: float = 0.12):
        """최솟값: 하위 20% 제거 / 최댓값: 상위 12% 제거."""
        if not values:
            return None
        trimmed = sorted(values)
        if len(trimmed) > 4:
            bot = len(trimmed) // 5          # 하위 20% 제거
            top = max(1, int(len(trimmed) * top_trim))  # 상위 12% 제거
            trimmed = trimmed[bot:-top]
        return {
            "최소": min(trimmed),
            "최대": max(trimmed),
            "중간": int(statistics.median(trimmed)),
        }

    # ── 7. 최종 가견적 생성 ──────────────────────────────

    def generate(self, inp: dict) -> dict:
        공종들     = inp.get("공종", [])
        시공범위   = inp.get("시공범위", "부분")
        query      = self.build_query(inp)
        cases      = self.retrieve_cases(query, inp)

        # 마감/공과잡비는 DB에서 비용 추출 불가 — cost 계산 대상에서 제외
        추출대상_공종들 = [c for c in 공종들 if c != "마감/공과잡비"]

        # 전체 시공: 요청 공종과 비슷한 공종 구성의 케이스만 사용
        # (범위가 넓은 케이스가 total_cost를 과대 산정하는 문제 방지)
        if 시공범위 == "전체":
            cases = self._filter_by_scope_coverage(cases, 추출대상_공종들)

        total_costs, cat_costs = self.extract_costs(cases, 추출대상_공종들)

        if not total_costs:
            return {"error": "유사 사례를 찾을 수 없습니다. 조건을 조정해 주세요."}

        factor, notes, extra, 마감비율_적용 = self.calc_factors(inp)

        # 공종별 개별 보정 (도배 범위·도배지·방 개수) — cat_costs에 직접 반영
        공종_factors = self.calc_공종_factors(inp)
        for 공종, (f, 공종_notes) in 공종_factors.items():
            if 공종 in cat_costs:
                cat_costs[공종] = [int(v * f) for v in cat_costs[공종]]
            notes.extend(공종_notes)

        # 공종별 단가 범위 (글로벌 보정계수 포함)
        공종별_범위 = {}
        for 공종 in 추출대상_공종들:
            r = self.cost_range(cat_costs.get(공종, []))
            if r:
                공종별_범위[공종] = {
                    "최소": int(r["최소"] * factor),
                    "중간": int(r["중간"] * factor),
                    "최대": int(r["최대"] * factor),
                }

        # 총 견적 범위 계산
        # 전체 리모델링: total_costs 기반 (cost_* 합산은 실제 총금액의 ~69%만 커버)
        # 부분 리모델링: cost_* 공종별 합산 (전체 total_cost 쓰면 과대산정)
        if inp.get("시공범위") == "전체" and total_costs:
            r = self.cost_range(total_costs)
            adj_lo = int(r["최소"] * factor) + extra
            adj_hi = int(r["최대"] * factor) + extra
            if 마감비율_적용:
                adj_lo = int(adj_lo * (1 + 마감_비율))
                adj_hi = int(adj_hi * (1 + 마감_비율))
            adj_lo = int(adj_lo * 전체_버퍼_LO)
            adj_hi = int(adj_hi * 전체_버퍼_HI)
        elif 공종별_범위:
            adj_lo = sum(r["최소"] for r in 공종별_범위.values()) + extra
            adj_hi = sum(r["최대"] for r in 공종별_범위.values()) + extra
            if 마감비율_적용:
                adj_lo = int(adj_lo * (1 + 마감_비율))
                adj_hi = int(adj_hi * (1 + 마감_비율))
        else:
            # fallback: 공종별 비용 데이터도 없을 때
            trimmed = sorted(total_costs)
            if len(trimmed) > 4:
                cut = len(trimmed) // 5
                trimmed = trimmed[cut:-cut]
            adj_lo = int(min(trimmed) * factor) + extra
            adj_hi = int(max(trimmed) * factor) + extra
            if 마감비율_적용:
                adj_lo = int(adj_lo * (1 + 마감_비율))
                adj_hi = int(adj_hi * (1 + 마감_비율))

        adj_mid = (adj_lo + adj_hi) // 2

        # 참고 사례 요약 (상위 5개)
        참고_사례 = sorted(
            [
                {
                    "article_id": c.get("article_id"),
                    "지역":   c.get("region"),
                    "평수":   c.get("size_pyeong"),
                    "총금액": int(c.get("total_cost") or 0),
                    "평당":   int(c.get("cost_per_pyeong") or 0),
                }
                for c in cases
                if c.get("total_cost")
            ],
            key=lambda x: abs(x["평수"] - inp.get("평수", 0))
        )[:5]

        # 공종별 항목 명세 (line_items 집계) — 마감/공과잡비 제외
        공종별_항목_명세 = self.collect_line_items(cases, 추출대상_공종들)

        output = {
            "총_견적_범위": {
                "최소": adj_lo,
                "최대": adj_hi,
                "중간": adj_mid,
            },
            "공종별_단가_범위": 공종별_범위,
            "공종별_항목_명세": 공종별_항목_명세,
            "보정_적용":    notes,
            "시공범위":     시공범위,
            "선택_공종":    공종들,
            "참고_사례_수": len(total_costs),
            "참고_사례":    참고_사례,
            "검색_쿼리":    query,
        }

        # 단일 공종(마감/공과잡비 제외) 요청 시 주의 문구
        실공종_수 = len([c for c in 공종들 if c != "마감/공과잡비"])
        if 실공종_수 == 1:
            output["단독시공_주의"] = (
                "단일 공종 요청입니다. 전체 리모델링 사례에서 해당 공종 비용을 추출하여 산출했으며, "
                "단독 시공 시 실제 가격이 5~10% 높을 수 있습니다."
            )

        return output


# ══════════════════════════════════════════════════════
# CLI 테스트
# ══════════════════════════════════════════════════════

SAMPLE_INPUT = {
    "공종": [
        "철거", "설비", "전기/조명", "목공", "도배",
        "마루", "욕실", "주방", "도장", "마감/공과잡비",
    ],
    "시공범위":   "전체",
    "공간유형":   "아파트",
    "평수":       30,
    "방개수":     3,
    "지역":       "서울",

    "건물연식":   "20년이상",
    "자재등급":   "중급",
    "철거여부":   "있음",
    "층수":       8,
    "엘리베이터": "있음",
    "트럭접근":   "가능",
    "거주중공사": "공실",
    "공사시기":   "1~3개월",

    "도배": {"범위": ["거실", "침실"], "도배지종류": "실크벽지", "초배포함": "있음"},
    "마루": {"자재종류": "강마루", "범위": "전체", "철거여부": "있음"},
    "욕실": {"개수": 1, "크기": "중형", "도기교체": "있음",
             "방수포함": "있음", "욕조샤워부스": "없음", "타일등급": "중급"},
}


def fmt(n: int) -> str:
    return f"{n:,}원 ({n // 10000}만원)"


def main():
    # .env 자동 로드 (현재 폴더 → 상위 폴더 순으로 탐색)
    try:
        from dotenv import load_dotenv
        _here = pathlib.Path(__file__).parent
        for _p in [_here / ".env", _here.parent / ".env"]:
            if _p.exists():
                load_dotenv(dotenv_path=_p)
                break
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="user_input JSON 파일 경로 (없으면 샘플 입력 사용)")
    args = parser.parse_args()

    if args.input:
        inp = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    else:
        inp = SAMPLE_INPUT
        print("[샘플 입력 사용]\n")

    engine = EstimateEngine()
    result = engine.generate(inp)

    if "error" in result:
        print(f"[ERR] {result['error']}")
        return

    print("=" * 55)
    print("  가견적 결과")
    print("=" * 55)
    print(f"  시공범위: {result.get('시공범위', '-')}  |  선택 공종: {', '.join(result.get('선택_공종', []))}")
    r = result["총_견적_범위"]
    print(f"  총 견적 범위")
    print(f"    최소: {fmt(r['최소'])}")
    print(f"    중간: {fmt(r['중간'])}")
    print(f"    최대: {fmt(r['최대'])}")

    if result["공종별_단가_범위"]:
        print(f"\n  공종별 단가 범위")
        for 공종, rng in result["공종별_단가_범위"].items():
            print(f"    {공종:5s}: {rng['최소']//10000}만 ~ {rng['최대']//10000}만원")

    명세 = result.get("공종별_항목_명세", {})
    if 명세:
        print(f"\n  공종별 항목 명세")
        for 공종, items in 명세.items():
            print(f"    [{공종}]")
            for item in items:
                lo = item["amount_range"]["최소"] // 10000
                hi = item["amount_range"]["최대"] // 10000
                mid = item["amount_range"]["중간"] // 10000
                cnt = item["등장_사례_수"]
                print(f"      {item['description']:20s}  {lo}만~{hi}만원 (중간 {mid}만원) | {cnt}건")

    if result["보정_적용"]:
        print(f"\n  적용된 보정")
        for note in result["보정_적용"]:
            print(f"    · {note}")

    print(f"\n  참고 사례 ({result['참고_사례_수']}건 중 상위 5건)")
    for c in result["참고_사례"]:
        print(f"    {c['article_id']} | {c['지역']} {c['평수']}평 | "
              f"{c['총금액']//10000}만원 (평당 {c['평당']//10000}만원)")

    print(f"\n  검색 쿼리: {result['검색_쿼리']}")
    print("=" * 55)
    print("\n※ 본 견적은 참고용이며, 정확한 견적은 현장 방문 후 확정됩니다.")


if __name__ == "__main__":
    main()
