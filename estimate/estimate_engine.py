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
      "공종":       ["도배", "마루", "욕실"],   # 도배/장판/마루/주방/욕실
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
import pathlib
import statistics
from collections import defaultdict

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════

COLLECTION_NAME = "estimates"
EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K           = 15   # 유사 사례 최대 조회 수
SIZE_RANGE      = 7    # 평수 ±7평 필터

# 사용자 지역 → DB region 매핑
REGION_MAP = {
    "서울":  ["서울"],
    "수도권": ["경기", "인천"],
    "지방":  ["부산", "대구", "울산", "광주", "대전", "세종",
               "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주", "기타"],
}

# 사용자 공종 → DB has_* 플래그
공종_TO_HAS = {
    "도배": "has_도배",
    "장판": "has_바닥",
    "마루": "has_바닥",
    "욕실": "has_욕실",
    "주방": "has_가구",
}

# 사용자 공종 → DB cost_* 키
공종_TO_COST = {
    "도배": ["cost_도배"],
    "장판": ["cost_바닥"],
    "마루": ["cost_바닥"],
    "욕실": ["cost_욕실", "cost_설비", "cost_타일"],
    "주방": ["cost_가구"],
    "철거": ["cost_철거"],
}

# 사용자 공종 → 파싱 데이터 category 이름 매핑
공종_TO_CATEGORY = {
    "도배": ["도배공사"],
    "마루": ["바닥공사"],
    "장판": ["바닥공사"],
    "욕실": ["타일공사", "수전공사", "도기공사"],
    "주방": ["가구공사"],
}

# 공종별 제외 description (마루↔장판 혼재 방지)
공종_EXCLUDE_DESC: dict[str, set] = {
    "마루": {"장판"},
    "장판": {"강마루"},
}

# description 정규화 테이블: category → [(포함 키워드 목록, 대표 이름)]
# 매칭 순서가 우선순위 — 위에 있을수록 먼저 매칭
NORM_MAP = {
    "도배공사": [
        (["초배"],                          "초배지"),
        (["실크"],                          "실크벽지"),
        (["합지"],                          "합지벽지"),
        (["퍼티", "바탕면처리", "벽지제거"], "바탕면처리·퍼티"),
        (["부자재"],                         "부자재"),
        (["인건비", "안건비"],               "인건비"),
        (["운송비"],                         "운송비"),
    ],
    "바닥공사": [
        (["강마루"],                         "강마루"),
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
}

# 사용자에게 보여줄 필요 없는 내부 항목 (식대, 주차비 등)
_SKIP_KEYWORDS = ["식대", "주차", "통행료"]

# ── 조정 계수 ──────────────────────────────────────────
자재_FACTOR = {"일반": 0.85, "중급": 1.0, "고급": 1.25}
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
양중_층당 = 30000   # 원/층

import chromadb
from chromadb.utils import embedding_functions
from chroma_client import get_chroma_client

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
        self._id_index = self._build_id_index()

    def _build_id_index(self):
        """estimate_data 디렉터리를 스캔해 article_id → JSON 경로 인덱스 생성."""
        index: dict[str, pathlib.Path] = {}
        base = pathlib.Path(__file__).parent / "estimate_data"
        for json_path in base.rglob("*.json"):
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                data = raw[0] if isinstance(raw, list) else raw
                aid = data.get("article_id")
                if aid:
                    index[str(aid)] = json_path
            except Exception:
                pass
        return index

    @staticmethod
    def _normalize_desc(category: str, desc: str):
        """description을 NORM_MAP 기준으로 대표 이름으로 정규화.
        _SKIP_KEYWORDS에 해당하면 None 반환 (집계 제외).
        매핑에 없으면 원본 description 반환.
        """
        for kw in _SKIP_KEYWORDS:
            if kw in desc:
                return None
        rules = NORM_MAP.get(category, [])
        for keywords, normalized in rules:
            if any(kw in desc for kw in keywords):
                return normalized
        return desc  # 매핑 없으면 원본 유지

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

        # article_id → JSON 로드 후 line_items 집계
        # amounts[category][normalized_desc] = [amount, ...]
        amounts = defaultdict(lambda: defaultdict(list))

        for case in cases:
            aid = str(case.get("article_id", ""))
            json_path = self._id_index.get(aid)
            if not json_path:
                continue
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                data = raw[0] if isinstance(raw, list) else raw
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
                    desc = item.get("description", "")
                    normalized = self._normalize_desc(cat, desc)
                    if normalized is None:
                        continue
                    amounts[cat][normalized].append(amt)
            except Exception:
                pass

        # 공종별로 집계 결과 생성
        result = {}
        for 공종 in 공종들:
            cats = 공종_TO_CATEGORY.get(공종, [])
            excluded = 공종_EXCLUDE_DESC.get(공종, set())

            # 서브 카테고리(타일공사·수전공사·도기공사 등)를 공종 단위로 합산
            # → 동일 normalized description 중복 방지
            merged: dict[str, list] = defaultdict(list)
            for cat in cats:
                for desc, amt_list in amounts.get(cat, {}).items():
                    if desc in excluded:
                        continue
                    merged[desc].extend(amt_list)

            items_for_공종: list[dict] = []
            for desc, amt_list in merged.items():
                if len(amt_list) < 2:  # 1건만 있는 항목 제외
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

            # 등장 사례 수 내림차순 정렬, 식대·운송비 등 부대비용은 하단으로
            ancillary = {"인건비", "운송비", "부자재"}
            items_for_공종.sort(
                key=lambda x: (x["description"] in ancillary, -x["등장_사례_수"])
            )
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

    # ── 3. ChromaDB 조회 (필터 실패 시 자동 완화) ─────────

    def retrieve_cases(self, query: str, filters):
        n = min(TOP_K, self.collection.count())

        def _query(where=None):
            kw = dict(query_texts=[query], n_results=n, include=["metadatas"])
            if where:
                kw["where"] = where
            return self.collection.query(**kw)["metadatas"][0]

        try:
            cases = _query(filters)
            if len(cases) >= 3:
                return cases
        except Exception:
            pass

        # 공종 필터만 유지 (평수·지역 완화)
        공종들 = [inp for inp in (filters or {}).get("$and", [])
                  if list(inp.keys())[0].startswith("has_")]
        try:
            relaxed = {"$and": 공종들} if len(공종들) > 1 else (공종들[0] if 공종들 else None)
            cases = _query(relaxed)
            if cases:
                return cases
        except Exception:
            pass

        # 필터 완전 제거
        return _query(None)

    # ── 4. 사례에서 비용 추출 ────────────────────────────

    def extract_costs(self, cases: list[dict], 공종들: list[str]) -> dict:
        total_costs: list[int] = []
        cat_costs: dict[str, list[int]] = defaultdict(list)

        for case in cases:
            tc = int(case.get("total_cost") or 0)
            if tc > 0:
                total_costs.append(tc)

            # cost_* 필드 직접 읽기 (build_rag.py에서 저장한 값)
            for 공종 in 공종들 + ["철거"]:
                for cost_key in 공종_TO_COST.get(공종, []):
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
            notes.append(f"자재등급 {inp.get('자재등급')} ({f:+.0%})")

        f = 연식_FACTOR.get(inp.get("건물연식", "10~20년"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"건물연식 {inp.get('건물연식')} ({f:+.0%})")

        f = 거주_FACTOR.get(inp.get("거주중공사", "공실"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"거주 중 공사 ({f:+.0%})")

        f = 시기_FACTOR.get(inp.get("공사시기", "미정"), 1.0)
        if f != 1.0:
            factor *= f
            label = "성수기 할증" if f > 1.0 else "비수기 할인"
            notes.append(f"{label} ({(f-1):+.1%})")

        f = 지역_FACTOR.get(inp.get("지역", "서울"), 1.0)
        if f != 1.0:
            factor *= f
            notes.append(f"지역 인건비 {inp.get('지역')} ({f:+.0%})")

        # 양중비 (엘리베이터 없음)
        양중 = 0
        if inp.get("엘리베이터") == "없음":
            층수 = int(inp.get("층수") or 1)
            양중 = 층수 * 양중_층당
            notes.append(f"사다리차 양중비 +{양중:,}원 ({층수}층)")

        # 철거 추가비 (DB 사례에 이미 반영됐을 수 있으나 보수적으로 추가)
        평수 = int(inp.get("평수") or 0)
        철거추가 = 철거_평당.get(inp.get("철거여부", "모름"), 0) * 평수
        if 철거추가:
            notes.append(f"철거비 보정 +{철거추가:,}원")

        return factor, notes, 양중 + 철거추가

    # ── 6. 공종별 단가 범위 계산 ─────────────────────────

    @staticmethod
    def cost_range(values):
        if not values:
            return None
        trimmed = sorted(values)
        if len(trimmed) > 4:
            cut = len(trimmed) // 5
            trimmed = trimmed[cut:-cut]
        return {
            "최소": min(trimmed),
            "최대": max(trimmed),
            "중간": int(statistics.median(trimmed)),
        }

    # ── 7. 최종 가견적 생성 ──────────────────────────────

    def generate(self, inp: dict) -> dict:
        공종들  = inp.get("공종", [])
        query   = self.build_query(inp)
        filters = self.build_filters(inp)
        cases   = self.retrieve_cases(query, filters)

        total_costs, cat_costs = self.extract_costs(cases, 공종들)

        if not total_costs:
            return {"error": "유사 사례를 찾을 수 없습니다. 조건을 조정해 주세요."}

        # 이상치 제거 후 총 견적 범위
        trimmed = sorted(total_costs)
        if len(trimmed) > 4:
            cut = len(trimmed) // 5
            trimmed = trimmed[cut:-cut]

        base_lo  = min(trimmed)
        base_hi  = max(trimmed)

        factor, notes, extra = self.calc_factors(inp)

        adj_lo = int(base_lo * factor) + extra
        adj_hi = int(base_hi * factor) + extra
        adj_mid = (adj_lo + adj_hi) // 2

        # 공종별 단가 범위
        공종별_범위 = {}
        for 공종 in 공종들:
            r = self.cost_range(cat_costs.get(공종, []))
            if r:
                공종별_범위[공종] = r

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

        # 공종별 항목 명세 (line_items 집계)
        공종별_항목_명세 = self.collect_line_items(cases, 공종들)

        output = {
            "총_견적_범위": {
                "최소": adj_lo,
                "최대": adj_hi,
                "중간": adj_mid,
            },
            "공종별_단가_범위": 공종별_범위,
            "공종별_항목_명세": 공종별_항목_명세,
            "보정_적용":    notes,
            "참고_사례_수": len(total_costs),
            "참고_사례":    참고_사례,
            "검색_쿼리":    query,
        }

        # 단일 공종 요청 시 주의 문구 추가
        if len(공종들) == 1:
            output["단독시공_주의"] = (
                "단일 공종 요청입니다. 전체 리모델링 사례에서 해당 공종 비용을 추출하여 산출했으며, "
                "단독 시공 시 실제 가격이 5~10% 높을 수 있습니다."
            )

        return output


# ══════════════════════════════════════════════════════
# CLI 테스트
# ══════════════════════════════════════════════════════

SAMPLE_INPUT = {
    "공종":       ["도배", "마루", "욕실"],
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

    "도배": {"범위": "전체", "도배지종류": "실크벽지", "초배포함": "있음"},
    "마루": {"자재종류": "강마루", "범위": "전체", "철거여부": "있음"},
    "욕실": {"개수": 1, "크기": "중형", "도기교체": "있음",
             "방수포함": "있음", "욕조샤워부스": "없음", "타일등급": "중급"},
}


def fmt(n: int) -> str:
    return f"{n:,}원 ({n // 10000}만원)"


def main():
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
