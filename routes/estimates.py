from typing import Optional

from fastapi import APIRouter, Body, Header, HTTPException

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "estimate"))
from estimate_engine import EstimateEngine
from estimate_service import save_estimate, list_estimates, delete_estimate

router = APIRouter(prefix="/estimates", tags=["estimates"])

_engine: Optional[EstimateEngine] = None

_SAMPLE_INPUT = {
    "공종": ["철거", "설비", "전기/조명", "목공", "도배", "마루", "욕실", "주방", "도장", "마감/공과잡비"],
    "시공범위": "전체",
    "공간유형": "아파트",
    "평수": 30,
    "방개수": 3,
    "지역": "서울",
    "건물연식": "20년이상",
    "자재등급": "중급",
    "철거여부": "있음",
    "층수": 8,
    "엘리베이터": "있음",
    "트럭접근": "가능",
    "거주중공사": "공실",
    "공사시기": "1~3개월",
    "도배": {"범위": ["거실", "침실"], "도배지종류": "실크벽지", "초배포함": "있음"},
    "마루": {"자재종류": "강마루", "범위": "전체", "철거여부": "있음"},
    "욕실": {"개수": 1, "크기": "중형", "도기교체": "있음", "방수포함": "있음", "욕조샤워부스": "없음", "타일등급": "중급"},
    "주방": {"싱크대교체": "있음", "형태": "I자형", "길이": "2~3m"},
}

_SAMPLE_RESULT = {
    "총_견적_범위": {"최소": 15000000, "중간": 20000000, "최대": 25000000},
    "공종별_단가_범위": {
        "도배": {"최소": 1200000, "중간": 1500000, "최대": 1800000},
        "마루": {"최소": 2000000, "중간": 2500000, "최대": 3000000},
    },
    "공종별_항목_명세": {},
    "보정_적용": ["서울 지역 보정 +5%", "20년 이상 건물 보정 +10%"],
    "시공범위": "전체",
    "선택_공종": ["철거", "설비", "전기/조명", "목공", "도배", "마루", "욕실", "주방", "도장", "마감/공과잡비"],
}


def get_engine() -> EstimateEngine:
    global _engine
    if _engine is None:
        _engine = EstimateEngine()
    return _engine


@router.post("/generate")
async def generate(body: dict = Body(example=_SAMPLE_INPUT)):
    result = get_engine().generate(body)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.post("/save", status_code=201)
async def save(
    body: dict = Body(example={"input": _SAMPLE_INPUT, "result": _SAMPLE_RESULT}),
    x_user_id: str = Header(..., example="user_abc123"),
):
    user_input = body.get("input")
    result = body.get("result")
    if not user_input or not result:
        raise HTTPException(status_code=400, detail="input과 result가 필요합니다.")

    estimate_id = await save_estimate(x_user_id, user_input, result)
    return {"id": estimate_id}


@router.get("")
async def get_estimates(x_user_id: str = Header(..., example="user_abc123")):
    return await list_estimates(x_user_id)


@router.delete("/{estimate_id}", status_code=204)
async def remove_estimate(
    estimate_id: str = "6801234567890abcdef12345",
    x_user_id: str = Header(..., example="user_abc123"),
):
    result = await delete_estimate(estimate_id, x_user_id)
    if result is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")
    if not result:
        raise HTTPException(status_code=404, detail="견적을 찾을 수 없습니다.")
