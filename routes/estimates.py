from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Body, Header, HTTPException
from bson import ObjectId
from db import get_collection
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "estimate"))
from estimate_engine import EstimateEngine

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


def serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


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

    doc = {
        "user_id": x_user_id,
        "created_at": datetime.now(timezone.utc),
        "input": user_input,
        "result": result,
    }
    res = await get_collection("estimates").insert_one(doc)
    return {"id": str(res.inserted_id)}


@router.get("")
async def list_estimates(x_user_id: str = Header(..., example="user_abc123")):
    col = get_collection("estimates")
    cursor = col.find({"user_id": x_user_id}, {"참고_사례": 0, "참고_사례_수": 0, "검색_쿼리": 0})
    docs = await cursor.to_list(length=100)
    return [serialize(d) for d in docs]


@router.delete("/{estimate_id}", status_code=204)
async def delete_estimate(estimate_id: str = "6801234567890abcdef12345", x_user_id: str = Header(..., example="user_abc123")):
    try:
        oid = ObjectId(estimate_id)
    except Exception:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")

    col = get_collection("estimates")
    res = await col.delete_one({"_id": oid, "user_id": x_user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="견적을 찾을 수 없습니다.")
