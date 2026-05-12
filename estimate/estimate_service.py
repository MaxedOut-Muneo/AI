from datetime import datetime, timezone

from bson import ObjectId

from db import get_collection


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


async def save_estimate(user_id: str, user_input: dict, result: dict) -> str:
    doc = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "input": user_input,
        "result": result,
    }
    res = await get_collection("estimates").insert_one(doc)
    return str(res.inserted_id)


async def list_estimates(user_id: str) -> list[dict]:
    col = get_collection("estimates")
    cursor = col.find(
        {"user_id": user_id},
        {"참고_사례": 0, "참고_사례_수": 0, "검색_쿼리": 0},
    )
    docs = await cursor.to_list(length=100)
    return [_serialize(d) for d in docs]


async def delete_estimate(estimate_id: str, user_id: str) -> bool:
    try:
        oid = ObjectId(estimate_id)
    except Exception:
        return None  # 잘못된 ID 형식

    col = get_collection("estimates")
    res = await col.delete_one({"_id": oid, "user_id": user_id})
    return res.deleted_count > 0
