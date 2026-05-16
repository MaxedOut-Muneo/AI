from datetime import datetime, timezone

from bson import ObjectId

from db import get_collection_from_db

DB_NAME = "risk_detector_db"
COLLECTION_NAME = "risk_reports"

def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc

async def save_risk_report(user_id: str, user_input: dict, result: dict) -> str:
    doc = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "input": user_input,
        "result": result,
    }
    res = await get_collection_from_db(DB_NAME, COLLECTION_NAME).insert_one(doc)
    return str(res.inserted_id)

async def list_risk_reports(user_id: str) -> list[dict]:
    col = get_collection_from_db(DB_NAME, COLLECTION_NAME)
    cursor = col.find({"user_id": user_id})
    docs = await cursor.to_list(length=100)
    return [_serialize(d) for d in docs]

async def delete_risk_report(report_id: str, user_id: str) -> bool | None:
    try:
        oid = ObjectId(report_id)
    except Exception:
        return None

    col = get_collection_from_db(DB_NAME, COLLECTION_NAME)
    res = await col.delete_one({"_id": oid, "user_id": user_id})
    return res.deleted_count > 0