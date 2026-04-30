"""
migrate_to_mongo.py — estimate_data/ JSON을 MongoDB estimate_cases 컬렉션에 마이그레이션

실행:
    python migrate_to_mongo.py
"""

import json
import pathlib
import os
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = pathlib.Path(__file__).parent / "estimate" / "estimate_data"


def main():
    client = MongoClient(os.environ["MONGO_URI"])
    col = client["estimate_db"]["estimate_cases"]

    col.create_index("article_id", unique=True)
    print("[INFO] article_id 인덱스 확인/생성 완료\n")

    ops = []
    skipped = 0

    for region_dir in sorted(DATA_DIR.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue
            json_path = article_dir / f"{article_dir.name}.json"
            if not json_path.exists():
                continue
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
                data = raw[0] if isinstance(raw, list) else raw
                aid = data.get("article_id")
                if not aid or not data.get("parsed_estimate"):
                    skipped += 1
                    continue
                data["article_id"] = str(aid)
                ops.append(UpdateOne(
                    {"article_id": str(aid)},
                    {"$set": data},
                    upsert=True,
                ))
            except Exception as e:
                print(f"  [WARN] {json_path}: {e}")

    if not ops:
        print("[INFO] 삽입할 데이터가 없습니다.")
        return

    result = col.bulk_write(ops)
    print(f"[OK] upsert 완료")
    print(f"     신규 삽입: {result.upserted_count}개")
    print(f"     갱신:      {result.modified_count}개")
    print(f"[SKIP] parsed_estimate 없음: {skipped}개")
    print(f"[DB] estimate_db.estimate_cases ({col.count_documents({})}개 총 문서)")


if __name__ == "__main__":
    main()
