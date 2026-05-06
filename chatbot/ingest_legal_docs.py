import json
from pathlib import Path

from db import get_sync_collection_from_db
from chatbot.chroma_legal_client import get_legal_collection

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def make_chunk_text(doc: dict, item: dict) -> str:
    keywords = ", ".join(item.get("keywords", []))
    criteria = item.get("criteria", "")
    page = item.get("page", "")

    return f"""
문서유형: {doc.get("source_type")}
문서명: {doc.get("title")}
발행기관: {doc.get("issuer")}
표준번호: {doc.get("standard_no", "")}
제정일: {doc.get("enacted_date", "")}
분류: {doc.get("category")}

항목: {item.get("article_no", "")} {item.get("section", "")}
하자분류: {item.get("defect_category", "")}
페이지: {page}
키워드: {keywords}

내용:
{item.get("text", "")}

판단기준:
{criteria}
""".strip()


def ingest_one_file(json_path: Path):
    if not json_path.exists():
        print(f"[SKIP] 파일 없음: {json_path}")
        return

    mongo_col = get_sync_collection_from_db("chatbot_db", "legal_documents")
    chroma_col = get_legal_collection()

    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    existing = mongo_col.find_one({
        "source_type": doc.get("source_type"),
        "title": doc.get("title")
    })

    if existing:
        mongo_id = str(existing["_id"])
        mongo_col.replace_one({"_id": existing["_id"]}, doc)
        print(f"[MongoDB] 기존 문서 갱신: {doc.get('title')}")
    else:
        inserted = mongo_col.insert_one(doc)
        mongo_id = str(inserted.inserted_id)
        print(f"[MongoDB] 신규 문서 저장: {doc.get('title')}")

    ids = []
    documents = []
    metadatas = []

    for idx, item in enumerate(doc.get("content", [])):
        chunk_text = make_chunk_text(doc, item)

        chunk_id = f"{doc.get('source_type')}_{mongo_id}_{idx}"

        ids.append(chunk_id)
        documents.append(chunk_text)
        metadatas.append({
            "mongo_id": mongo_id,
            "source_type": doc.get("source_type", ""),
            "title": doc.get("title", ""),
            "issuer": doc.get("issuer", ""),
            "category": doc.get("category", ""),
            "article_no": item.get("article_no", ""),
            "section": item.get("section", ""),
            "defect_category": item.get("defect_category", ""),
            "page": item.get("page", ""),
            "keywords": ",".join(item.get("keywords", []))
        })

    if ids:
        chroma_col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[ChromaDB] upsert 완료: {len(ids)} chunks")
    else:
        print(f"[ChromaDB] 저장할 chunk 없음: {doc.get('title')}")

    print(f"[OK] 저장 완료: {doc.get('title')}\n")


def main():
    ingest_one_file(DATA_DIR / "standard_contract.json")
    ingest_one_file(DATA_DIR / "defect_standard.json")


if __name__ == "__main__":
    main()