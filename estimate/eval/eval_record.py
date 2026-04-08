"""
eval_record.py — 성능 평가 결과를 eval_log.csv에 자동 기록

실행:
    python eval_record.py                        # 자동 평가 + 기록
    python eval_record.py --note "프롬프트 v2"   # 변경사항 메모 포함
    python eval_record.py --show                 # 기록 이력 출력

누적 기록 파일: eval_log.csv
  date, data_count, parsed_count, coverage_rate, consistency_rate,
  duplicate_rate, size_missing_rate, rag_hit5, rag_mrr, note

사용 흐름:
    크롤링/재파싱/DB 재구축 등 변경 후마다 실행
    → eval_log.csv에 한 줄씩 누적
    → 나중에 엑셀/그래프로 시각화
"""

import sys
import csv
import json
import argparse
import pathlib
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── 설정 ─────────────────────────────────────────────
DATA_DIR      = "./estimate_data"
COLLECTION    = "estimates"
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
TESTSET_FILE  = "./eval_data/rag_testset.json"
LOG_FILE      = "./eval_data/eval_log.csv"

CONSISTENCY_TOLERANCE = 0.20
DUPLICATE_THRESHOLD   = 1.30

LOG_COLUMNS = [
    "date", "data_count", "parsed_count", "coverage_rate",
    "consistency_rate", "duplicate_rate", "size_missing_rate",
    "rag_hit5", "rag_mrr", "note",
]


# ══════════════════════════════════════════════════════
# 자동 평가 함수 (eval_auto.py 핵심 로직 재사용)
# ══════════════════════════════════════════════════════

def load_all_json(data_root: pathlib.Path) -> list[dict]:
    records = []
    for region_dir in sorted(data_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue
            json_path = article_dir / f"{article_dir.name}.json"
            if not json_path.exists():
                continue
            try:
                records.append(json.loads(json_path.read_text(encoding="utf-8")))
            except Exception:
                pass
    return records


def run_auto_eval(records: list[dict]) -> dict:
    total  = len(records)
    parsed = sum(1 for r in records if r.get("parsed_estimate"))
    missing_size = sum(1 for r in records if not r.get("size_pyeong"))

    # 내부합산 일치율 / 중복집계
    match = dup = mismatch = 0
    for r in records:
        pe = r.get("parsed_estimate")
        if not pe or not pe.get("total_cost"):
            continue
        tc = int(pe["total_cost"])
        ls = sum(
            int(i.get("amount") or 0)
            for i in pe.get("line_items", [])
            if isinstance(i.get("amount"), (int, float))
        )
        ratio = ls / tc
        err   = abs(ls - tc) / tc
        if ratio >= DUPLICATE_THRESHOLD:
            dup += 1
        elif err <= CONSISTENCY_TOLERANCE:
            match += 1
        else:
            mismatch += 1

    valid = match + dup + mismatch
    return {
        "data_count":         total,
        "parsed_count":       parsed,
        "coverage_rate":      round(parsed / total * 100, 1) if total else 0,
        "consistency_rate":   round(match / valid * 100, 1) if valid else 0,
        "duplicate_rate":     round(dup / valid * 100, 1) if valid else 0,
        "size_missing_rate":  round(missing_size / total * 100, 1) if total else 0,
    }


def run_rag_eval(top_k: int = 5) -> dict:
    """rag_testset.json이 있으면 Hit@K / MRR 측정. 없으면 None 반환."""
    testset_path = pathlib.Path(TESTSET_FILE)
    if not testset_path.exists():
        return {"rag_hit5": None, "rag_mrr": None}

    try:
        import chromadb
        from chromadb.utils import embedding_functions
        from chroma_client import get_chroma_client

        test_queries = json.loads(testset_path.read_text(encoding="utf-8"))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client     = get_chroma_client()
        collection = client.get_collection(name=COLLECTION, embedding_function=ef)

        hits = 0
        rrs  = []
        for q in test_queries:
            n = min(top_k, collection.count())
            results   = collection.query(query_texts=[q["query"]], n_results=n,
                                         include=["metadatas"])
            ret_ids   = [m["article_id"] for m in results["metadatas"][0]]
            ans_ids   = set(q["정답_article_ids"])
            if ans_ids & set(ret_ids):
                hits += 1
            rr = 0.0
            for rank, aid in enumerate(ret_ids, 1):
                if aid in ans_ids:
                    rr = 1.0 / rank
                    break
            rrs.append(rr)

        n_q = len(test_queries)
        return {
            "rag_hit5": round(hits / n_q, 4) if n_q else None,
            "rag_mrr":  round(sum(rrs) / n_q, 4) if n_q else None,
        }
    except Exception as e:
        print(f"  [WARN] RAG 평가 실패: {e}")
        return {"rag_hit5": None, "rag_mrr": None}


# ══════════════════════════════════════════════════════
# CSV 기록
# ══════════════════════════════════════════════════════

def append_log(row: dict):
    log_path  = pathlib.Path(LOG_FILE)
    is_new    = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def show_log():
    log_path = pathlib.Path(LOG_FILE)
    if not log_path.exists():
        print("기록 없음. eval_record.py 실행 후 생성됩니다.")
        return

    with open(log_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("기록 없음.")
        return

    # 헤더
    print(f"\n{'날짜':<12} {'데이터':>5} {'파싱':>5} {'보유율':>6} "
          f"{'일치율':>6} {'중복':>5} {'평수':>5} "
          f"{'Hit@5':>7} {'MRR':>7}  변경사항")
    print("─" * 90)

    for r in rows:
        hit  = r["rag_hit5"] if r["rag_hit5"] not in ("", "None") else "  -  "
        mrr  = r["rag_mrr"]  if r["rag_mrr"]  not in ("", "None") else "  -  "
        dup  = r["duplicate_rate"]
        size = r["size_missing_rate"]

        # 목표치 미달 강조
        cons_mark = "✅" if float(r["consistency_rate"] or 0) >= 80 else "❌"
        dup_mark  = "✅" if float(dup or 0) == 0 else "❌"

        print(f"{r['date']:<12} {r['data_count']:>5} {r['parsed_count']:>5} "
              f"{r['coverage_rate']:>5}% "
              f"{r['consistency_rate']:>5}%{cons_mark} "
              f"{dup:>4}%{dup_mark} "
              f"{size:>4}%  "
              f"{str(hit):>7} {str(mrr):>7}  {r['note']}")

    # 목표치 기준선
    print("─" * 90)
    print(f"{'목표':.<12} {'300+':>5} {'80%+':>5} {'80%+':>6} "
          f"{'90%+':>6} {'0%':>5} {'10%-':>5} "
          f"{'0.70+':>7} {'0.60+':>7}")


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=str, default="",
                        help="변경사항 메모 (예: '프롬프트 v2', '데이터 200개 추가')")
    parser.add_argument("--show", action="store_true",
                        help="기록 이력만 출력 (평가 없음)")
    args = parser.parse_args()

    if args.show:
        show_log()
        return

    print("성능 평가 시작...\n")

    # 자동 평가
    data_root = pathlib.Path(DATA_DIR)
    records   = load_all_json(data_root)
    auto      = run_auto_eval(records)

    print(f"  데이터:     {auto['data_count']}개 (파싱완료 {auto['parsed_count']}개)")
    print(f"  보유율:     {auto['coverage_rate']}%")
    print(f"  일치율:     {auto['consistency_rate']}%")
    print(f"  중복집계:   {auto['duplicate_rate']}%")
    print(f"  평수미추출: {auto['size_missing_rate']}%")

    # RAG 평가
    print("\n  RAG 평가 중...")
    rag = run_rag_eval()
    print(f"  Hit@5: {rag['rag_hit5']}")
    print(f"  MRR:   {rag['rag_mrr']}")

    # CSV 기록
    row = {
        "date":               datetime.now().strftime("%Y-%m-%d"),
        "note":               args.note,
        **auto,
        **rag,
    }
    append_log(row)
    print(f"\n[OK] {LOG_FILE} 에 기록 완료")

    # 이력 출력
    show_log()


if __name__ == "__main__":
    main()
