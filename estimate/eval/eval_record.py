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
_HERE         = pathlib.Path(__file__).parent
DATA_DIR      = str(_HERE.parent / "estimate_data")
COLLECTION    = "estimates"
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"
GOLDEN_FILE   = str(_HERE.parent / "docs" / "eval_data" / "golden_vision.json")
TESTSET_FILE  = str(_HERE.parent / "docs" / "eval_data" / "rag_testset.json")
LOG_FILE      = str(_HERE.parent / "docs" / "eval_data" / "eval_log.csv")

CONSISTENCY_TOLERANCE = 0.20
DUPLICATE_THRESHOLD   = 1.30

LOG_COLUMNS = [
    "date", "data_count", "parsed_count", "coverage_rate",
    "consistency_rate", "duplicate_rate", "size_missing_rate",
    "golden_cost_error", "golden_cat_error",
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


def run_golden_eval(records: list[dict]) -> dict:
    """eval_golden.py 로직으로 총금액 오차율 / 카테고리 오차율 측정."""
    from eval_golden import load_golden, evaluate_one

    golden_path = pathlib.Path(GOLDEN_FILE)
    if not golden_path.exists():
        return {"golden_cost_error": None, "golden_cat_error": None}

    golden_list = load_golden(GOLDEN_FILE)
    if not golden_list:
        return {"golden_cost_error": None, "golden_cat_error": None}

    data_root   = pathlib.Path(DATA_DIR)
    cost_errors = []
    cat_errors  = []

    for g in golden_list:
        r = evaluate_one(g, data_root, verbose=False)
        if not r:
            continue
        if r.get("cost_error_%") is not None:
            cost_errors.append(r["cost_error_%"])
        if r.get("avg_cat_error_%") is not None:
            cat_errors.append(r["avg_cat_error_%"])

    return {
        "golden_cost_error": round(sum(cost_errors) / len(cost_errors), 1) if cost_errors else None,
        "golden_cat_error":  round(sum(cat_errors)  / len(cat_errors),  1) if cat_errors  else None,
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
          f"{'금액오차':>8} {'카테오차':>8} "
          f"{'Hit@5':>7} {'MRR':>7}  변경사항")
    print("─" * 110)

    for r in rows:
        hit      = r.get("rag_hit5", "") if r.get("rag_hit5") not in ("", "None", None) else "  -  "
        mrr      = r.get("rag_mrr", "")  if r.get("rag_mrr")  not in ("", "None", None) else "  -  "
        ge_cost  = r.get("golden_cost_error", "")
        ge_cat   = r.get("golden_cat_error", "")
        ge_cost  = f"{ge_cost}%" if ge_cost not in ("", "None", None) else "   -  "
        ge_cat   = f"{ge_cat}%"  if ge_cat  not in ("", "None", None) else "   -  "
        dup      = r["duplicate_rate"]
        size     = r["size_missing_rate"]

        cons_mark = "✅" if float(r["consistency_rate"] or 0) >= 80 else "❌"
        dup_mark  = "✅" if float(dup or 0) == 0 else "❌"

        print(f"{r['date']:<12} {r['data_count']:>5} {r['parsed_count']:>5} "
              f"{r['coverage_rate']:>5}% "
              f"{r['consistency_rate']:>5}%{cons_mark} "
              f"{dup:>4}%{dup_mark} "
              f"{size:>4}%  "
              f"{str(ge_cost):>8} {str(ge_cat):>8} "
              f"{str(hit):>7} {str(mrr):>7}  {r['note']}")

    print("─" * 110)
    print(f"{'목표':.<12} {'300+':>5} {'80%+':>5} {'80%+':>6} "
          f"{'90%+':>6} {'0%':>5} {'10%-':>5} "
          f"{'≤5%':>8} {'≤10%':>8} "
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

    # 골든셋 평가
    print("\n  골든셋 평가 중...")
    golden = run_golden_eval(records)
    ge_cost = golden["golden_cost_error"]
    ge_cat  = golden["golden_cat_error"]
    print(f"  총금액 오차율:   {f'{ge_cost}%' if ge_cost is not None else '-'}")
    print(f"  카테고리 오차율: {f'{ge_cat}%'  if ge_cat  is not None else '-'}")

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
        **golden,
        **rag,
    }
    append_log(row)
    print(f"\n[OK] {LOG_FILE} 에 기록 완료")

    # 이력 출력
    show_log()


if __name__ == "__main__":
    main()
