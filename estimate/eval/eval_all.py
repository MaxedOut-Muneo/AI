"""
eval_all.py — 전체 성능 지표 통합 평가 + 이력 기록

실행:
    python eval_all.py                          # 전체 평가
    python eval_all.py --note "프롬프트 v2"     # 변경사항 메모 포함
    python eval_all.py --history                # 이력만 출력 (평가 없음)
    python eval_all.py --skip-rag               # RAG 평가 제외 (ChromaDB 미구성 시)

평가 항목:
    PHASE 1 — 파싱 자동 평가   (eval_auto.py 로직, 항상 실행 가능)
    PHASE 2 — 골든셋 파싱 정확도 (eval_golden.py 로직, golden_vision.json 필요)
    PHASE 3 — RAG 검색 성능   (eval_rag.py 로직, ChromaDB + rag_testset.json 필요)
    PHASE 4 — 가견적 정확도   (미구현 — 가견적 로직 완성 후 자동 활성화)

결과 저장:
    eval_results/eval_YYYYMMDD_HHMMSS.json  ← 매 실행마다 누적

이력 시각화:
    python eval_all.py --history
    → 각 지표의 시간별 변화를 텍스트 그래프로 출력
"""

import sys
import json
import argparse
import pathlib
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── 경로 설정 ──────────────────────────────────────────
_HERE        = pathlib.Path(__file__).parent
DATA_DIR     = str(_HERE.parent / "estimate_data")
GOLDEN_FILE  = str(_HERE.parent / "docs" / "eval_data" / "golden_vision.json")
TESTSET_FILE = str(_HERE.parent / "docs" / "eval_data" / "rag_testset.json")
RESULTS_DIR  = str(_HERE / "eval_results")

EMBED_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "estimates"

CONSISTENCY_TOLERANCE = 0.20
DUPLICATE_THRESHOLD   = 1.30
AMOUNT_MATCH_TOL      = 0.10


# ══════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════

def sep(title: str):
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print("=" * 58)


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


# ══════════════════════════════════════════════════════
# PHASE 1 — 파싱 자동 평가
# ══════════════════════════════════════════════════════

def run_phase1(records: list[dict]) -> dict:
    sep("PHASE 1 — 파싱 자동 평가")

    total  = len(records)
    parsed = sum(1 for r in records if r.get("parsed_estimate"))
    missing_size = sum(1 for r in records if not r.get("size_pyeong"))

    match = dup = mismatch = no_cost = 0
    for r in records:
        pe = r.get("parsed_estimate")
        if not pe:
            continue
        tc = pe.get("total_cost") or 0
        if not tc:
            no_cost += 1
            continue
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
    coverage_rate    = round(parsed / total * 100, 1) if total else 0
    consistency_rate = round(match / valid * 100, 1) if valid else 0
    duplicate_rate   = round(dup / valid * 100, 1) if valid else 0
    size_miss_rate   = round(missing_size / total * 100, 1) if total else 0

    metrics = {
        "data_count":        total,
        "parsed_count":      parsed,
        "coverage_rate":     coverage_rate,
        "consistency_rate":  consistency_rate,
        "duplicate_rate":    duplicate_rate,
        "size_missing_rate": size_miss_rate,
    }

    goals = {
        "coverage_rate ≥ 80%":      coverage_rate >= 80.0,
        "consistency_rate ≥ 80%":   consistency_rate >= 80.0,
        "duplicate_rate = 0%":      duplicate_rate == 0.0,
        "size_missing_rate ≤ 10%":  size_miss_rate <= 10.0,
    }

    print(f"  데이터 전체:     {total}개")
    print(f"  파싱 완료:       {parsed}개  ({coverage_rate}%)")
    print(f"  내부합산 일치율:  {consistency_rate}%  (검사대상 {valid}건)")
    print(f"  중복집계 의심률:  {duplicate_rate}%")
    print(f"  평수 미추출률:   {size_miss_rate}%")
    print()
    for label, ok in goals.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {label}")

    return {"status": "ok", "metrics": metrics, "goals": goals}


# ══════════════════════════════════════════════════════
# PHASE 2 — 골든셋 파싱 정확도
# ══════════════════════════════════════════════════════

def _calc_parsed_category_totals(line_items: list) -> dict:
    totals = defaultdict(int)
    for item in line_items:
        cat = item.get("category", "").strip()
        amt = item.get("amount") or 0
        if cat and cat != "0":
            totals[cat] += int(amt)
    return dict(totals)


def _match_categories(golden_cats: dict, parsed_cats: dict) -> dict:
    used = set()
    result = {}

    for g in golden_cats:
        if g in parsed_cats and g not in used:
            result[g] = {"parsed_amt": parsed_cats[g], "missing": False}
            used.add(g)

    for g in golden_cats:
        if g in result:
            continue
        g_norm = g.replace(" ", "")
        for p, pa in parsed_cats.items():
            if p in used:
                continue
            if g_norm in p.replace(" ", "") or p.replace(" ", "") in g_norm:
                result[g] = {"parsed_amt": pa, "missing": False}
                used.add(p)
                break

    for g, ga in golden_cats.items():
        if g in result or not ga:
            continue
        cands = [(p, pa) for p, pa in parsed_cats.items()
                 if p not in used and abs(pa - ga) / ga <= AMOUNT_MATCH_TOL]
        if cands:
            best_p, best_a = min(cands, key=lambda x: abs(x[1] - ga))
            result[g] = {"parsed_amt": best_a, "missing": False}
            used.add(best_p)

    for g in golden_cats:
        if g not in result:
            result[g] = {"parsed_amt": None, "missing": True}

    return result


def run_phase2(records: list[dict]) -> dict:
    sep("PHASE 2 — 골든셋 파싱 정확도")

    golden_path = pathlib.Path(GOLDEN_FILE)
    if not golden_path.exists():
        print(f"  [SKIP] 골든셋 파일 없음: {GOLDEN_FILE}")
        print("  → eval_data/golden_vision.json 에 정답 레이블 작성 필요")
        return {"status": "skipped", "reason": "golden_vision.json 없음",
                "metrics": {}, "goals": {}}

    golden_list = [d for d in json.loads(golden_path.read_text(encoding="utf-8"))
                   if d.get("article_id")]
    if not golden_list:
        print("  [SKIP] 골든셋이 비어 있음")
        return {"status": "skipped", "reason": "골든셋 항목 0건",
                "metrics": {}, "goals": {}}

    data_root   = pathlib.Path(DATA_DIR)
    cost_errors = []
    cat_errors  = []
    miss_rates  = []

    for g in golden_list:
        aid = g["article_id"]
        json_path = next(
            (data_root / rd.name / aid / f"{aid}.json"
             for rd in data_root.iterdir()
             if rd.is_dir() and (data_root / rd.name / aid / f"{aid}.json").exists()),
            None,
        )
        if not json_path:
            continue
        record = json.loads(json_path.read_text(encoding="utf-8"))
        pe = record.get("parsed_estimate")
        if not pe:
            continue

        actual = g.get("실제총금액") or 0
        parsed = pe.get("total_cost") or 0
        if actual and parsed:
            cost_errors.append(abs(actual - parsed) / actual * 100)

        golden_cats = g.get("카테고리별_소계") or {}
        if golden_cats:
            parsed_cats = _calc_parsed_category_totals(pe.get("line_items", []))
            matched     = _match_categories(golden_cats, parsed_cats)
            errs = [abs(ga - matched[gn]["parsed_amt"]) / ga * 100
                    for gn, ga in golden_cats.items()
                    if not matched[gn]["missing"] and ga > 0]
            miss = sum(1 for gn in golden_cats if matched[gn]["missing"])
            if errs:
                cat_errors.append(sum(errs) / len(errs))
            miss_rates.append(miss / len(golden_cats) * 100)

    if not cost_errors and not cat_errors:
        print(f"  [SKIP] 평가 가능한 항목 없음 (골든셋 {len(golden_list)}건 모두 미파싱)")
        return {"status": "skipped", "reason": "평가 가능한 파싱 결과 없음",
                "metrics": {}, "goals": {}}

    avg_cost_err = round(sum(cost_errors) / len(cost_errors), 1) if cost_errors else None
    avg_cat_err  = round(sum(cat_errors) / len(cat_errors), 1) if cat_errors else None
    avg_miss     = round(sum(miss_rates) / len(miss_rates), 1) if miss_rates else None

    metrics = {
        "golden_count":       len(golden_list),
        "evaluated_count":    len(cost_errors),
        "avg_cost_error_%":   avg_cost_err,
        "avg_cat_error_%":    avg_cat_err,
        "avg_missing_rate_%": avg_miss,
    }

    goals = {}
    if avg_cost_err is not None:
        goals["avg_cost_error ≤ 5%"]    = avg_cost_err <= 5.0
    if avg_cat_err is not None:
        goals["avg_cat_error ≤ 10%"]    = avg_cat_err <= 10.0
    if avg_miss is not None:
        goals["avg_missing_rate = 0%"]  = avg_miss == 0.0

    print(f"  골든셋:         {len(golden_list)}건 (평가완료 {len(cost_errors)}건)")
    if avg_cost_err is not None:
        print(f"  총금액 오차율:   평균 {avg_cost_err}%")
    if avg_cat_err is not None:
        print(f"  카테고리 오차율: 평균 {avg_cat_err}%")
    if avg_miss is not None:
        print(f"  카테고리 누락률: 평균 {avg_miss}%")
    print()
    for label, ok in goals.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {label}")

    return {"status": "ok", "metrics": metrics, "goals": goals}


# ══════════════════════════════════════════════════════
# PHASE 3 — RAG 검색 성능
# ══════════════════════════════════════════════════════

def run_phase3(top_k: int = 5) -> dict:
    sep("PHASE 3 — RAG 검색 성능")

    testset_path = pathlib.Path(TESTSET_FILE)
    if not testset_path.exists():
        print(f"  [SKIP] 테스트셋 없음: {TESTSET_FILE}")
        print("  → 실행 필요: python split_rag_testset.py --apply")
        return {"status": "skipped", "reason": "rag_testset.json 없음",
                "metrics": {}, "goals": {}}

    try:
        from chromadb.utils import embedding_functions
        from chroma_client import get_chroma_client
    except ImportError as e:
        print(f"  [SKIP] ChromaDB 미설치: {e}")
        return {"status": "skipped", "reason": f"import 실패: {e}",
                "metrics": {}, "goals": {}}

    try:
        test_queries = json.loads(testset_path.read_text(encoding="utf-8"))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client     = get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    except Exception as e:
        print(f"  [SKIP] ChromaDB 연결 실패: {e}")
        print("  → build_rag.py 실행 후 재시도")
        return {"status": "skipped", "reason": f"ChromaDB 연결 실패: {e}",
                "metrics": {}, "goals": {}}

    from eval_rag import is_relevant

    hits            = 0
    rrs             = []
    size_deviations = []
    fallback_1      = 0  # 전체 필터 성공
    fallback_2      = 0  # 2차 폴백
    fallback_3      = 0  # 3차 폴백 (필터 없음)

    for q in test_queries:
        article_id = q["메타"]["article_id"]

        # ChromaDB에서 쿼리 원본 메타 조회 (has_* 필드 포함)
        q_data = collection.get(ids=[article_id], include=["metadatas"])
        q_meta = q_data["metadatas"][0] if q_data["metadatas"] else q["메타"]

        # top_k+1 검색 후 자기 자신 제외
        n = min(top_k + 1, collection.count())
        results  = collection.query(query_texts=[q["query"]], n_results=n,
                                    include=["metadatas"])
        ret_meta = [m for m in results["metadatas"][0]
                    if m.get("article_id") != article_id][:top_k]

        # 관련성 기반 Hit / MRR
        relevant_flags = [is_relevant(q_meta, m) for m in ret_meta]
        if any(relevant_flags):
            hits += 1

        rr = 0.0
        for rank, flag in enumerate(relevant_flags, 1):
            if flag:
                rr = 1.0 / rank
                break
        rrs.append(rr)

        query_size = int(q_meta.get("size_pyeong") or 0)
        if query_size:
            for m in ret_meta:
                rs = int(m.get("size_pyeong") or 0)
                if rs:
                    size_deviations.append(abs(query_size - rs))

        fb = q.get("fallback_stage", 1)
        if fb == 1:
            fallback_1 += 1
        elif fb == 2:
            fallback_2 += 1
        else:
            fallback_3 += 1

    n_q     = len(test_queries)
    hit_atk = round(hits / n_q, 4) if n_q else None
    mrr     = round(sum(rrs) / n_q, 4) if n_q else None
    avg_dev = round(sum(size_deviations) / len(size_deviations), 1) if size_deviations else None

    metrics = {
        "test_query_count": n_q,
        f"hit_at_{top_k}":  hit_atk,
        "mrr":              mrr,
        "avg_size_dev":     avg_dev,
        "fallback_1_count": fallback_1,
        "fallback_2_count": fallback_2,
        "fallback_3_count": fallback_3,
        "fallback_2_rate":  round(fallback_2 / n_q * 100, 1) if n_q else None,
        "fallback_3_rate":  round(fallback_3 / n_q * 100, 1) if n_q else None,
    }

    goals = {}
    if hit_atk is not None:
        goals[f"hit@{top_k} ≥ 0.70"] = hit_atk >= 0.70
    if mrr is not None:
        goals["mrr ≥ 0.60"]           = mrr >= 0.60
    if avg_dev is not None:
        goals["avg_size_dev ≤ 5평"]    = avg_dev <= 5.0

    print(f"  테스트 쿼리:     {n_q}건")
    print(f"  Hit@{top_k}:      {hit_atk}  (목표 ≥ 0.70)")
    print(f"  MRR:            {mrr}  (목표 ≥ 0.60)")
    print(f"  평균 평수 편차:  {avg_dev}평  (목표 ≤ 5평)")
    print(f"  폴백 — 1차성공 {fallback_1}건 / 2차폴백 {fallback_2}건 / 3차폴백 {fallback_3}건")
    print()
    for label, ok in goals.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {label}")

    return {"status": "ok", "metrics": metrics, "goals": goals}


# ══════════════════════════════════════════════════════
# PHASE 4 — 가견적 정확도 (미구현)
# ══════════════════════════════════════════════════════

def run_phase4() -> dict:
    sep("PHASE 4 — 가견적 정확도")

    estimate_testset = pathlib.Path("./eval_data/estimate_testset.json")
    if not estimate_testset.exists():
        print("  [NOT IMPLEMENTED] 가견적 평가 로직 미완성")
        print("  → 가견적 계산 로직 완성 후 eval_estimate.py 구현 예정")
        print("  → 구현 완료 시 이 함수에 자동 연결됩니다")
        return {"status": "not_implemented", "metrics": {}, "goals": {}}

    # 구현 완료 후 여기에 eval_estimate.py 로직 연결
    # from eval_estimate import run_estimate_eval
    # return run_estimate_eval()
    print("  [NOT IMPLEMENTED] estimate_testset.json 존재하지만 평가 로직 미연결")
    return {"status": "not_implemented", "metrics": {}, "goals": {}}


# ══════════════════════════════════════════════════════
# 결과 저장
# ══════════════════════════════════════════════════════

def save_result(result: dict) -> pathlib.Path:
    out_dir = pathlib.Path(RESULTS_DIR)
    out_dir.mkdir(exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{ts}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ══════════════════════════════════════════════════════
# 이력 출력 (텍스트 트렌드 그래프)
# ══════════════════════════════════════════════════════

def _sparkline(values: list, width: int = 20) -> str:
    """수치 리스트를 간단한 텍스트 바 그래프로 변환."""
    clean = [v for v in values if v is not None]
    if not clean:
        return "(데이터 없음)"
    vmin, vmax = min(clean), max(clean)
    bars = "▁▂▃▄▅▆▇█"
    result = []
    for v in values:
        if v is None:
            result.append("·")
        elif vmax == vmin:
            result.append("▄")
        else:
            idx = int((v - vmin) / (vmax - vmin) * (len(bars) - 1))
            result.append(bars[idx])
    return "".join(result)


def show_history():
    results_dir = pathlib.Path(RESULTS_DIR)
    if not results_dir.exists():
        print("이력 없음. eval_all.py 실행 후 생성됩니다.")
        return

    files = sorted(results_dir.glob("eval_*.json"))
    if not files:
        print("이력 없음. eval_all.py 실행 후 생성됩니다.")
        return

    history = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            history.append(data)
        except Exception:
            pass

    sep(f"성능 이력 ({len(history)}회)")

    def extract(records, *keys):
        val = None
        for h in records:
            v = h
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            yield v

    labels_and_paths = [
        ("파싱 보유율(%)",       ("phase1", "metrics", "coverage_rate")),
        ("내부합산 일치율(%)",    ("phase1", "metrics", "consistency_rate")),
        ("중복집계 의심률(%)",    ("phase1", "metrics", "duplicate_rate")),
        ("평수 미추출률(%)",      ("phase1", "metrics", "size_missing_rate")),
        ("총금액 오차율(%)",      ("phase2", "metrics", "avg_cost_error_%")),
        ("카테고리 오차율(%)",    ("phase2", "metrics", "avg_cat_error_%")),
        (f"Hit@5",               ("phase3", "metrics", "hit_at_5")),
        ("MRR",                  ("phase3", "metrics", "mrr")),
        ("평균 평수 편차(평)",    ("phase3", "metrics", "avg_size_dev")),
    ]

    dates = [h.get("timestamp", "?")[:10] for h in history]
    print(f"\n  {'지표':<22} {'최신값':>8}  {'추이 (오래된 순 →)'}")
    print("  " + "─" * 70)

    for label, path in labels_and_paths:
        vals = list(extract(history, *path))
        latest = next((v for v in reversed(vals) if v is not None), None)
        spark  = _sparkline(vals)
        latest_str = f"{latest:.3f}" if isinstance(latest, float) and latest < 10 \
                     else (f"{latest:.1f}" if latest is not None else "  -  ")
        print(f"  {label:<22} {latest_str:>8}  {spark}")

    print(f"\n  측정일: {' | '.join(dates)}")

    # 목표 달성 요약
    sep("최신 측정 — 목표 달성 현황")
    latest_result = history[-1]
    for phase_key in ("phase1", "phase2", "phase3", "phase4"):
        phase = latest_result.get(phase_key, {})
        if phase.get("status") not in ("ok",):
            status = phase.get("status", "unknown")
            reason = phase.get("reason", "")
            label  = {"phase1":"PHASE 1","phase2":"PHASE 2",
                      "phase3":"PHASE 3","phase4":"PHASE 4"}[phase_key]
            print(f"  {label}: [{status}] {reason}")
            continue
        for goal_label, ok in phase.get("goals", {}).items():
            mark = "✅" if ok else "❌"
            phase_label = {"phase1":"P1","phase2":"P2",
                           "phase3":"P3","phase4":"P4"}[phase_key]
            print(f"  {mark} [{phase_label}] {goal_label}")


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--note",     type=str, default="",
                        help="변경사항 메모 (예: '프롬프트 v2')")
    parser.add_argument("--history",  action="store_true",
                        help="이력만 출력 (평가 없음)")
    parser.add_argument("--skip-rag", action="store_true",
                        help="RAG 평가 건너뜀 (ChromaDB 미구성 시)")
    parser.add_argument("--top",      type=int, default=5,
                        help="Hit@K의 K값 (기본: 5)")
    args = parser.parse_args()

    if args.history:
        show_history()
        return

    print(f"\n[eval_all] 통합 성능 평가 시작  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.note:
        print(f"[메모] {args.note}")

    data_root = pathlib.Path(DATA_DIR)
    if not data_root.exists():
        print(f"[ERR] 데이터 폴더 없음: {DATA_DIR}")
        return

    records = load_all_json(data_root)
    print(f"\n총 {len(records)}개 JSON 로드")

    p1 = run_phase1(records)
    p2 = run_phase2(records)
    p3 = run_phase3(top_k=args.top) if not args.skip_rag else {
        "status": "skipped", "reason": "--skip-rag 옵션", "metrics": {}, "goals": {}
    }
    p4 = run_phase4()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "note":      args.note,
        "phase1":    p1,
        "phase2":    p2,
        "phase3":    p3,
        "phase4":    p4,
    }

    out_path = save_result(result)

    sep("저장 완료")
    print(f"  {out_path}")
    print()

    show_history()


if __name__ == "__main__":
    main()
