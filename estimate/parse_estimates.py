"""
parse_estimates.py — 견적서 이미지를 Claude Vision API로 파싱

실행 (실시간):
    python parse_estimates.py [--reparse] [--id ARTICLE_ID]

배치 모드 (50% 비용 절감):
    python parse_estimates.py --batch-submit   # 배치 제출
    python parse_estimates.py --batch-status   # 진행 상황 확인
    python parse_estimates.py --batch-apply    # 결과 적용

동작:
    - estimate_data/{지역}/{article_id}/ 폴더 순회 (지역별 구조 지원)
    - 이미지 파일을 Claude Vision API로 파싱
    - 견적서 항목(라인 아이템, 총액)을 JSON으로 추출
    - 각 폴더의 {article_id}.json에 "parsed_estimate" 필드로 병합
    - 이미 파싱된 항목은 skip (재실행 안전)

사전 준비:
    ANTHROPIC_API_KEY 환경변수 설정 (또는 .env 파일)
"""

from __future__ import annotations

import re
import io
import base64
import json
import time
import pathlib
from collections import defaultdict

import anthropic
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════

DATA_DIR         = "./estimate_data"
BATCH_STATE_FILE = "./batch_state.json"
MODEL            = "claude-sonnet-4-6"

SLEEP_BETWEEN  = 1.0
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

MEDIA_TYPE_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
}

# 이미지 분할 설정
SPLIT_HEIGHT_THRESHOLD = 3000   # 이 픽셀 이상이면 분할
CHUNK_HEIGHT            = 2000  # 청크 1개 높이
CHUNK_OVERLAP           = 200   # 청크 간 겹침 (행 잘림 방지)

# 파싱 전 리사이징 (토큰 절약)
MAX_PARSE_WIDTH = 1400          # 파싱용 최대 너비 (px)

PARSE_PROMPT = """이 이미지가 인테리어 공사 견적서인지 판단하고, 맞다면 아래 JSON 형식으로만 출력해줘.

견적서가 아닌 경우 (평면도, 현장사진, 로고, 배너 등):
{"is_estimate": false}

견적서인 경우:
{
  "is_estimate": true,
  "total_cost": 21800000,
  "line_items": [
    {
      "code": "227",
      "category": "창호공사",
      "description": "ABS도어폼(일반)",
      "unit_price": 150000,
      "quantity": 5.0,
      "unit": "짝",
      "amount": 750000
    }
  ]
}

━━━ 규칙 1: 어떤 테이블을 파싱할 것인가 ━━━

견적서에는 두 종류의 테이블이 있을 수 있다:
  [요약] 공사 구분별 합계만 나열 (수량·단가 없음, 행 10~20개)
  [상세] 개별 품목마다 수량·단가·금액 있음 (행 30개 이상)

규칙:
  → 상세 테이블이 보이면 반드시 상세 테이블만 파싱한다.
  → 요약 테이블만 있을 때만 요약 테이블을 파싱한다.
  → 상세가 있는데 요약만 파싱하면 오류다.

━━━ 규칙 2: 집계 행은 반드시 제외 ━━━

아래 행들은 line_items에 절대 포함하지 않는다:

  ✗ description이 "소계", "합계", "총계", "공사비합계", "계", "공사합계"인 행
  ✗ description이 카테고리명과 동일한 행
      예: category="창호공사", description="창호공사" → 제외
  ✗ 품명 칸이 비어 있고 금액만 있는 행
  ✗ 상세 테이블 내 카테고리 구분 소계 행
      예: "철거공사 ───────── 4,200,000" → 제외

포함하는 행:
  ✓ 구체적인 품명이 있는 개별 항목
      예: "기존 벽체 철거 및 폐기물 처리 | 1식 | 300,000"
      예: "ABS도어폼(일반) | 5짝 | 150,000 | 750,000"

판별 기준: description에 구체적인 재료명·작업명이 있으면 포함, 공사 분류명만 있으면 제외.

━━━ 규칙 3: total_cost 선택 방법 ━━━

우선순위:
  1순위: "공사비합계" 또는 "합계" 행의 금액 (부가세 제외 금액)
  2순위: "부가세 포함 합계"만 있으면 → 그 값 / 1.1 (int 변환)
  3순위: 표에서 가장 큰 합계 행의 금액

중요:
  - "부가세" 또는 "VAT" 행의 금액은 total_cost에 더하지 않는다.
  - 부가세가 별도 표시된 경우 → total_cost = 부가세 제외 합계

━━━ 규칙 4: amount와 unit_price 구분 ━━━

열 순서: 코드 | 품명 | 규격 | 수량 | 단위 | 단가 | 금액
  amount    = 금액 열 = 단가 × 수량 (행 전체 청구금액)
  unit_price = 단가 열 = 단위당 가격 (항상 amount 이하)

열이 뒤바뀐 경우 자동 수정:
  unit_price > amount 이고 amount × quantity ≈ unit_price 이면
  → unit_price와 amount를 교환한다.

예시:
  단가=3,500 / 수량=28 / 금액=98,000  →  unit_price=3500, amount=98000  (올바름)
  단가=98,000 / 수량=28 / 금액=3,500  →  뒤바뀜 → unit_price=3500, amount=98000 으로 교환

━━━ 규칙 5: 이미지가 잘린 경우 ━━━

이미지 하단이 잘려 마지막 행이 불완전하게 보이면:
  - 완전히 보이는 행만 파싱한다 (잘린 행은 제외).
  - total_cost는 이미지에 명시된 합계 값만 사용한다 (추정 금지).

━━━ 규칙 6: 파싱 후 자기검증 ━━━

파싱 완료 후 반드시 아래 두 가지를 확인한다.

1. 합계 검증
   sum(line_items[*].amount) 와 total_cost의 차이가 10% 이상이면
   → 이미지를 처음부터 다시 검토하여 누락된 행을 찾아 추가한다.
   → 재검토 후에도 차이가 나면 이미지에 명시된 합계 값을 그대로 total_cost로 쓴다.
   → sum을 total_cost로 덮어쓰지 않는다.

2. 단가 이상값 검증
   unit_price < 1000 이고 quantity >= 1000 인 항목은
   → unit_price와 quantity가 뒤바뀐 것으로 판단하고 교환한다.
   → 교환 후 unit_price × quantity = amount 인지 확인한다.
   예: unit_price=7, quantity=22000, amount=154000
     → unit_price=22000, quantity=7, amount=154000 으로 교환

━━━ 규칙 7: 카테고리명 표준화 ━━━

아래 표준 카테고리명을 우선 사용한다:

  이미지 표기          → 표준 category
  ─────────────────────────────────────
  ABS도어, 도어, 문공사, 목문, 목문틀, 현관문  → 도어공사
  샷시, 새시, 창문      → 창호공사
  발코니확장, 확장       → 확장공사
  필름, 시트, 랩핑      → 시트공사
  수전, 위생, 욕실설비   → 수전공사
  도기, 양변기, 세면대   → 도기공사

  세내수복공사, 수복공사    → 세내수복공사 (원본 명칭 유지)

  단, 이미지에 명시된 공사 구분명이 위 표기와 다르면 이미지 표기를 그대로 사용한다.

━━━ 출력 규칙 ━━━

- category: 이미지에서 해당 항목이 속한 공사 구분명 (예: 창호공사, 타일공사)
- 모든 금액·단가는 int (원 단위, 쉼표·원화기호 제거)
- quantity: 이미지에 표시된 수량 (없으면 1.0)
- unit_price: 이미지에 표시된 단가 (없으면 0)
- code, unit 없으면 빈 문자열 ""
- amount가 0이거나 비어 있는 항목은 제외
- JSON만 출력 (설명·마크다운 코드블록·주석 없음)
"""

# ══════════════════════════════════════════════════════
# 이미지 유틸
# ══════════════════════════════════════════════════════

def _split_image_vertically(img: Image.Image) -> list[Image.Image]:
    """세로로 긴 이미지를 CHUNK_HEIGHT 크기로 분할 (CHUNK_OVERLAP 겹침)."""
    w, h = img.size
    chunks = []
    top = 0
    while top < h:
        bottom = min(top + CHUNK_HEIGHT, h)
        chunks.append(img.crop((0, top, w, bottom)))
        if bottom == h:
            break
        top += CHUNK_HEIGHT - CHUNK_OVERLAP
    return chunks


def _resize_for_parse(img: Image.Image) -> Image.Image:
    """MAX_PARSE_WIDTH를 초과하면 비율 유지하며 축소 (토큰 절약)."""
    w, h = img.size
    if w <= MAX_PARSE_WIDTH:
        return img
    new_h = int(h * MAX_PARSE_WIDTH / w)
    return img.resize((MAX_PARSE_WIDTH, new_h), Image.LANCZOS)


def _image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _prepare_chunks(image_path: pathlib.Path) -> list[bytes]:
    """이미지를 로드·리사이즈·분할하여 PNG bytes 리스트 반환."""
    ext = image_path.suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return []
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"  [WARN] 손상된 이미지 삭제: {image_path.name} ({e})")
        image_path.unlink(missing_ok=True)
        return []
    img = _resize_for_parse(img)
    _, h = img.size
    if h <= SPLIT_HEIGHT_THRESHOLD:
        return [_image_to_bytes(img)]
    return [_image_to_bytes(c) for c in _split_image_vertically(img)]


# ══════════════════════════════════════════════════════
# JSON 파싱 / 후처리
# ══════════════════════════════════════════════════════

def _parse_raw(text: str) -> dict:
    """API 응답 텍스트에서 JSON 객체 추출."""
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("JSON 객체를 찾을 수 없음", raw, 0)
    obj, _ = json.JSONDecoder().raw_decode(raw, start)
    return obj


def _fix_column_swap(item: dict) -> dict:
    """unit_price 파싱 오류 자동 수정. 두 가지 패턴 처리:

    Case 1 — 열 뒤바뀜:
        unit_price > amount 이고 amount × qty ≈ unit_price
        → unit_price와 amount 교환

    Case 2 — unit_price에 합계가 들어간 경우:
        unit_price == amount 이고 qty > 1
        → unit_price = amount / qty (amount는 유지)
    """
    up  = item.get("unit_price") or 0
    qty = item.get("quantity")   or 1.0
    amt = item.get("amount")     or 0

    if up <= 0 or amt <= 0 or qty <= 1:
        return item

    # Case 1: amount × quantity 가 unit_price 와 10% 이내 → 열 뒤바뀜
    if up > amt:
        expected = amt * qty
        if abs(expected - up) / up < 0.10:
            item = dict(item)
            item["unit_price"] = amt
            item["amount"]     = up
        return item

    # Case 2: unit_price == amount (합계가 단가 열에 들어간 경우)
    if up == amt:
        per_unit = amt / qty
        if abs(per_unit - round(per_unit)) < 0.5:
            item = dict(item)
            item["unit_price"] = int(round(per_unit))
        return item

    # Case 3: unit_price < 1000 이고 quantity >= 1000 → 열 뒤바뀜 의심
    # 예: unit_price=7, quantity=22000, amount=154000
    #   → 7 × 22000 = 154000 이지만 단가 7원은 비현실적 → 교환
    if up < 1000 and qty >= 1000 and amt > 0:
        if abs(int(qty) * up - amt) / amt < 0.01:
            item = dict(item)
            item["unit_price"] = int(qty)
            item["quantity"]   = float(up)

    return item


def _chunk_dedup_key(item: dict) -> tuple:
    """청크 중복 제거용 키.

    OCR 오차(× vs *, 공백 차이 등)로 description이 조금씩 달라지는 문제를 해결하기 위해
    (category, amount, unit_price) 조합을 사용한다.
    code가 있으면 추가해 더 정밀하게 구분.
    """
    code      = item.get("code", "")
    cat       = item.get("category", "")
    amt       = int(item.get("amount") or 0)
    unit_p    = int(item.get("unit_price") or 0)
    desc_pre  = item.get("description", "")[:4]   # 앞 4자만 (OCR 오차 허용)

    if code:
        return (code, cat, amt, unit_p)
    return (cat, amt, unit_p, desc_pre)


def _merge_chunk_results(chunk_results: list[dict]) -> dict:
    """청크별 파싱 결과 병합 — line_items 합치고 (category, amount, unit_price)로 중복 제거."""
    seen = set()
    all_items = []
    max_total = 0

    for r in chunk_results:
        if not r.get("is_estimate"):
            continue
        total = int(r.get("total_cost") or 0)
        if total > max_total:
            max_total = total
        for item in r.get("line_items", []):
            if not item.get("amount"):
                continue
            key = _chunk_dedup_key(item)
            if key not in seen:
                seen.add(key)
                all_items.append(item)

    if not all_items:
        return {"is_estimate": False}
    all_items = [_fix_column_swap(it) for it in all_items]
    all_items = _remove_aggregate_items(all_items, max_total)
    result = {"is_estimate": True, "total_cost": max_total, "line_items": all_items}
    return _add_consistency_warning(result)


# 집계 행으로 판단할 description 키워드
_AGGREGATE_KEYWORDS = {"합계", "소계", "총계", "공사비합계", "공사합계", "계", "합 계", "소 계"}


def _remove_aggregate_items(line_items: list[dict], total_cost: int = 0) -> list[dict]:
    """파싱 결과에서 집계 행(소계·합계·카테고리 소계)을 제거.

    1단계: description 키워드 매칭 제거
    2단계: description == category 인 행 제거
    3단계: line_sum > total_cost × 1.1 이면 카테고리 내 소계 행 탐지 제거
    """
    cleaned = []
    for item in line_items:
        desc = (item.get("description") or "").strip()
        cat  = (item.get("category")    or "").strip()
        desc_norm = desc.replace(" ", "")
        cat_norm  = cat.replace(" ", "")

        # 1단계: 키워드 매칭
        if desc_norm in _AGGREGATE_KEYWORDS or any(kw in desc_norm for kw in {"합계", "소계", "총계"}):
            continue

        # 2단계: description이 카테고리명과 동일
        if cat_norm and desc_norm and desc_norm == cat_norm:
            continue
        if cat_norm and desc_norm and desc_norm == cat_norm + "합계":
            continue

        cleaned.append(item)

    # 3단계: 여전히 line_sum > total_cost × 1.2 이면 카테고리 내 소계 탐지
    if total_cost > 0:
        line_sum = sum(item.get("amount") or 0 for item in cleaned)
        if line_sum > total_cost * 1.2:
            cleaned = _remove_category_subtotals(cleaned)

    return cleaned


def _remove_category_subtotals(line_items: list[dict]) -> list[dict]:
    """같은 category 내에서 amount ≈ 다른 항목 합인 행(카테고리 소계)을 제거."""
    from collections import defaultdict

    # 인덱스 유지하며 카테고리별 그룹화
    cat_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, item in enumerate(line_items):
        cat_groups[item.get("category", "")].append((idx, item))

    remove_idxs: set[int] = set()
    for cat, idx_items in cat_groups.items():
        if len(idx_items) < 3:   # 3개 미만은 오탐 위험 — 건드리지 않음
            continue
        amounts = [(idx, item.get("amount") or 0) for idx, item in idx_items]
        for i, (idx, amt) in enumerate(amounts):
            if amt == 0:
                continue
            other_sum = sum(a for j, (_, a) in enumerate(amounts) if j != i)
            if other_sum > 0 and abs(amt - other_sum) / max(amt, other_sum) <= 0.05:
                remove_idxs.add(idx)

    return [item for i, item in enumerate(line_items) if i not in remove_idxs]


def _add_consistency_warning(result: dict) -> dict:
    """line_sum과 total_cost 차이가 10% 이상이면 _parse_warning 필드 추가."""
    total = int(result.get("total_cost") or 0)
    if total == 0:
        return result
    line_sum = sum(
        int(item.get("amount") or 0)
        for item in result.get("line_items", [])
        if isinstance(item.get("amount"), (int, float))
    )
    error_rate = abs(line_sum - total) / total
    if error_rate > 0.20:
        result = dict(result)
        result["_parse_warning"] = (
            f"합계 불일치: items={line_sum:,} total={total:,} "
            f"오차={error_rate*100:.1f}%"
        )
    return result


def _calc_consistency(r: dict) -> float:
    """line_items 합계와 total_cost의 오차율 (0.0 = 완전 일치, inf = 측정불가)."""
    total = int(r.get("total_cost") or 0)
    if total == 0:
        return float("inf")
    line_sum = sum(
        item.get("amount") or 0
        for item in r.get("line_items", [])
        if isinstance(item.get("amount"), (int, float))
    )
    return abs(line_sum - total) / total


def merge_parsed_results(results: list[dict]) -> dict:
    """여러 이미지 파싱 결과를 하나의 parsed_estimate로 병합.

    전략:
    - 총금액이 서로 다른 이미지가 섞여 있으면 독립된 견적서로 판단 →
      내부 일관성(line_sum ≈ total_cost)이 가장 높은 단일 이미지만 사용
    - 총금액이 같거나 하나에만 있으면 동일 견적서의 여러 페이지 →
      중복 제거 병합
    """
    estimate_results = [r for r in results if r.get("is_estimate")]
    if not estimate_results:
        return {}

    if len(estimate_results) == 1:
        r = estimate_results[0]
        total = int(r.get("total_cost") or 0)
        items = [_fix_column_swap(it) for it in r.get("line_items", []) if it.get("amount")]
        items = _remove_aggregate_items(items, total)
        return {"total_cost": total, "line_items": items}

    # 총금액 집합 (0 제외)
    totals = {int(r.get("total_cost") or 0) for r in estimate_results}
    totals.discard(0)

    if len(totals) >= 2:
        # 서로 다른 총금액 → 독립된 견적서 혼재 → 가장 일관성 높은 것만 선택
        best = min(estimate_results, key=_calc_consistency)
        best_total = int(best.get("total_cost") or 0)
        items = [_fix_column_swap(it) for it in best.get("line_items", []) if it.get("amount")]
        items = _remove_aggregate_items(items, best_total)
        print(f"        [다중이미지] 총금액 불일치({sorted(totals)}) -> "
              f"일관성 최고 이미지 선택 (total_cost={best_total:,})")
        result = {"total_cost": best_total, "line_items": items}
        return _add_consistency_warning(result)

    # 총금액이 같거나 하나에만 있는 경우 → 중복 제거 병합
    seen = set()
    all_items = []
    max_total = 0
    for r in estimate_results:
        total = int(r.get("total_cost") or 0)
        if total > max_total:
            max_total = total
        for item in r.get("line_items", []):
            if not item.get("amount"):
                continue
            desc_prefix = item.get("description", "")[:8]
            key = (item.get("code", ""), desc_prefix)
            if key not in seen:
                seen.add(key)
                all_items.append(item)

    if not all_items:
        return {}
    all_items = [_fix_column_swap(it) for it in all_items]
    all_items = _remove_aggregate_items(all_items, max_total)
    result = {"total_cost": max_total, "line_items": all_items}
    return _add_consistency_warning(result)


# ══════════════════════════════════════════════════════
# API 클라이언트
# ══════════════════════════════════════════════════════

client = anthropic.Anthropic()


def _build_api_params(image_bytes: bytes) -> dict:
    """배치·실시간 공용 API params 생성 (PNG 고정)."""
    return {
        "model": MODEL,
        "max_tokens": 16384,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                    },
                },
                {"type": "text", "text": PARSE_PROMPT},
            ],
        }],
    }


# ══════════════════════════════════════════════════════
# 실시간 모드
# ══════════════════════════════════════════════════════

def _call_vision_api(image_bytes: bytes) -> dict:
    """PNG bytes로 Vision API 호출. 파싱된 dict 반환."""
    response = client.messages.create(**_build_api_params(image_bytes))
    return _parse_raw(response.content[0].text)


def parse_image(image_path: pathlib.Path) -> dict | None:
    """이미지 1장을 Claude Vision API로 파싱. 긴 이미지는 자동 분할. 실패 시 None 반환."""
    chunks = _prepare_chunks(image_path)
    if not chunks:
        return None

    if len(chunks) == 1:
        return _call_vision_api(chunks[0])

    print(f"        >> {len(chunks)}개 청크로 분할")
    chunk_results = []
    for idx, chunk_bytes in enumerate(chunks):
        try:
            result = _call_vision_api(chunk_bytes)
            items = len(result.get("line_items", []))
            print(f"        청크 {idx+1}/{len(chunks)}: {items}항목")
            chunk_results.append(result)
        except Exception as e:
            print(f"        [ERR] 청크 {idx+1} 파싱 실패: {e}")
        time.sleep(SLEEP_BETWEEN)

    return _merge_chunk_results(chunk_results)


def process_article(article_dir: pathlib.Path, reparse: bool = False) -> bool:
    json_path = article_dir / f"{article_dir.name}.json"
    if not json_path.exists():
        print(f"  [WARN]  JSON 없음: {article_dir.name} -> skip")
        return False

    data = json.loads(json_path.read_text(encoding="utf-8"))

    if data.get("parsed_estimate") and not reparse:
        print(f"  [SKIP]  이미 파싱됨: {article_dir.name}")
        return True

    image_files = sorted([
        f for f in article_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    ])

    if not image_files:
        print(f"  이미지 없음: {article_dir.name}")
        return False

    print(f"  [PARSE] 파싱 중: {article_dir.name} ({len(image_files)}장)")

    parsed_results = []
    for img_path in image_files:
        try:
            result = parse_image(img_path)
            if result is None:
                continue
            flag = "O 견적서" if result.get("is_estimate") else "X 비견적서"
            items_count = len(result.get("line_items", []))
            print(f"      {flag}  {img_path.name}" +
                  (f" ({items_count}항목)" if result.get("is_estimate") else ""))
            parsed_results.append(result)
        except json.JSONDecodeError as e:
            print(f"      [ERR] JSON 파싱 실패 {img_path.name}: {e}")
        except Exception as e:
            print(f"      [ERR] API 오류 {img_path.name}: {e}")
        time.sleep(SLEEP_BETWEEN)

    merged = merge_parsed_results(parsed_results)
    if merged:
        data["parsed_estimate"] = merged
        json_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"      [SAVE] 저장 완료: "
              f"항목 {len(merged['line_items'])}개, "
              f"총액 {merged['total_cost']:,}원")
        if merged.get("_parse_warning"):
            print(f"      [WARN] {merged['_parse_warning']} -> 재파싱 권장")
    else:
        print(f"      [WARN]  견적서 이미지 없음 - parsed_estimate 미저장")

    return True


# ══════════════════════════════════════════════════════
# 배치 모드
# ══════════════════════════════════════════════════════

def _collect_batch_requests(
    article_dirs: list[pathlib.Path],
    reparse: bool = False,
) -> tuple[list[dict], dict]:
    """
    배치 요청 목록 및 메타데이터 생성.
    custom_id 형식: "{article_id}__{img_idx}__{chunk_idx}"

    Returns:
        requests : Batch API용 요청 리스트
        meta     : custom_id → {article_id, img_idx, chunk_idx} 매핑
    """
    requests = []
    meta = {}

    for article_dir in article_dirs:
        json_path = article_dir / f"{article_dir.name}.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("parsed_estimate") and not reparse:
            continue

        image_files = sorted([
            f for f in article_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ])

        for img_idx, img_path in enumerate(image_files):
            chunks = _prepare_chunks(img_path)
            for chunk_idx, chunk_bytes in enumerate(chunks):
                custom_id = f"{article_dir.name}__{img_idx}__{chunk_idx}"
                requests.append({
                    "custom_id": custom_id,
                    "params": _build_api_params(chunk_bytes),
                })
                meta[custom_id] = {
                    "article_id": article_dir.name,
                    "img_idx": img_idx,
                    "chunk_idx": chunk_idx,
                }

    return requests, meta


BATCH_CHUNK_SIZE = 150  # 한 번에 제출할 최대 요청 수 (256MB 제한 대응)


def cmd_batch_submit(article_dirs: list[pathlib.Path], reparse: bool = False) -> None:
    """배치 생성 및 제출. 요청이 많으면 BATCH_CHUNK_SIZE 단위로 나눠 제출.
    batch_state.json에 배치 목록 저장."""
    print("이미지 로드 및 청크 생성 중...")
    all_requests, all_meta = _collect_batch_requests(article_dirs, reparse)

    if not all_requests:
        print("[WARN] 제출할 요청이 없음 (이미 모두 파싱됨)")
        return

    # BATCH_CHUNK_SIZE 단위로 분할
    chunks = [
        all_requests[i:i + BATCH_CHUNK_SIZE]
        for i in range(0, len(all_requests), BATCH_CHUNK_SIZE)
    ]
    print(f"총 {len(all_requests)}개 요청 → {len(chunks)}개 배치로 분할 제출\n")

    batches = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_ids = {r["custom_id"] for r in chunk}
        chunk_meta = {k: v for k, v in all_meta.items() if k in chunk_ids}

        print(f"  배치 {idx}/{len(chunks)} 제출 중 ({len(chunk)}개 요청)...")
        batch = client.messages.batches.create(requests=chunk)
        batches.append({
            "batch_id":       batch.id,
            "submitted_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_requests": len(chunk),
            "meta":           chunk_meta,
        })
        print(f"  배치 {idx} ID: {batch.id}")
        time.sleep(1.0)

    state = {"batches": batches}
    pathlib.Path(BATCH_STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n[OK] 전체 {len(batches)}개 배치 제출 완료 ({len(all_requests)}개 요청)")
    print(f"     상태 파일 : {BATCH_STATE_FILE}")
    print(f"\n결과 확인:")
    print(f"  python parse_estimates.py --batch-status")
    print(f"완료 후 결과 적용:")
    print(f"  python parse_estimates.py --batch-apply")


def cmd_batch_status() -> None:
    """배치 현재 상태 출력."""
    state_path = pathlib.Path(BATCH_STATE_FILE)
    if not state_path.exists():
        print(f"[ERR] 상태 파일 없음: {BATCH_STATE_FILE}")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_list = state.get("batches") or [state]  # 구버전 호환

    total_req = sum(b["total_requests"] for b in batch_list)
    total_ok = total_err = total_proc = 0

    for b in batch_list:
        batch = client.messages.batches.retrieve(b["batch_id"])
        counts = batch.request_counts
        total_ok   += counts.succeeded
        total_err  += counts.errored
        total_proc += counts.processing
        status_str = batch.processing_status
        print(f"배치 ID     : {b['batch_id']}")
        print(f"제출 시각   : {b['submitted_at']}")
        print(f"상태        : {status_str}  ({counts.succeeded}성공 / {counts.errored}오류 / {counts.processing}처리중)")
        print()

    if len(batch_list) > 1:
        print(f"전체 합계   : {total_req}개 요청 / {total_ok}성공 / {total_err}오류 / {total_proc}처리중")


def cmd_batch_apply(article_dirs: list[pathlib.Path]) -> None:
    """batch_state.json을 읽어 결과를 각 JSON에 저장."""
    state_path = pathlib.Path(BATCH_STATE_FILE)
    if not state_path.exists():
        print(f"[ERR] 상태 파일 없음: {BATCH_STATE_FILE}")
        print("먼저 --batch-submit 을 실행하세요.")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    batch_list = state.get("batches") or [state]  # 구버전 호환

    # 전체 배치 완료 여부 확인
    not_ended = []
    for b in batch_list:
        batch = client.messages.batches.retrieve(b["batch_id"])
        print(f"배치 상태: {batch.processing_status}  (ID: {b['batch_id']})")
        if batch.processing_status != "ended":
            not_ended.append(b["batch_id"])
    if not_ended:
        print(f"\n[WARN] 아직 완료되지 않은 배치 {len(not_ended)}개. --batch-status 로 확인하세요.")
        return

    # 결과 수집: article_id → img_idx → chunk_idx → parsed dict
    article_chunks: dict[str, dict[int, dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    succeeded = failed = 0

    for b in batch_list:
        meta = b["meta"]
        for result in client.messages.batches.results(b["batch_id"]):
            cid = result.custom_id
            m   = meta.get(cid, {})
            article_id = m.get("article_id", "")
            img_idx    = int(m.get("img_idx", 0))
            chunk_idx  = int(m.get("chunk_idx", 0))

            if result.result.type == "succeeded":
                try:
                    parsed = _parse_raw(result.result.message.content[0].text)
                    article_chunks[article_id][img_idx][chunk_idx] = parsed
                    succeeded += 1
                except Exception as e:
                    print(f"  [ERR] JSON 파싱 실패 {cid}: {e}")
                    failed += 1
            else:
                print(f"  [ERR] API 오류 {cid}: {result.result.error}")
                failed += 1

    print(f"결과 수신: {succeeded}개 성공, {failed}개 실패\n")

    # article_dir 매핑
    dir_map = {d.name: d for d in article_dirs}
    saved = 0

    for article_id, img_map in article_chunks.items():
        article_dir = dir_map.get(article_id)
        if not article_dir:
            continue
        json_path = article_dir / f"{article_id}.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # 이미지별 청크 병합
        img_results = []
        for img_idx in sorted(img_map.keys()):
            chunk_map = img_map[img_idx]
            chunks = [chunk_map[ci] for ci in sorted(chunk_map.keys())]
            img_results.append(
                chunks[0] if len(chunks) == 1 else _merge_chunk_results(chunks)
            )

        merged = merge_parsed_results(img_results)
        if merged:
            data["parsed_estimate"] = merged
            json_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  [SAVE] {article_id}: {len(merged['line_items'])}항목, {merged['total_cost']:,}원")
            saved += 1

    print(f"\n[OK] {saved}개 article 저장 완료")


# ══════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════

def collect_article_dirs(data_root: pathlib.Path) -> list[pathlib.Path]:
    """estimate_data/{지역}/{article_id}/ 구조에서 article_id 폴더 목록 반환."""
    dirs = []
    for region_dir in sorted(data_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue
            if (article_dir / f"{article_dir.name}.json").exists():
                dirs.append(article_dir)
    return dirs


def collect_warning_dirs(all_dirs: list[pathlib.Path]) -> list[pathlib.Path]:
    """_parse_warning이 있는 article_dir만 반환."""
    result = []
    for d in all_dirs:
        json_path = d / f"{d.name}.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if data.get("parsed_estimate", {}).get("_parse_warning"):
                result.append(d)
        except Exception:
            pass
    return result


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reparse", action="store_true",
                        help="이미 파싱된 항목도 다시 파싱")
    parser.add_argument("--id", type=str, default=None,
                        help="특정 article_id만 처리 (예: --id 877065)")
    parser.add_argument("--hires", action="store_true",
                        help="고해상도 모드: MAX_PARSE_WIDTH=2000, CHUNK_HEIGHT=2000 "
                             "(--reparse-warnings 또는 --id 와 함께 사용)")
    parser.add_argument("--reparse-warnings", action="store_true",
                        help="_parse_warning이 있는 항목만 골라 재파싱 (--hires와 함께 사용 권장)")

    # 배치 모드
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch-submit", action="store_true",
                       help="배치 제출 (50%% 비용 절감, 최대 24시간 소요)")
    group.add_argument("--batch-status", action="store_true",
                       help="배치 진행 상황 확인")
    group.add_argument("--batch-apply", action="store_true",
                       help="배치 완료 후 결과 적용")

    args = parser.parse_args()

    data_root = pathlib.Path(DATA_DIR)
    if not data_root.exists():
        print(f"[ERR] 폴더 없음: {DATA_DIR}")
        return

    # --batch-status는 article_dirs 불필요
    if args.batch_status:
        cmd_batch_status()
        return

    article_dirs = collect_article_dirs(data_root)

    if args.id:
        article_dirs = [d for d in article_dirs if d.name == args.id]
        if not article_dirs:
            print(f"[ERR] article_id '{args.id}' 를 찾을 수 없음")
            return

    # --reparse-warnings: _parse_warning 붙은 항목만 필터
    if args.reparse_warnings:
        article_dirs = collect_warning_dirs(article_dirs)
        if not article_dirs:
            print("[OK] _parse_warning 항목 없음 — 재파싱 불필요")
            return
        print(f"[MODE] --reparse-warnings: 경고 항목 {len(article_dirs)}개만 재파싱\n")

    # --hires: 해상도 설정 전역 override
    if args.hires:
        global MAX_PARSE_WIDTH, CHUNK_HEIGHT, SPLIT_HEIGHT_THRESHOLD
        MAX_PARSE_WIDTH          = 2000
        CHUNK_HEIGHT             = 2000
        SPLIT_HEIGHT_THRESHOLD   = 2000
        print("[MODE] --hires: 고해상도 모드 (2000px, 청크 2000px)\n")

    # ── 배치 모드 ──────────────────────────────────────
    if args.batch_submit:
        if args.reparse:
            print("[MODE] --reparse: 이미 파싱된 항목도 재파싱\n")
        cmd_batch_submit(article_dirs, reparse=args.reparse)
        return

    if args.batch_apply:
        cmd_batch_apply(article_dirs)
        return

    # ── 실시간 모드 ────────────────────────────────────
    if args.reparse or args.reparse_warnings:
        print("[MODE] --reparse: 이미 파싱된 항목도 재파싱\n")

    print(f"[DIR] 총 {len(article_dirs)}개 폴더 처리 시작\n")
    success = failed = 0

    force_reparse = args.reparse or args.reparse_warnings

    for i, article_dir in enumerate(article_dirs, 1):
        print(f"[{i}/{len(article_dirs)}] {article_dir.name}")
        try:
            process_article(article_dir, reparse=force_reparse)
            success += 1
        except Exception as e:
            print(f"  [ERR] 처리 오류: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"[OK] 완료: {success}개 처리, {failed}개 실패")
    print("=" * 50)


if __name__ == "__main__":
    main()
