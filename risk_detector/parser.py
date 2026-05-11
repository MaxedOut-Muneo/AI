import base64
import json
import os
import re
from typing import Any

from anthropic import Anthropic

from risk_detector.constants import MAX_VISION_TOKENS, MODEL, PARSE_PROMPT


class ClaudeVisionParser:
    def __init__(self) -> None:
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def parse_chunk(self, chunk: bytes) -> dict[str, Any]:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_VISION_TOKENS,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(chunk).decode("utf-8")}},
                        {"type": "text", "text": PARSE_PROMPT},
                    ],
                }
            ],
        )
        return self._extract_json_safe(response.content[0].text)

    def _extract_json_safe(self, text: str) -> dict[str, Any]:
        raw = text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        # 1) 1차 파싱
        parsed = self._try_raw_decode(raw)
        if parsed is not None:
            return self._normalize(parsed)

        # 2) JSON 블록 범위 재추출 후 재시도
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            parsed = self._try_loads(candidate)
            if parsed is not None:
                return self._normalize(parsed)

            # 3) 자주 나는 LLM JSON 오류(후행 콤마) 보정
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            parsed = self._try_loads(candidate)
            if parsed is not None:
                return self._normalize(parsed)

        # 4) 실패 시 예외 대신 빈 구조 반환 (전체 요청 실패 방지)
        return {"line_items": []}

    def _try_raw_decode(self, raw: str) -> Any | None:
        object_start = raw.find("{")
        array_start = raw.find("[")
        starts = [idx for idx in [object_start, array_start] if idx != -1]
        if not starts:
            return None
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw, min(starts))
            return obj if isinstance(obj, (dict, list)) else None
        except json.JSONDecodeError:
            return None

    def _try_loads(self, raw: str) -> Any | None:
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, (dict, list)) else None
        except json.JSONDecodeError:
            return None

    def _normalize(self, obj: Any) -> dict[str, Any]:
        if isinstance(obj, list):
            return {"line_items": self._collect_line_items(obj)}

        items = obj.get("line_items") or obj.get("items") or obj.get("estimate_items") or []
        if not isinstance(items, list):
            items = self._collect_line_items(obj)
        if not items:
            items = self._collect_line_items(obj)
        obj["line_items"] = items
        return obj

    def _collect_line_items(self, value: Any, parent_key: str = "") -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        if isinstance(value, list):
            for entry in value:
                collected.extend(self._collect_line_items(entry, parent_key))
        elif isinstance(value, dict):
            if any(key in value for key in ["category", "description", "amount", "unit_price"]):
                item = dict(value)
                if parent_key and not item.get("category"):
                    item["category"] = parent_key
                collected.append(item)
            else:
                for key, entry in value.items():
                    next_parent = key if isinstance(key, str) else parent_key
                    collected.extend(self._collect_line_items(entry, next_parent))
        return collected