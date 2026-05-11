from concurrent.futures import ThreadPoolExecutor
from typing import Any

from risk_detector.analyzer import RiskAnalyzer
from risk_detector.chunker import EstimateImageChunker
from risk_detector.constants import SUPPORTED_SPACE_TYPES
from risk_detector.formatter import ResponseFormatter
from risk_detector.models import RiskIssue
from risk_detector.parser import ClaudeVisionParser


class RiskDetectorService:
    def __init__(self) -> None:
        self.chunker = EstimateImageChunker()
        self.parser = ClaudeVisionParser()
        self.analyzer = RiskAnalyzer()
        self.formatter = ResponseFormatter()

    def analyze(self, *, space_type: str, image_files: list[bytes], company_name: str, pyeong: int, room_count: int, floor: int, elevator: bool, region: str, building_age: str) -> dict[str, Any]:
        self._validate_input(space_type, image_files)

        all_items = []
        for raw in image_files:
            chunks = self.chunker.prepare_chunks(raw)
            with ThreadPoolExecutor(max_workers=min(3, len(chunks) or 1)) as ex:
                chunk_results = list(ex.map(self.parser.parse_chunk, chunks))
            all_items.extend(self._merge_chunk_results(chunk_results))

        if all_items:
            issues, detected_processes = self.analyzer.analyze(all_items)
        else:
            issues = [
                RiskIssue(
                    "불분명",
                    "견적서",
                    "견적서 항목 추출 실패",
                    "업로드한 견적서에서 분석 가능한 품목을 추출하지 못했습니다.",
                    "이미지 해상도, 파일 형식, 견적서 표 영역이 선명한지 확인한 뒤 다시 업로드하세요.",
                )
            ]
            detected_processes = ["견적서"]

        return self.formatter.build(
            company_name=company_name,
            space_type=space_type,
            pyeong=pyeong,
            room_count=room_count,
            floor=floor,
            elevator=elevator,
            region=region,
            building_age=building_age,
            line_items=all_items,
            issues=issues,
            requested_processes=detected_processes,
        )

    def _merge_chunk_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        merged = []
        for result in results:
            for item in result.get("line_items", []):
                key = (item.get("category", ""), item.get("description", ""), int(item.get("amount") or 0))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _validate_input(self, space_type: str, image_files: list[bytes]) -> None:
        if space_type not in SUPPORTED_SPACE_TYPES:
            raise ValueError(f"지원하지 않는 공간유형입니다: {space_type}")
        if not image_files:
            raise ValueError("최소 1개 이상의 견적서 이미지가 필요합니다.")