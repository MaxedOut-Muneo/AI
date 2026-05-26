from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any

from risk_detector.analyzer import RiskAnalyzer
from risk_detector.chunker import EstimateImageChunker
from risk_detector.constants import SUPPORTED_SPACE_TYPES
from risk_detector.dto import AnalyzeRiskCommand
from risk_detector.formatter import ResponseFormatter
from risk_detector.models import RiskIssue
from risk_detector.parser import ClaudeVisionParser

CONTEXT_CARRYING_KEYWORDS = [
    "양중",
    "운반",
    "사다리차",
    "엘리베이터",
    "EV",
    "계단",
    "양중비",
    "운반비",
    "하역",
]

class RiskDetectorService:
    def __init__(self) -> None:
        self.chunker = EstimateImageChunker()
        self.parser = ClaudeVisionParser()
        self.analyzer = RiskAnalyzer()
        self.formatter = ResponseFormatter()

    def analyze(self, command: AnalyzeRiskCommand) -> dict[str, Any]:
        self._validate_input(command)

        all_items = []
        for raw in command.image_files:
            chunks = self.chunker.prepare_chunks(raw)
            configured_workers = int(os.getenv("RISK_DETECTOR_MAX_WORKERS", "1"))
            configured_workers = max(1, min(2, configured_workers))
            max_workers = min(configured_workers, len(chunks) or 1)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                chunk_results = list(ex.map(self.parser.parse_chunk, chunks))
            all_items.extend(self._merge_chunk_results(chunk_results))

        if all_items:
            issues, detected_processes = self.analyzer.analyze(all_items)
            self._add_contextual_issues(command, all_items, issues, detected_processes)
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
            company_name=command.company_name,
            space_type=command.space_type,
            pyeong=command.pyeong,
            room_count=command.room_count,
            floor=command.floor,
            elevator=command.elevator,
            region=command.region,
            building_age=command.building_age,
            line_items=all_items,
            issues=issues,
            requested_processes=detected_processes,
        )

    def _merge_chunk_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen = set()
        merged = []
        for result in results:
            for item in result.get("line_items", []):
                key = (
                    item.get("category", ""),
                    item.get("description", ""),
                    int(item.get("amount") or 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _validate_input(self, command: AnalyzeRiskCommand) -> None:
        if command.space_type not in SUPPORTED_SPACE_TYPES:
            raise ValueError(f"지원하지 않는 공간유형입니다: {command.space_type}")
        if not command.image_files or not any(command.image_files):
            raise ValueError("최소 1개 이상의 견적서 이미지가 필요합니다.")
    
    def _add_contextual_issues(
        self,
        command: AnalyzeRiskCommand,
        line_items: list[dict[str, Any]],
        issues: list[RiskIssue],
        detected_processes: list[str],
    ) -> None:
        if command.floor < 5:
            return

        search_text = " ".join(
            f"{item.get('category', '')} {item.get('description', '')} {item.get('notes', '')}"
            for item in line_items
        )
        has_carrying_cost = any(keyword in search_text for keyword in CONTEXT_CARRYING_KEYWORDS)
        if has_carrying_cost:
            return

        issues.append(
            RiskIssue(
                "불분명",
                "공통",
                "고층 시공 운반/양중 비용 정보 미기재",
                f"{command.floor}층 시공 조건이지만 견적서에서 양중/운반 관련 항목이 확인되지 않습니다.",
                "고층 작업 시 운반비·양중비·사다리차 비용 포함 여부를 업체에 확인하세요.",
            )
        )
        if "공통" not in detected_processes:
            detected_processes.append("공통")