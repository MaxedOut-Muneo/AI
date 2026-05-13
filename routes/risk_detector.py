from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from risk_detector.dto import AnalyzeRiskCommand, AnalyzeRiskResponse
from risk_detector.service import RiskDetectorService

router = APIRouter(prefix="/risk-detector", tags=["risk-detector"])
service = RiskDetectorService()


@router.post("/analyze", response_model=AnalyzeRiskResponse)
async def analyze_risk(
    space_type: str = Form(..., description="아파트|빌라|오피스텔|단독주택"),
    pyeong: int = Form(..., description="면적(평)"),
    room_count: int = Form(..., description="방 개수"),
    floor: int = Form(..., description="층수"),
    elevator: bool = Form(..., description="엘리베이터 유무 (true/false)"),
    region: str = Form(..., description="지역"),
    building_age: str = Form(..., description="건물 연식"),
    company_name: str = Form(..., description="업체명"),
    files: list[UploadFile] = File(..., description="견적서 이미지 여러개 업로드 가능"),
):
    try:
        image_bytes = [await f.read() for f in files]
        command = AnalyzeRiskCommand(
            space_type=space_type,
            pyeong=pyeong,
            room_count=room_count,
            floor=floor,
            elevator=elevator,
            region=region,
            building_age=building_age,
            company_name=company_name,
            image_files=image_bytes,
        )
        return service.analyze(command)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리스크 분석 실패: {e}")
