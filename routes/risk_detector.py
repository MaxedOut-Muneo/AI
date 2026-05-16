from fastapi import APIRouter, Body, File, Form, Header, HTTPException, UploadFile

from risk_detector.dto import AnalyzeRiskCommand, AnalyzeRiskResponse
from risk_detector.report_service import delete_risk_report, list_risk_reports, save_risk_report
from risk_detector.service import RiskDetectorService

router = APIRouter(prefix="/risk-detector", tags=["risk-detector"])
service = RiskDetectorService()

_SAMPLE_INPUT = {
    "space_type": "아파트",
    "pyeong": 30,
    "room_count": 3,
    "floor": 8,
    "elevator": True,
    "region": "서울",
    "building_age": "20년이상",
    "company_name": "홍길동 인테리어",
}

_SAMPLE_RESULT = AnalyzeRiskResponse.model_config["json_schema_extra"]["example"]

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

@router.post("/save", status_code=201)
async def save_report(
    body: dict = Body(example={"input": _SAMPLE_INPUT, "result": _SAMPLE_RESULT}),
    x_user_id: str = Header(..., example="user_abc123"),
):
    user_input = body.get("input")
    result = body.get("result")
    if not user_input or not result:
        raise HTTPException(status_code=400, detail="input과 result가 필요합니다.")

    report_id = await save_risk_report(x_user_id, user_input, result)
    return {"id": report_id}


@router.get("")
async def get_reports(x_user_id: str = Header(..., example="user_abc123")):
    return await list_risk_reports(x_user_id)


@router.delete("/{report_id}", status_code=204)
async def remove_report(
    report_id: str = "6801234567890abcdef12345",
    x_user_id: str = Header(..., example="user_abc123"),
):
    result = await delete_risk_report(report_id, x_user_id)
    if result is None:
        raise HTTPException(status_code=400, detail="유효하지 않은 ID입니다.")
    if not result:
        raise HTTPException(status_code=404, detail="리스크 진단 리포트를 찾을 수 없습니다.")
