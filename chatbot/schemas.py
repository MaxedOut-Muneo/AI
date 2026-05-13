from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="사용자 질문")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("question은 공백일 수 없습니다.")
        return question


class ChatUsed(BaseModel):
    is_interior: bool = Field(..., description="인테리어 관련 질문 여부")
    use_estimate_cases: bool = Field(..., description="견적 사례 RAG 사용 여부")
    use_legal_docs: bool = Field(..., description="표준계약서 검색 사용 여부")
    use_defect_docs: bool = Field(..., description="하자판정기준 검색 사용 여부")


class ChatSource(BaseModel):
    title: Optional[str] = Field(None, description="문서 제목")
    source_type: Optional[str] = Field(None, description="문서 유형")
    article_no: Optional[str] = Field(None, description="조항 번호")
    section: Optional[str] = Field(None, description="조항/섹션명")
    defect_category: Optional[str] = Field(None, description="하자 분류")
    page: Optional[int] = Field(None, description="문서 페이지")
    distance: Optional[float] = Field(None, description="벡터 검색 거리")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="챗봇 답변")
    used: ChatUsed = Field(..., description="답변 생성에 사용한 기능 플래그")
    sources: list[ChatSource] = Field(default_factory=list, description="참고한 법적 문서 목록")