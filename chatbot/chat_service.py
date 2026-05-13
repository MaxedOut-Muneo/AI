import os
import re
from typing import Any

from dotenv import load_dotenv

from chatbot.estimate_adapter import search_estimate_cases_for_reference
from chatbot.legal_rag_search import search_legal_docs
from chatbot.question_classifier import classify_question
from chatbot.schemas import ChatResponse, ChatSource, ChatUsed

load_dotenv()

NON_INTERIOR_GUIDE_MESSAGE = (
    "이 챗봇은 인테리어 견적서, 계약, 하자, 시공 관련 상담을 돕는 용도입니다. "
    "인테리어 관련 질문을 입력해 주세요."
)


def format_legal_docs(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "검색된 계약/하자 기준 자료 없음"

    lines = []

    for idx, doc in enumerate(docs, start=1):
        meta = doc.get("metadata", {})

        lines.append(
            f"[근거 {idx}]\n"
            f"자료: {meta.get('title')}\n"
            f"유형: {meta.get('source_type')}\n"
            f"항목: {meta.get('article_no')} {meta.get('section')}\n"
            f"하자분류: {meta.get('defect_category', '')}\n"
            f"내용:\n{doc.get('document', '')}"
        )

    return "\n\n".join(lines)


def build_prompt(question: str, flags: dict[str, bool], estimate_context: str, legal_context: str) -> str:
    return f"""
너는 인테리어 견적서 검토, 계약 체크, 하자 상담을 도와주는 챗봇이다.

중요한 정책:
1. 견적서 사례는 가격 예측 목적이 아니다.
2. 견적서 사례는 견적서 양식, 포함 항목, 공사 범위, 누락 가능 항목을 참고하는 용도로만 사용한다.
3. 표준계약서와 하자판정기준은 근거 기반 답변에 사용한다.
4. 저장된 자료와 직접 관련 없는 일반 인테리어 질문은 일반 지식으로 답변해도 된다.
5. 법적 판단, 확정 하자 판단, 확정 가격 판단은 단정하지 말고 추가 확인이 필요하다고 말한다.
6. 비인테리어 질문이면 인테리어 상담 범위의 질문을 해달라고 안내한다.
7. 반드시 "일반 텍스트"로 답변할 것, Markdown(#, **, -, 표, --- 등) 사용 금지
8. 줄바꿈은 자연스럽게 문장 단위로만 사용, 읽기 쉬운 문장 형태로 작성

[사용자 질문]
{question}

[질문 분류 결과]
{flags}

[견적서 사례 참고 결과]
{estimate_context}

[표준계약서/하자판정기준 검색 결과]
{legal_context}

[답변 형식]
핵심 답변
참고한 기준 또는 사례
확인해야 할 항목
주의사항
""".strip()

def clean_answer(text: str) -> str:
    text = re.sub(r"#.*\n", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"^-\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


class ChatbotService:
    def __init__(self, anthropic_client: Any | None = None, model: str | None = None):
        self._client = anthropic_client
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._client

    def chat(self, question: str) -> ChatResponse:
        flags = classify_question(question)
        used = ChatUsed(**flags)

        if not used.is_interior:
            return ChatResponse(
                answer=NON_INTERIOR_GUIDE_MESSAGE,
                used=used,
                sources=[],
            )

        estimate_context = "견적서 사례 검색 사용 안 함"
        legal_docs: list[dict[str, Any]] = []

        if used.use_estimate_cases:
            estimate_context = search_estimate_cases_for_reference(question)

        if used.use_legal_docs or used.use_defect_docs:
            legal_docs = search_legal_docs(question, top_k=4)

        prompt = build_prompt(
            question=question,
            flags=flags,
            estimate_context=estimate_context,
            legal_context=format_legal_docs(legal_docs),
        )

        response = self.client.messages.create(
            model=self._model,
            max_tokens=1200,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )

        return ChatResponse(
            answer=clean_answer(response.content[0].text),
            used=used,
            sources=self._build_sources(legal_docs),
        )

    @staticmethod
    def _build_sources(legal_docs: list[dict[str, Any]]) -> list[ChatSource]:
        sources = []
        for doc in legal_docs:
            metadata = doc.get("metadata", {})
            sources.append(
                ChatSource(
                    title=metadata.get("title"),
                    source_type=metadata.get("source_type"),
                    article_no=metadata.get("article_no"),
                    section=metadata.get("section"),
                    defect_category=metadata.get("defect_category"),
                    page=metadata.get("page"),
                    distance=doc.get("distance"),
                )
            )
        return sources


_default_service = ChatbotService()


def generate_chat_answer(question: str) -> dict[str, Any]:
    return _default_service.chat(question).model_dump()