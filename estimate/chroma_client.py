"""
chroma_client.py — ChromaDB 클라이언트 공유 유틸리티

Chroma Cloud 연결 설정을 한 곳에서 관리합니다.
모든 스크립트는 이 모듈에서 get_chroma_client()를 import하여 사용합니다.

환경 변수 (.env):
    CHROMA_API_KEY  — Chroma Cloud API 키
    CHROMA_TENANT   — Chroma Cloud 테넌트 ID
    CHROMA_DATABASE — Chroma Cloud 데이터베이스 이름
"""

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()


def get_chroma_client() -> chromadb.HttpClient:
    """Chroma Cloud HttpClient 반환."""
    api_key  = os.environ.get("CHROMA_API_KEY")
    tenant   = os.environ.get("CHROMA_TENANT")
    database = os.environ.get("CHROMA_DATABASE")

    missing = [k for k, v in {
        "CHROMA_API_KEY":  api_key,
        "CHROMA_TENANT":   tenant,
        "CHROMA_DATABASE": database,
    }.items() if not v]

    if missing:
        raise EnvironmentError(
            f"Chroma Cloud 환경변수 누락: {', '.join(missing)}\n"
            f"  .env 파일에 CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE를 설정하세요.\n"
            f"  참고: .env.example"
        )

    return chromadb.HttpClient(
        ssl=True,
        host="api.trychroma.com",
        tenant=tenant,
        database=database,
        headers={"x-chroma-token": api_key},
    )
