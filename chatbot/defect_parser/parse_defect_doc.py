import os
from pathlib import Path

import fitz
from dotenv import load_dotenv
from anthropic import Anthropic

from chatbot.defect_parser.prompt_defect import DEFECT_PARSE_PROMPT
from chatbot.defect_parser.utils import (
    ensure_dir,
    encode_image_to_base64,
    parse_json_from_text,
    save_json
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE_DIR / "defect_images"
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "defect_standard.json"

PDF_PATH = Path(os.getenv("DEFECT_PDF_PATH", r"C:\Standards for Defect.pdf"))

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 4000


def pdf_to_images(pdf_path: Path, output_dir: Path, zoom: float = 2.0):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    ensure_dir(output_dir)

    doc = fitz.open(str(pdf_path))
    image_paths = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_path = output_dir / f"page_{page_index + 1:03d}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


def parse_one_page(client: Anthropic, image_path: Path):
    image_base64 = encode_image_to_base64(image_path)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": DEFECT_PARSE_PROMPT
                    }
                ]
            }
        ]
    )

    text = response.content[0].text
    return parse_json_from_text(text)


def normalize_item(item: dict, page_no: int):
    keywords = item.get("keywords", [])

    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",") if k.strip()]

    text = item.get("text", "").strip()
    criteria = item.get("criteria", "").strip()

    merged_text = text
    if criteria and criteria not in merged_text:
        merged_text = f"{text}\n하자 판단 기준: {criteria}".strip()

    return {
        "article_no": item.get("article_no", f"page_{page_no}"),
        "section": item.get("section", "항목명 없음"),
        "defect_category": item.get("defect_category", "기타"),
        "text": merged_text,
        "criteria": criteria,
        "keywords": keywords,
        "page": page_no
    }


def merge_page_results(page_results: list[dict]):
    content = []
    seen = set()

    for page_no, result in page_results:
        for item in result.get("items", []):
            normalized = normalize_item(item, page_no)

            key = (
                normalized["article_no"],
                normalized["section"],
                normalized["text"][:80]
            )

            if key in seen:
                continue

            seen.add(key)
            content.append(normalized)

    return {
        "source_type": "하자판정기준",
        "title": "공동주택 하자의 조사, 보수비용 산정 및 하자판정기준",
        "issuer": "국토교통부",
        "category": "하자",
        "content": content
    }


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY가 .env에 없습니다.")

    client = Anthropic(api_key=api_key)

    print(f"[1] PDF → 이미지 변환 시작: {PDF_PATH}")
    image_paths = pdf_to_images(PDF_PATH, IMAGE_DIR)
    print(f"[1] 이미지 생성 완료: {len(image_paths)}페이지")

    page_results = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[2] Claude Vision 파싱 중: {image_path.name}")

        try:
            parsed = parse_one_page(client, image_path)
            page_results.append((idx, parsed))
        except Exception as e:
            print(f"[WARN] {image_path.name} 파싱 실패: {e}")

    final_json = merge_page_results(page_results)

    ensure_dir(DATA_DIR)
    save_json(final_json, OUTPUT_PATH)

    print(f"[3] JSON 저장 완료: {OUTPUT_PATH}")
    print(f"[3] 추출 항목 수: {len(final_json['content'])}")


if __name__ == "__main__":
    main()