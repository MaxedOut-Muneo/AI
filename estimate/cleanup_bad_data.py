"""
cleanup_bad_data.py — 불량 데이터 폴더 정리

실행:
    python cleanup_bad_data.py          # dry-run (삭제 없이 목록만 출력)
    python cleanup_bad_data.py --delete # 실제 삭제

불량 판단 기준:
    1. JSON이 없는 폴더 (이미지만 존재)
    2. body_text가 비어있는 JSON (iframe 실패로 빈 껍데기)
    3. image_urls는 있는데 local_images가 비어있는 폴더 (다운로드 실패)
"""

import sys
import json
import shutil
import pathlib

DATA_DIR = "./estimate_data"


def scan_bad_dirs(data_root: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    """불량 폴더 목록 반환: [(path, reason), ...]"""
    bad = []
    for region_dir in sorted(data_root.iterdir()):
        if not region_dir.is_dir():
            continue
        for article_dir in sorted(region_dir.iterdir()):
            if not article_dir.is_dir():
                continue

            json_path = article_dir / f"{article_dir.name}.json"

            # 기준 1: JSON 없음
            if not json_path.exists():
                bad.append((article_dir, "JSON 없음 (이미지만 존재)"))
                continue

            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                bad.append((article_dir, "JSON 파싱 오류"))
                continue

            # 기준 2: body_text 비어있음
            if not data.get("body_text", "").strip():
                bad.append((article_dir, "body_text 비어있음 (크롤링 실패)"))
                continue

            # 기준 3: 이미지 URL은 있는데 실제 다운로드된 이미지 없음
            has_image_urls = bool(data.get("image_urls"))
            has_local = bool(data.get("local_images"))
            image_files = [
                f for f in article_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ]
            if has_image_urls and not image_files:
                bad.append((article_dir, "이미지 URL 있으나 다운로드 실패"))

    return bad


def main():
    delete_mode = "--delete" in sys.argv
    data_root = pathlib.Path(DATA_DIR)

    if not data_root.exists():
        print(f"[ERR] 폴더 없음: {DATA_DIR}")
        return

    bad_dirs = scan_bad_dirs(data_root)

    if not bad_dirs:
        print("[OK] 불량 폴더 없음.")
        return

    print(f"{'[DRY-RUN] ' if not delete_mode else ''}불량 폴더 {len(bad_dirs)}개 발견:\n")
    for path, reason in bad_dirs:
        region = path.parent.name
        print(f"  [{region}/{path.name}]  {reason}")

    if not delete_mode:
        print(f"\n실제 삭제하려면: python cleanup_bad_data.py --delete")
        return

    print()
    confirm = input(f"위 {len(bad_dirs)}개 폴더를 삭제하시겠습니까? (y/N): ").strip().lower()
    if confirm != "y":
        print("취소됨.")
        return

    deleted = 0
    for path, reason in bad_dirs:
        shutil.rmtree(path)
        print(f"  [DEL] {path.parent.name}/{path.name}  ({reason})")
        deleted += 1

    print(f"\n[OK] {deleted}개 폴더 삭제 완료.")


if __name__ == "__main__":
    main()
