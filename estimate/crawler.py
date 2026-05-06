"""
박목수 열린견적서 카페 크롤러
────────────────────────────────────────────────
특정 유저의 게시글 목록 → 견적의뢰 원문 → 견적서 이미지 자동 수집

출력 구조:
    estimate_data/
    ├── estimates.json
    ├── 서울/
    │   └── {article_id}/
    │       ├── {article_id}_0.jpg
    │       └── {article_id}.json
    ├── 경기/
    │   └── ...
    └── 기타/
        └── ...
"""

import os
import re
import json
import time
import pathlib
import requests

from region_utils import detect_region
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


# ══════════════════════════════════════════════════════
# 설정 (여기만 수정하면 됩니다)
# ══════════════════════════════════════════════════════

CAFE_ID    = "17593353"
BASE_DIR   = "./estimate_data"          # 저장 폴더 경로
MAX_PAGES  = 50                          # 유저 게시글 최대 페이지 수
SLEEP_SEC  = 1.5                         # 요청 간 대기 시간 (너무 짧으면 차단될 수 있음)

# 저장 경로 자동 생성
OUTPUT_FILE = os.path.join(BASE_DIR, "estimates.json")
os.makedirs(BASE_DIR, exist_ok=True)

# ── 이미지 필터링 / 화질 업그레이드
NOISE_KEYWORDS = [
    "logo_icon",
    "필독_큰_버튼",
    "dthumb-phinf",
    "f100_100",
    "f1480_240_banner",
    "ConfigProfileFileName",   # 업체 프로필 사진
    "로고",                    # 업체 로고
]

def is_valid_estimate_image(url):
    return not any(kw in url for kw in NOISE_KEYWORDS)

def upgrade_image_url(url):
    """썸네일 URL → 원본 크기로 변환 (type=w1600)"""
    return re.sub(r'type=[^&"\']+', 'type=w1600', url)


# ══════════════════════════════════════════════════════
# 드라이버 & 로그인
# ══════════════════════════════════════════════════════

def get_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # options.add_argument("--headless")  # 브라우저 창 숨기려면 주석 해제 (로그인 후에만 사용)
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


def naver_login(driver):
    """네이버 로그인 (수동)"""
    driver.get("https://nid.naver.com/nidlogin.login")
    print("=" * 50)
    print("브라우저에서 네이버 로그인을 완료한 후")
    print("이 터미널에서 Enter 키를 누르세요.")
    print("=" * 50)
    input()
    print("로그인 확인 완료\n")


# ══════════════════════════════════════════════════════
# iframe 진입 헬퍼
# ══════════════════════════════════════════════════════

def enter_iframe(driver, timeout=10):
    """cafe_main iframe 진입. 성공하면 True 반환"""
    try:
        WebDriverWait(driver, timeout).until(
            EC.frame_to_be_available_and_switch_to_it("cafe_main")
        )
        time.sleep(1.5)
        return True
    except Exception:
        driver.switch_to.default_content()
        return False


# ══════════════════════════════════════════════════════
# 1단계: 유저 게시글 목록 수집
# ══════════════════════════════════════════════════════

def get_user_articles(driver, member_hash, max_pages=MAX_PAGES,
                      existing_ids: set = None, max_new: int = None):
    """
    특정 유저의 게시글 목록 수집.
    max_new: 신규 수집 가능 건수 목표 — 달성하면 페이지 탐색 조기 종료.
    반환: [{"title": ..., "url": ...}, ...]
    """
    base_url = f"https://cafe.naver.com/f-e/cafes/{CAFE_ID}/members/{member_hash}"
    all_articles = []
    new_count = 0

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        driver.get(url)

        if not enter_iframe(driver):
            print(f"  [WARN] iframe 진입 실패 (page {page}) → 수집 종료")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.select(".article-board tbody tr")

        if not rows:
            print(f"  더 이상 게시글 없음 (page {page}) → 수집 종료")
            driver.switch_to.default_content()
            break

        page_articles = []
        for row in rows:
            title_el = row.select_one("a.article")
            if not title_el:
                continue
            href = title_el.get("href", "")
            full_url = "https://cafe.naver.com" + href if href.startswith("/") else href
            article = {"title": title_el.get_text(strip=True), "url": full_url}
            page_articles.append(article)

            # 신규 건수 카운트 (article_id 미리 추출)
            if existing_ids is not None and max_new is not None:
                m = re.search(r'articleid(?:%3D|=)(\d+)', full_url, re.IGNORECASE)
                aid = m.group(1) if m else None
                if aid and aid not in existing_ids:
                    new_count += 1

        all_articles.extend(page_articles)
        print(f"  페이지 {page}: {len(page_articles)}건 (누적 {len(all_articles)}건"
              + (f", 신규 {new_count}건)" if max_new else ")"))

        driver.switch_to.default_content()

        # 목표 달성 시 조기 종료
        if max_new is not None and new_count >= max_new:
            print(f"  목표 신규 {max_new}건 달성 → 탐색 종료")
            break

        time.sleep(SLEEP_SEC)

    return all_articles


# ══════════════════════════════════════════════════════
# 2단계: 견적방 게시글 → 견적의뢰 링크 추출
# ══════════════════════════════════════════════════════

def get_estimate_link_from_post(driver, article_url):
    """
    견적방 게시글 본문에서 pcarpenter 견적의뢰 링크 추출
    반환: URL 문자열 or None
    """
    driver.get(article_url)

    if not enter_iframe(driver):
        return None

    soup = BeautifulSoup(driver.page_source, "html.parser")

    estimate_link = None
    for a in soup.select("a"):
        href = a.get("href", "")
        if "cafe.naver.com/pcarpenter" in href:
            estimate_link = href
            break

    driver.switch_to.default_content()
    return estimate_link


# ══════════════════════════════════════════════════════
# 3단계: 견적의뢰 원문 수집
# ══════════════════════════════════════════════════════

def get_estimate_detail(driver, estimate_url, retry: int = 2):
    """
    견적의뢰 원문 글에서 공사정보 + 이미지 URL 수집
    반환: dict (body_text가 비어있으면 None 반환)
    """
    for attempt in range(1, retry + 1):
        driver.get(estimate_url)
        if not enter_iframe(driver):
            print(f"    [WARN] iframe 진입 실패 (시도 {attempt}/{retry})")
            time.sleep(3)
            continue

        soup = BeautifulSoup(driver.page_source, "html.parser")
        body_el = (
            soup.select_one(".se-main-container") or
            soup.select_one(".ContentRenderer") or
            soup.select_one("#postContent")
        )
        body_text = body_el.get_text("\n", strip=True) if body_el else ""
        driver.switch_to.default_content()

        if body_text:
            return _parse_detail(soup, body_text)

        print(f"    [WARN] 본문 비어있음 (시도 {attempt}/{retry}), {3}초 후 재시도...")
        time.sleep(3)

    return None  # 재시도 모두 실패


def _parse_detail(soup, body_text: str) -> dict:
    """파싱된 soup + body_text로 detail dict 구성."""
    def extract(pattern, text):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""

    location   = extract(r"공사지역\s*[:：]\s*(.+)", body_text)
    deadline   = extract(r"공사희망일\s*[:：]\s*(.+)", body_text)
    company    = extract(r"지정\s*열린업체명\s*[:：]\s*(.+)", body_text)
    size_match = re.search(r"(\d+)\s*평", body_text)
    size       = int(size_match.group(1)) if size_match else 0

    # ── 이미지 URL 수집 (견적서 이미지, 중복 제거)
    seen_urls: set[str] = set()
    image_urls = []
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-lazy-src") or img.get("data-src") or ""
        if src and ("postfiles.pstatic.net" in src or "cafeptthumb" in src):
            if is_valid_estimate_image(src):
                upgraded = upgrade_image_url(src)
                if upgraded not in seen_urls:
                    seen_urls.add(upgraded)
                    image_urls.append(upgraded)

    # ── 견적의뢰글 URL 추출 (본문 내 링크)
    request_url_match = re.search(r'https?://cafe\.naver\.com/pcarpenter/(\d+)', body_text)
    request_url = request_url_match.group(0) if request_url_match else ""

    return {
        "location":    location,
        "deadline":    deadline,
        "company":     company,
        "size_pyeong": size,
        "body_text":   body_text,
        "request_url": request_url,
        "image_urls":  image_urls,
    }


# ══════════════════════════════════════════════════════
# 3-2단계: 견적의뢰글 원문 텍스트 수집
# ══════════════════════════════════════════════════════

def get_request_body(driver, request_url):
    """
    고객이 작성한 견적의뢰글 원문 텍스트 수집
    반환: str
    """
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import UnexpectedAlertPresentException, NoAlertPresentException

    driver.get(request_url)

    # 삭제된 게시글 등 alert 팝업 처리
    try:
        WebDriverWait(driver, 3).until(EC.alert_is_present())
        driver.switch_to.alert.accept()
        return ""
    except Exception:
        pass

    if not enter_iframe(driver):
        return ""

    try:
        soup = BeautifulSoup(driver.page_source, "html.parser")
    except UnexpectedAlertPresentException:
        try:
            driver.switch_to.alert.accept()
        except NoAlertPresentException:
            pass
        return ""
    body_el = (
        soup.select_one(".se-main-container") or
        soup.select_one(".ContentRenderer") or
        soup.select_one("#postContent")
    )
    text = body_el.get_text("\n", strip=True) if body_el else ""
    driver.switch_to.default_content()
    return text


# ══════════════════════════════════════════════════════
# 3-3단계: 본문 내 링크된 견적서 페이지 이미지 수집
# ══════════════════════════════════════════════════════

def collect_linked_images(driver, body_text: str) -> list[str]:
    """
    body_text 내 모든 pcarpenter 링크를 방문해 이미지 URL 수집.
    래퍼 포스트(열린견적서 목록)가 실제 견적 이미지를 링크로만 가리킬 때 사용.
    반환: 이미지 URL 리스트 (중복 제거)
    """
    linked_urls = re.findall(r'https?://cafe\.naver\.com/pcarpenter/\d+', body_text)
    if not linked_urls:
        return []

    seen: set[str] = set()
    all_image_urls: list[str] = []

    for url in linked_urls:
        try:
            driver.get(url)
            # 삭제된 게시글 alert 처리
            try:
                from selenium.webdriver.common.alert import Alert
                Alert(driver).dismiss()
                print(f"    [SKIP] 삭제된 게시글: {url}")
                continue
            except Exception:
                pass

            if not enter_iframe(driver):
                driver.switch_to.default_content()
                continue

            soup = BeautifulSoup(driver.page_source, "html.parser")
            for img in soup.select("img"):
                src = (img.get("src") or img.get("data-lazy-src")
                       or img.get("data-src") or "")
                if src and ("postfiles.pstatic.net" in src or "cafeptthumb" in src):
                    if is_valid_estimate_image(src):
                        upgraded = upgrade_image_url(src)
                        if upgraded not in seen:
                            seen.add(upgraded)
                            all_image_urls.append(upgraded)

            driver.switch_to.default_content()
            print(f"    링크 방문: {url} → 이미지 {len(all_image_urls)}개 누적")
        except Exception as e:
            print(f"    [ERR] 링크 방문 실패 {url}: {e}")
            try:
                driver.switch_to.default_content()
            except Exception:
                pass
        time.sleep(SLEEP_SEC)

    return all_image_urls


# ══════════════════════════════════════════════════════
# 4단계: 이미지 다운로드
# ══════════════════════════════════════════════════════

def download_images(image_urls, article_dir: str):
    """
    지정된 폴더에 견적서 이미지 저장
    반환: 저장된 로컬 경로 리스트
    """
    if not image_urls:
        return []

    os.makedirs(article_dir, exist_ok=True)

    headers = {
        "Referer":    "https://cafe.naver.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    saved_paths = []
    folder_name = os.path.basename(article_dir)

    for idx, url in enumerate(image_urls):
        if not url:
            continue
        try:
            res = requests.get(url, headers=headers, timeout=15)
            res.raise_for_status()

            # 확장자 추출
            ext = url.split(".")[-1].split("?")[0].lower()
            if ext not in ["jpg", "jpeg", "png", "webp", "gif"]:
                ext = "jpg"

            fname = os.path.join(article_dir, f"{folder_name}_{idx}.{ext}")
            with open(fname, "wb") as f:
                f.write(res.content)

            saved_paths.append(fname)
            print(f"      저장: {fname}")

        except Exception as e:
            print(f"      [ERR] 실패 ({url[:60]}): {e}")

    return saved_paths


# ══════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════

def _existing_article_ids() -> set[str]:
    """이미 수집된 article_id 목록 반환 (재실행 시 skip용)."""
    ids = set()
    base = pathlib.Path(BASE_DIR)
    if not base.exists():
        return ids
    for region_dir in base.iterdir():
        if not region_dir.is_dir():
            continue
        for article_dir in region_dir.iterdir():
            if not article_dir.is_dir():
                continue
            if (article_dir / f"{article_dir.name}.json").exists():
                ids.add(article_dir.name)
    return ids


def crawl_user(member_hash: str, max_pages: int = MAX_PAGES, max_articles: int = 100):
    """
    유저 member_hash를 받아 전체 데이터 수집
    반환: 수집된 데이터 리스트
    """
    driver = get_driver()

    try:
        naver_login(driver)

        # ── 이미 수집된 article_id 목록 (재실행 안전)
        existing_ids = _existing_article_ids()
        print(f"[INFO] 기존 수집 항목: {len(existing_ids)}개 (skip 대상)\n")

        # ── 1단계: 게시글 목록 (신규 max_articles건 도달 시 조기 종료)
        print("게시글 목록 수집 중...\n")
        articles = get_user_articles(
            driver, member_hash, max_pages,
            existing_ids=existing_ids, max_new=max_articles,
        )
        print(f"\n{len(articles)}건 게시글 처리 시작\n")

        results = []
        failed = []

        for i, article in enumerate(articles):
            print(f"[{i+1}/{len(articles)}] {article['title'][:50]}")

            # ── 2단계: 견적의뢰 링크
            estimate_link = get_estimate_link_from_post(driver, article["url"])
            if not estimate_link:
                print("  [SKIP] 의뢰글 링크 없음 → 건너뜀\n")
                continue

            # article_id 미리 추출해서 중복 skip
            m = re.search(r'articleid(?:%3D|=)(\d+)', estimate_link, re.IGNORECASE)
            article_id = m.group(1) if m else str(i)

            if article_id in existing_ids:
                print(f"  [SKIP] 이미 수집됨: {article_id}\n")
                continue

            print(f"  의뢰글: {estimate_link}")

            # ── 3단계: 의뢰 원문 수집 (실패 시 None)
            detail = get_estimate_detail(driver, estimate_link)
            if detail is None:
                print(f"  [FAIL] 본문 수집 실패 → skip\n")
                failed.append({"article_id": article_id, "estimate_url": estimate_link})
                continue

            preview = detail["body_text"][:200].replace("\n", " ")
            print(f"  견적서 미리보기: {preview}")

            # ── 3-2단계: 견적의뢰글 원문 수집
            if detail.get("request_url"):
                print(f"  의뢰글 수집: {detail['request_url']}")
                detail["request_body_text"] = get_request_body(driver, detail["request_url"])
                if detail["request_body_text"]:
                    preview2 = detail["request_body_text"][:200].replace("\n", " ")
                    print(f"  의뢰글 미리보기: {preview2}")
                time.sleep(SLEEP_SEC)
            else:
                detail["request_body_text"] = ""

            # ── 3-3단계: 본문 내 링크된 견적서 페이지 이미지 수집
            # (래퍼 포스트가 실제 견적 이미지를 링크로만 가리키는 경우 대응)
            if detail.get("body_text"):
                linked_imgs = collect_linked_images(driver, detail["body_text"])
                if linked_imgs:
                    existing = set(detail.get("image_urls", []))
                    added = [u for u in linked_imgs if u not in existing]
                    detail["image_urls"] = detail.get("image_urls", []) + added
                    print(f"  링크 이미지 추가: {len(added)}개")

            # ── 4단계: 이미지 다운로드
            # 지역 감지 → 지역별 폴더 결정
            region = detect_region(detail)
            article_dir = os.path.join(BASE_DIR, region, str(article_id))
            print(f"  지역: {region}")

            if detail.get("image_urls"):
                print(f"  이미지 {len(detail['image_urls'])}개 다운로드 중...")
                detail["local_images"] = download_images(detail["image_urls"], article_dir)
            else:
                detail["local_images"] = []
                print("  이미지 없음")

            record = {
                "article_id":   article_id,
                "region":       region,
                "post_title":   article["title"],
                "post_url":     article["url"],
                "estimate_url": estimate_link,
                **detail,
            }

            # ── 개별 JSON 저장 ({region}/{article_id}/{article_id}.json)
            os.makedirs(article_dir, exist_ok=True)
            json_path = os.path.join(article_dir, f"{article_id}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            results.append(record)
            existing_ids.add(article_id)

            print(f"  완료 (이미지 {len(detail['local_images'])}장 저장)\n")
            time.sleep(SLEEP_SEC)

        # ── JSON 저장
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("=" * 50)
        print(f"전체 완료: {len(results)}건 수집")
        if failed:
            print(f"[FAIL] 수집 실패 {len(failed)}건:")
            for f_item in failed:
                print(f"  - {f_item['article_id']}  {f_item['estimate_url']}")
        print(f"JSON  → {OUTPUT_FILE}")
        print(f"데이터 → {BASE_DIR}/{{지역}}/")
        print("=" * 50)

        return results

    finally:
        driver.quit()


# ══════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 여기에 수집할 유저의 member_hash 입력
    MEMBER_HASH = "CGUcEN20XRZl8bWyApIj-A"

    crawl_user(MEMBER_HASH, max_pages=MAX_PAGES, max_articles=50)