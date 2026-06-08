"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Sử dụng crawl4ai để crawl bài báo. Mỗi bài lưu 1 file JSON.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Danh sách URL bài báo về nghệ sĩ liên quan tới ma tuý
ARTICLE_URLS = [
    "https://vnexpress.net/thi-truong/mac-hong-quan-bi-bat-vi-lien-quan-ma-tuy-4795060.html",
    "https://vnexpress.net/phap-luat/kien-nghi-khoi-to-vu-an-ma-tuy-lien-quan-den-truong-my-lan-4818900.html",
    "https://thanhnien.vn/nghi-sy-bi-phat-hien-su-dung-ma-tuy-trong-khach-san-185240528095140688.htm",
    "https://tuoitre.vn/di-en-ma-co-ma-tuy-trong-nguoi-nghe-si-bi-bat-20231015172345067.htm",
    "https://vnexpress.net/giai-tri/nghe-si-viet-va-hinh-anh-tieu-cuc-lien-quan-ma-tuy-4200000.html",
    "https://www.24h.com.vn/tin-tuc-trong-ngay/nghe-si-bi-bat-vi-ma-tuy-c46a1500000.html",
    "https://dantri.com.vn/giai-tri/nhieu-nghe-si-viet-lien-quan-den-ma-tuy-20231001120000000.htm",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            title = "Unknown"
            if result.metadata and result.metadata.get("title"):
                title = result.metadata["title"]
            elif result.markdown:
                # Extract title from first H1 heading
                for line in result.markdown.split("\n"):
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": result.markdown or "",
            }
    except Exception as e:
        # Fallback: dùng requests nếu crawl4ai chưa cài
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Pipeline/1.0)"}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.encoding = "utf-8"
            content = resp.text[:5000]  # Lấy một phần nội dung
            return {
                "url": url,
                "title": "Article from " + url.split("/")[-1][:50],
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": content,
            }
        except Exception as e2:
            return {
                "url": url,
                "title": "Failed to crawl",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": f"Error: {e2}",
            }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url[:80]}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  OK Saved: {filepath} | title: {article['title'][:60]}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
