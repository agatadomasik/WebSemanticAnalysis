import json
import time
import hashlib
import requests
from pathlib import Path

from src.config import CRAWL_DELAY, CRAWL_TIMEOUT, CRAWL_MAX_PAGES

ROOT = Path(__file__).parent.parent
GRAPH_JSON = ROOT / "data" / "raw" / "graph.json"
PAGES_DIR = ROOT / "data" / "raw" / "pages"

DELAY = CRAWL_DELAY
TIMEOUT = CRAWL_TIMEOUT
MAX_PAGES = CRAWL_MAX_PAGES
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-crawler/1.0)"
}


def url_to_filename(url: str) -> str:
    h = hashlib.md5(url.encode()).hexdigest()
    return h + ".html"


def load_urls(graph_path: Path) -> list[str]:
    with open(graph_path, encoding="utf-8") as f:
        data = json.load(f)
    urls = list({entry["Url"] for entry in data})
    print(f"{len(urls)} unique URL's found in graph.json")
    return urls


def fetch_and_save(urls: list[str]) -> dict[str, str]:
    url_to_file = {}
    skipped = 0
    failed = 0

    for i, url in enumerate(urls[:MAX_PAGES]):
        filename = url_to_filename(url)
        filepath = PAGES_DIR / filename

        if filepath.exists():
            url_to_file[url] = str(filepath)
            skipped += 1
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if resp.status_code == 200:
                filepath.write_text(resp.text, encoding="utf-8", errors="replace")
                url_to_file[url] = str(filepath)
            else:
                failed += 1
        except Exception as e:
            failed += 1

        # progress
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(urls)}] downloaded: {len(url_to_file)}, errors: {failed}")

        time.sleep(DELAY)

    print(f"\nDone. Downloaded: {len(url_to_file)}, errors: {failed}")
    return url_to_file


def save_index(url_to_file: dict[str, str]):
    # saves index {url: file} in data/raw/index.json
    index_path = ROOT / "data" / "raw" / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(url_to_file, f, ensure_ascii=False, indent=2)
    print(f"Index saved: {index_path}")


if __name__ == "__main__":
    urls = load_urls(GRAPH_JSON)
    url_to_file = fetch_and_save(urls)
    save_index(url_to_file)