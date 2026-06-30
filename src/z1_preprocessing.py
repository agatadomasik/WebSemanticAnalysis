import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource.startswith("punkt") else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from src.config import MIN_TOKENS, MIN_TOKEN_LEN

ROOT = Path(__file__).parent.parent
INDEX_JSON = ROOT / "data" / "raw" / "index.json"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_OUT = PROCESSED_DIR / "corpus.json"

STOP_WORDS = set(stopwords.words("english"))

EXTRA_STOP = {
    "cookie", "cookies", "privacy", "copyright", "menu", "navigation",
    "search", "login", "home", "page", "click", "javascript", "skip",
    "content", "link", "links", "site", "website", "email", "contact",
    "university", "warwick",  # very frequent, little information
    "false","true", "chat", "tag", "status", "window", "open", "location"
}
STOP_WORDS |= EXTRA_STOP

stemmer = SnowballStemmer("english")


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # remove navigation elements by class/id
    for tag in soup.find_all(True, {"class": re.compile(r"nav|menu|sidebar|footer|header|breadcrumb", re.I)}):
        tag.decompose()
    for tag in soup.find_all(True, {"id": re.compile(r"nav|menu|sidebar|footer|header|breadcrumb", re.I)}):
        tag.decompose()

    text = soup.get_text(separator=" ")
    # remove whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"[a-zA-Z]+", text.lower())

    tokens = []
    for t in raw_tokens:
        if len(t) < MIN_TOKEN_LEN:
            continue
        if t in STOP_WORDS:
            continue
        stemmed = stemmer.stem(t)
        if stemmed in STOP_WORDS:
            continue
        tokens.append(stemmed)

    return tokens


def build_corpus() -> list[dict]:
    with open(INDEX_JSON, encoding="utf-8") as f:
        index = json.load(f)

    corpus = []
    rejected = 0

    for url, filepath in index.items():
        try:
            html = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except Exception:
            rejected += 1
            continue

        text = extract_text(html)
        tokens = tokenize(text)

        if len(tokens) < MIN_TOKENS:
            rejected += 1
            continue

        corpus.append({"url": url, "tokens": tokens})

    print(f"Documents accepted: {len(corpus)}, rejected: {rejected}")
    return corpus


def save_corpus(corpus: list[dict]):
    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f"Corpus saved: {CORPUS_OUT}  ({len(corpus)} documents)")


if __name__ == "__main__":
    corpus = build_corpus()
    save_corpus(corpus)