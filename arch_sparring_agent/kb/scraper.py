"""Scrape AWS Well-Architected Framework documentation into per-best-practice markdown files."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import yaml

logger = logging.getLogger(__name__)

WAF_BASE = "https://docs.aws.amazon.com/wellarchitected/latest/framework/"
LENS_INDEX = "https://docs.aws.amazon.com/wellarchitected/latest/"

PILLARS = {
    "operational-excellence": "Operational Excellence",
    "security": "Security",
    "reliability": "Reliability",
    "performance-efficiency": "Performance Efficiency",
    "cost-optimization": "Cost Optimization",
    "sustainability": "Sustainability",
}

OFFICIAL_LENSES = [
    "serverlessapps-lens",
    "saas-lens",
    "data-analytics-lens",
    "machine-learning-lens",
    "iot-lens",
    "container-build-lens",
    "games-industry-lens",
    "financial-services-industry-lens",
    "healthcare-industry-lens",
]

REQUEST_DELAY = 1.5  # seconds between HTTP requests


@dataclass
class ScrapedPage:
    source: str
    pillar: str
    question_id: str
    title: str
    content: str
    url: str
    extra_metadata: dict = field(default_factory=dict)


def _fetch(url: str) -> str | None:
    """Fetch a page with rate limiting. Returns HTML or None on error."""
    try:
        import requests
    except ImportError:
        import urllib.request

        try:
            time.sleep(REQUEST_DELAY)
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode()
        except Exception:
            logger.warning("Failed to fetch: %s", url)
            return None

    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "arch-review-scraper/1.0"})
        resp.raise_for_status()
        return resp.text
    except Exception:
        logger.warning("Failed to fetch: %s", url)
        return None


def _html_to_markdown(html: str) -> str:
    """Convert HTML to clean markdown."""
    try:
        import html2text

        h = html2text.HTML2Text()
        h.body_width = 0
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        return h.handle(html).strip()
    except ImportError:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n").strip()


def _extract_content(html: str) -> str:
    """Extract the main content div from an AWS docs page."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _html_to_markdown(html)

    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(id="main-col-body") or soup.find("main") or soup.find("article")
    if not main:
        return _html_to_markdown(html)
    return _html_to_markdown(str(main))


def _extract_links(html: str, base_url: str, pattern: str | None = None) -> list[str]:
    """Extract links from HTML page, optionally filtered by regex pattern."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if pattern and not re.search(pattern, full_url):
            continue
        if full_url not in links:
            links.append(full_url)
    return links


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name).strip("_")[:80]


def _derive_question_id(url: str, pillar: str) -> str:
    """Try to extract a question/BP ID from the URL path."""
    path = url.rstrip("/").rsplit("/", 1)[-1]
    path = path.replace(".html", "")
    if re.match(r"[a-z]{2,5}\d{2}-bp\d{2}", path, re.IGNORECASE):
        return path.upper()
    return f"{pillar[:3].upper()}-{_sanitize_filename(path)}"


# ---------------------------------------------------------------------------
# Pillar scraping
# ---------------------------------------------------------------------------


def _scrape_pillar(pillar_slug: str, pillar_name: str) -> list[ScrapedPage]:
    """Scrape all best-practice pages for a single WAF pillar."""
    pillar_url = urljoin(WAF_BASE, f"{pillar_slug}.html")
    logger.info("Scraping pillar: %s (%s)", pillar_name, pillar_url)

    html = _fetch(pillar_url)
    if not html:
        return []

    bp_links = _extract_links(html, pillar_url, pattern=r"bp\d{2}")
    if not bp_links:
        bp_links = _extract_links(html, pillar_url, pattern=pillar_slug)
        bp_links = [lnk for lnk in bp_links if lnk != pillar_url]

    pages: list[ScrapedPage] = []

    if not bp_links:
        content = _extract_content(html)
        if content:
            pages.append(ScrapedPage(
                source="well-architected-framework",
                pillar=pillar_slug,
                question_id=f"{pillar_slug[:3].upper()}-OVERVIEW",
                title=pillar_name,
                content=content,
                url=pillar_url,
            ))
        return pages

    for link in bp_links:
        bp_html = _fetch(link)
        if not bp_html:
            continue
        content = _extract_content(bp_html)
        if not content:
            continue

        question_id = _derive_question_id(link, pillar_slug)

        try:
            from bs4 import BeautifulSoup

            title_tag = BeautifulSoup(bp_html, "html.parser").find("title")
            title = title_tag.get_text(strip=True) if title_tag else question_id
        except ImportError:
            title = question_id

        pages.append(ScrapedPage(
            source="well-architected-framework",
            pillar=pillar_slug,
            question_id=question_id,
            title=title,
            content=content,
            url=link,
        ))

    logger.info("Scraped %d pages from pillar: %s", len(pages), pillar_name)
    return pages


# ---------------------------------------------------------------------------
# Lens scraping
# ---------------------------------------------------------------------------


def _scrape_lens(lens_slug: str) -> list[ScrapedPage]:
    """Scrape an official lens landing page and its sub-pages."""
    lens_url = f"https://docs.aws.amazon.com/wellarchitected/latest/{lens_slug}/welcome.html"
    logger.info("Scraping lens: %s", lens_slug)

    html = _fetch(lens_url)
    if not html:
        alt_url = f"https://docs.aws.amazon.com/wellarchitected/latest/{lens_slug}/"
        html = _fetch(alt_url)
        if not html:
            return []
        lens_url = alt_url

    sub_links = _extract_links(html, lens_url, pattern=lens_slug)
    sub_links = [lnk for lnk in sub_links if lnk != lens_url]

    pages: list[ScrapedPage] = []

    main_content = _extract_content(html)
    if main_content:
        pages.append(ScrapedPage(
            source=lens_slug,
            pillar="lens",
            question_id=f"{lens_slug}-overview",
            title=lens_slug.replace("-", " ").title(),
            content=main_content,
            url=lens_url,
        ))

    for link in sub_links[:50]:
        sub_html = _fetch(link)
        if not sub_html:
            continue
        content = _extract_content(sub_html)
        if not content or len(content) < 50:
            continue

        question_id = _derive_question_id(link, lens_slug)

        try:
            from bs4 import BeautifulSoup

            title_tag = BeautifulSoup(sub_html, "html.parser").find("title")
            title = title_tag.get_text(strip=True) if title_tag else question_id
        except ImportError:
            title = question_id

        pages.append(ScrapedPage(
            source=lens_slug,
            pillar="lens",
            question_id=question_id,
            title=title,
            content=content,
            url=link,
        ))

    logger.info("Scraped %d pages from lens: %s", len(pages), lens_slug)
    return pages


# ---------------------------------------------------------------------------
# Write to disk
# ---------------------------------------------------------------------------


def _write_page(page: ScrapedPage, output_dir: Path) -> Path:
    """Write a single scraped page as a markdown file with YAML frontmatter."""
    filename = _sanitize_filename(f"{page.question_id}.md")
    sub_dir = output_dir / _sanitize_filename(page.source) / _sanitize_filename(page.pillar)
    sub_dir.mkdir(parents=True, exist_ok=True)
    filepath = sub_dir / filename

    frontmatter = {
        "source": page.source,
        "pillar": page.pillar,
        "question": page.question_id,
        "title": page.title,
        "url": page.url,
    }
    if page.extra_metadata:
        frontmatter.update(page.extra_metadata)

    fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = f"---\n{fm_text}---\n\n"
    content += f"# {page.title}\n\n{page.content}\n"

    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scrape_waf(output_dir: str | Path) -> int:
    """Scrape all 6 WAF pillars and official lenses.

    Args:
        output_dir: Directory to write markdown files to.

    Returns:
        Total number of pages scraped.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0

    for slug, name in PILLARS.items():
        pages = _scrape_pillar(slug, name)
        for page in pages:
            _write_page(page, output_path)
        total += len(pages)

    for lens_slug in OFFICIAL_LENSES:
        pages = _scrape_lens(lens_slug)
        for page in pages:
            _write_page(page, output_path)
        total += len(pages)

    logger.info("Scraping complete — %d total pages written to %s", total, output_path)
    return total
