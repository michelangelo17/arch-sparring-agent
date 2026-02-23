"""Scrape AWS Well-Architected Framework documentation into per-best-practice markdown files."""

from __future__ import annotations

import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import html2text
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

WAF_BASE = "https://docs.aws.amazon.com/wellarchitected/latest/framework/"

PILLARS = {
    "operational-excellence": "Operational Excellence",
    "security": "Security",
    "reliability": "Reliability",
    "performance-efficiency": "Performance Efficiency",
    "cost-optimization": "Cost Optimization",
    "sustainability": "Sustainability",
}

OFFICIAL_LENSES = {
    # Technology & architecture
    "serverless-applications-lens": "Serverless Applications",
    "saas-lens": "SaaS",
    "analytics-lens": "Data Analytics",
    "machine-learning-lens": "Machine Learning",
    "generative-ai-lens": "Generative AI",
    "iot-lens": "IoT",
    "container-build-lens": "Container Build",
    "high-performance-computing-lens": "High Performance Computing",
    "streaming-media-lens": "Streaming Media",
    "amazon-opensearch-service-lens": "Amazon OpenSearch Service",
    "devops-guidance": "DevOps",
    "migration-lens": "Migration",
    # Industry
    "games-industry-lens": "Games Industry",
    "financial-services-industry-lens": "Financial Services",
    "healthcare-industry-lens": "Healthcare",
    "government-lens": "Government",
    "sap-lens": "SAP",
    "supply-chain-lens": "Supply Chain",
    "connected-mobility-lens": "Connected Mobility",
    "modern-industrial-data-technology-lens": "Modern Industrial Data Technology",
    "mergers-and-acquisitions-lens": "Mergers and Acquisitions",
}

REQUEST_DELAY = 1.0

_converter = html2text.HTML2Text()
_converter.body_width = 0
_converter.ignore_links = False
_converter.ignore_images = True
_converter.ignore_emphasis = False


@dataclass
class ScrapedPage:
    source: str
    pillar: str
    page_id: str
    title: str
    content: str
    url: str
    extra_metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fetching and parsing
# ---------------------------------------------------------------------------


def _fetch(url: str) -> str | None:
    """Fetch a page with rate limiting. Returns HTML or None on error."""
    time.sleep(REQUEST_DELAY)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "arch-review-scraper/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type:
                return None
            return resp.read().decode()
    except (urllib.error.URLError, OSError, TimeoutError):
        logger.warning("Failed to fetch: %s", url)
        return None


def _html_to_markdown(html: str) -> str:
    return _converter.handle(html).strip()


def _extract_content(html: str) -> str:
    """Extract the main content div from an AWS docs page."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(id="main-col-body") or soup.find("main") or soup.find("article")
    if not main:
        return _html_to_markdown(html)
    return _html_to_markdown(str(main))


def _extract_title(html: str, fallback: str) -> str:
    title_tag = BeautifulSoup(html, "html.parser").find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        text = re.sub(r"\s*-\s*AWS Well-Architected Framework$", "", text)
        return text or fallback
    return fallback


def _extract_html_links(html: str, base_url: str, scope: str) -> list[str]:
    """Extract unique HTML links within the given scope URL prefix.

    Filters out PDFs, anchors-only, and external links.
    """
    soup = BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    links: list[str] = []

    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"])
        if href.startswith("#"):
            continue
        full_url = urljoin(base_url, href)
        full_url = full_url.split("#")[0]

        if not full_url.startswith(scope):
            continue
        if not full_url.endswith(".html"):
            continue
        if "pdf" in full_url.lower():
            continue
        if full_url == base_url or full_url in seen:
            continue

        seen.add(full_url)
        links.append(full_url)

    return links


def _page_id_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    return path.replace(".html", "")


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name).strip("_")[:80]


# ---------------------------------------------------------------------------
# 2-hop crawl
# ---------------------------------------------------------------------------


def _crawl(start_url: str, scope: str, max_depth: int = 3) -> list[str]:
    """Crawl from start_url up to *max_depth* hops, staying within scope.

    Returns all unique discovered page URLs (excluding the start page).
    """
    html = _fetch(start_url)
    if not html:
        return []

    all_urls: dict[str, None] = {}
    frontier = _extract_html_links(html, start_url, scope)
    for url in frontier:
        all_urls[url] = None

    for _depth in range(2, max_depth + 1):
        next_frontier: list[str] = []
        for url in frontier:
            sub_html = _fetch(url)
            if not sub_html:
                continue
            for link in _extract_html_links(sub_html, url, scope):
                if link != start_url and link not in all_urls:
                    all_urls[link] = None
                    next_frontier.append(link)
        frontier = next_frontier
        if not frontier:
            break

    return list(all_urls)


# ---------------------------------------------------------------------------
# Pillar scraping
# ---------------------------------------------------------------------------


def _scrape_pillar(pillar_slug: str, pillar_name: str) -> list[ScrapedPage]:
    """Scrape all best-practice pages for a single WAF pillar."""
    pillar_url = urljoin(WAF_BASE, f"{pillar_slug}.html")
    logger.info("Scraping pillar: %s (%s)", pillar_name, pillar_url)

    page_urls = _crawl(pillar_url, WAF_BASE)
    if not page_urls:
        logger.warning("No sub-pages found for pillar: %s", pillar_name)
        return []

    pages: list[ScrapedPage] = []
    for url in page_urls:
        html = _fetch(url)
        if not html:
            continue
        content = _extract_content(html)
        if not content or len(content) < 100:
            continue

        page_id = _page_id_from_url(url)
        title = _extract_title(html, page_id)

        pages.append(
            ScrapedPage(
                source="well-architected-framework",
                pillar=pillar_slug,
                page_id=page_id,
                title=title,
                content=content,
                url=url,
            )
        )

    logger.info("Scraped %d pages from pillar: %s", len(pages), pillar_name)
    return pages


# ---------------------------------------------------------------------------
# Lens scraping
# ---------------------------------------------------------------------------

LENS_URL_PATTERNS = [
    "https://docs.aws.amazon.com/wellarchitected/latest/{slug}/welcome.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/{slug}/{slug}.html",
    "https://docs.aws.amazon.com/wellarchitected/latest/{slug}/",
]


def _is_redirect_page(html: str) -> str | None:
    """Detect meta-refresh redirect pages and return the target URL (relative)."""
    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"http-equiv": re.compile(r"refresh", re.I)})
    if meta:
        content = str(meta.get("content", ""))
        match = re.search(r"URL=([^\s\"']+)", content, re.I)
        if match:
            return match.group(1)
    return None


def _find_lens_landing(slug: str) -> tuple[str, str | None]:
    """Try known URL patterns for a lens. Returns (url, html) or (url, None).

    Follows meta-refresh redirects (common on ``welcome.html`` pages).
    """
    for pattern in LENS_URL_PATTERNS:
        url = pattern.format(slug=slug)
        html = _fetch(url)
        if not html:
            continue

        redirect_target = _is_redirect_page(html)
        if redirect_target:
            resolved = urljoin(url, redirect_target)
            resolved_html = _fetch(resolved)
            if resolved_html:
                return resolved, resolved_html
            continue

        return url, html

    return "", None


def _scrape_lens(lens_slug: str, lens_name: str) -> list[ScrapedPage]:
    """Scrape an official lens and its sub-pages."""
    logger.info("Scraping lens: %s", lens_name)

    lens_scope = f"https://docs.aws.amazon.com/wellarchitected/latest/{lens_slug}/"
    landing_url, landing_html = _find_lens_landing(lens_slug)
    if not landing_html:
        logger.warning("Could not find lens: %s", lens_slug)
        return []

    pages: list[ScrapedPage] = []

    landing_content = _extract_content(landing_html)
    if landing_content and len(landing_content) >= 100:
        pages.append(
            ScrapedPage(
                source=lens_slug,
                pillar="lens",
                page_id=_page_id_from_url(landing_url),
                title=_extract_title(landing_html, lens_name),
                content=landing_content,
                url=landing_url,
            )
        )

    page_urls = _crawl(landing_url, lens_scope)
    for url in page_urls:
        html = _fetch(url)
        if not html:
            continue
        content = _extract_content(html)
        if not content or len(content) < 100:
            continue

        page_id = _page_id_from_url(url)
        title = _extract_title(html, page_id)

        pages.append(
            ScrapedPage(
                source=lens_slug,
                pillar="lens",
                page_id=page_id,
                title=title,
                content=content,
                url=url,
            )
        )

    logger.info("Scraped %d pages from lens: %s", len(pages), lens_name)
    return pages


# ---------------------------------------------------------------------------
# Write to disk
# ---------------------------------------------------------------------------


def _write_page(page: ScrapedPage, output_dir: Path) -> Path:
    """Write a single scraped page as a markdown file with YAML frontmatter."""
    filename = _sanitize_filename(f"{page.page_id}.md")
    sub_dir = output_dir / _sanitize_filename(page.source) / _sanitize_filename(page.pillar)
    sub_dir.mkdir(parents=True, exist_ok=True)
    filepath = sub_dir / filename

    frontmatter = {
        "source": page.source,
        "pillar": page.pillar,
        "page_id": page.page_id,
        "title": page.title,
        "url": page.url,
    }
    if page.extra_metadata:
        frontmatter.update(page.extra_metadata)

    fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
    content = f"---\n{fm_text}---\n\n# {page.title}\n\n{page.content}\n"

    filepath.write_text(content, encoding="utf-8")
    return filepath


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scrape_waf(output_dir: str | Path) -> int:
    """Scrape all 6 WAF pillars and official lenses.

    Returns the total number of pages scraped.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0

    for slug, name in PILLARS.items():
        pages = _scrape_pillar(slug, name)
        for page in pages:
            _write_page(page, output_path)
        total += len(pages)

    for slug, name in OFFICIAL_LENSES.items():
        pages = _scrape_lens(slug, name)
        for page in pages:
            _write_page(page, output_path)
        total += len(pages)

    logger.info("Scraping complete — %d total pages written to %s", total, output_path)
    return total
