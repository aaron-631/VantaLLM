import asyncio
import aiohttp
import feedparser
import trafilatura
import json
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from bs4 import BeautifulSoup

FEEDS_AND_SITEMAPS = {
    "google_project_zero": {
        "rss": "https://googleprojectzero.blogspot.com/feeds/posts/default",
        "sitemap": "https://googleprojectzero.blogspot.com/sitemap.xml"
    },
    "google_security_blog": {
        "rss": "https://security.googleblog.com/feeds/posts/default",
        "sitemap": "https://security.googleblog.com/sitemap.xml"
    },
    "talos_intelligence": {
        "rss": "https://blog.talosintelligence.com/rss/",
        "sitemap": "https://blog.talosintelligence.com/sitemap.xml"
    },
    "unit42": {
        "rss": "https://unit42.paloaltonetworks.com/feed/",
        "sitemap": "https://unit42.paloaltonetworks.com/sitemap_index.xml"
    },
    "crowdstrike": {
        "rss": "https://www.crowdstrike.com/blog/feed/",
        "sitemap": "https://www.crowdstrike.com/sitemap_index.xml"
    },
    "mandiant": {
        "rss": "https://www.mandiant.com/resources/blog/rss.xml",
        "sitemap": "https://www.mandiant.com/sitemap.xml"
    },
    "securelist": {
        "rss": "https://securelist.com/feed/",
        "sitemap": "https://securelist.com/sitemap.xml"
    },
    "welivesecurity": {
        "rss": "https://www.welivesecurity.com/feed/",
        "sitemap": "https://www.welivesecurity.com/sitemap_index.xml"
    },
    "sentinelone": {
        "rss": "https://www.sentinelone.com/blog/feed/",
        "sitemap": "https://www.sentinelone.com/sitemap_index.xml"
    },
    "rapid7": {
        "rss": "https://www.rapid7.com/blog/rss/",
        "sitemap": "https://www.rapid7.com/sitemap_index.xml"
    },
    "portswigger": {
        "rss": "https://portswigger.net/research/rss",
        "sitemap": "https://portswigger.net/sitemap.xml"
    },
    "nccgroup": {
        "rss": "https://research.nccgroup.com/feed/",
        "sitemap": "https://research.nccgroup.com/sitemap_index.xml"
    },
    "trailofbits": {
        "rss": "https://blog.trailofbits.com/feed/",
        "sitemap": "https://blog.trailofbits.com/sitemap_index.xml"
    },
    "krebsonsecurity": {
        "rss": "https://krebsonsecurity.com/feed/",
        "sitemap": "https://krebsonsecurity.com/sitemap.xml"
    },
    "schneier": {
        "rss": "https://www.schneier.com/blog/atom.xml",
        "sitemap": "https://www.schneier.com/sitemap.xml"
    },
    "bleepingcomputer": {
        "rss": "https://www.bleepingcomputer.com/feed/",
        "sitemap": "https://www.bleepingcomputer.com/sitemap_index.xml"
    },
    "darkreading": {
        "rss": "https://www.darkreading.com/rss/all.xml",
        "sitemap": "https://www.darkreading.com/sitemap.xml"
    },
    "sans_isc": {
        "rss": "https://isc.sans.edu/rssfeed_full.xml",
        "sitemap": "https://isc.sans.edu/sitemap.xml"
    },
    "thehackernews": {
        "rss": "https://feeds.feedburner.com/TheHackersNews",
        "sitemap": "https://thehackernews.com/sitemap.xml"
    },
    "recorded_future": {
        "rss": "https://www.recordedfuture.com/blog/rss.xml",
        "sitemap": "https://www.recordedfuture.com/sitemap.xml"
    },
    "tenable": {
        "rss": "https://www.tenable.com/blog/feed",
        "sitemap": "https://www.tenable.com/sitemap.xml"
    },
    "qualys": {
        "rss": "https://blog.qualys.com/feed",
        "sitemap": "https://blog.qualys.com/sitemap.xml"
    },
    "offsec": {
        "rss": "https://www.offsec.com/blog/feed/",
        "sitemap": "https://www.offsec.com/sitemap.xml"
    },
    "cloudflare_security": {
        "rss": "https://blog.cloudflare.com/tag/security/rss",
        "sitemap": "https://blog.cloudflare.com/sitemap.xml"
    }
}

HARDENED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

SEEN_HASHES = set()

async def fetch_article(session, url, source_name):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text or len(text) < 250:
            return None
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in SEEN_HASHES:
            return None
        SEEN_HASHES.add(url_hash)
        return {
            "text": text,
            "source": source_name,
            "url": url,
            "date": datetime.now(UTC).isoformat(),
            "type": "blog_post"
        }
    except Exception:
        return None

async def extract_urls_from_sitemap(session, sitemap_url):
    urls = []
    try:
        async with session.get(sitemap_url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return urls
            xml_content = await resp.text()
        soup = BeautifulSoup(xml_content, "xml")
        locs = [loc.text.strip() for loc in soup.find_all("loc")]
        for loc in locs:
            if "sitemap" in loc and loc.endswith(".xml") and loc != sitemap_url:
                urls.extend(await extract_urls_from_sitemap(session, loc))
            else:
                urls.append(loc)
        return list(set(urls))
    except Exception:
        return urls

async def crawl_all_sources(output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector, headers=HARDENED_HEADERS) as session:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for source_name, targets in FEEDS_AND_SITEMAPS.items():
                print(f"[INFO] Collecting from source: {source_name}")
                target_urls = set()
                try:
                    feed = feedparser.parse(targets["rss"])
                    for entry in feed.entries:
                        if hasattr(entry, "link"):
                            target_urls.add(entry.link)
                except Exception:
                    pass
                
                sitemap_urls = await extract_urls_from_sitemap(session, targets["sitemap"])
                target_urls.update(sitemap_urls)
                
                if not target_urls:
                    continue
                
                tasks = [fetch_article(session, url, source_name) for url in target_urls]
                results = await asyncio.gather(*tasks)
                
                count = 0
                for record in results:
                    if record:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                print(f"  → Written {count}/{len(target_urls)} records for {source_name}")
                await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(crawl_all_sources("data_pipeline/raw/pretrain/security_blogs.jsonl"))
