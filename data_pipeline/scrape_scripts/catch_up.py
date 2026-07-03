import os
import json
import asyncio
import aiohttp
import feedparser
import trafilatura
import hashlib
import subprocess
import urllib.request
import tarfile
import io
from datetime import datetime, UTC
from pathlib import Path

# 1. The Missing Category 2 Targets + 0x00sec RSS
MISSING_FEEDS = {
    "microsoft_security": "https://www.microsoft.com/en-us/security/blog/feed/",
    "gynvael":            "https://gynvael.coldwind.pl/rss_en.php",
    "lcamtuf":            "https://lcamtuf.blogspot.com/feeds/posts/default",
    "immunity_canvas":    "https://immunityinc.com/resources/rss.xml",
    "0x00sec":            "https://0x00sec.org/latest.rss"
}

HARDENED_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

SEEN_HASHES = set()
RAW_DIR = Path("data_pipeline/raw/pretrain")
RAW_DIR.mkdir(parents=True, exist_ok=True)

async def fetch_article(session, url, source_name, out_f):
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200: return
            html = await resp.text()
            
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if not text or len(text) < 250: return
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in SEEN_HASHES: return
        SEEN_HASHES.add(url_hash)
        
        record = {
            "text": text, "source": source_name, "url": url,
            "date": datetime.now(UTC).isoformat(), "type": "blog_post"
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

async def pull_missing_rss(output_path):
    print("[INFO] Phase 1: Pulling Missing RSS Feeds...")
    async with aiohttp.ClientSession(headers=HARDENED_HEADERS) as session:
        with open(output_path, "a", encoding="utf-8") as out_f:
            for source, url in MISSING_FEEDS.items():
                try:
                    feed = feedparser.parse(url)
                    tasks = [fetch_article(session, entry.link, source, out_f) 
                             for entry in feed.entries if hasattr(entry, "link")]
                    await asyncio.gather(*tasks)
                    print(f"  → Collected recent articles for {source}")
                except Exception:
                    pass

def pull_exploitdb(output_path):
    print("\n[INFO] Phase 2: Pulling Exploit-DB Papers Offline Repository...")
    clone_dir = RAW_DIR / "cloned_repos" / "exploitdb-papers"
    
    if not clone_dir.exists():
        subprocess.run(["git", "clone", "--depth=1", "https://github.com/offensive-security/exploitdb-papers.git", str(clone_dir)], check=True)
    
    count = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for text_file in clone_dir.rglob("*"):
            if text_file.is_file() and text_file.suffix in ['.txt', '.md']:
                try:
                    text = text_file.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(text) < 100: continue
                    record = {"text": text, "source": "exploit_db_papers", "url": f"local://exploitdb/{text_file.name}", "type": "exploit_paper"}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                except Exception:
                    pass
    print(f"  → Parsed {count} Exploit-DB papers.")

def pull_phrack(output_path):
    print("\n[INFO] Phase 3: Streaming Phrack Magazine Archives...")
    req = urllib.request.Request("http://www.phrack.org/archives/tgz/", headers=HARDENED_HEADERS)
    count = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        # Loop through issues 1 to 70
        for issue in range(1, 71): 
            url = f"http://www.phrack.org/archives/tgz/phrack{issue}.tar.gz"
            try:
                req = urllib.request.Request(url, headers=HARDENED_HEADERS)
                with urllib.request.urlopen(req, timeout=10) as response:
                    with tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz") as tar:
                        for member in tar.getmembers():
                            if member.isfile() and member.name.endswith(".txt"):
                                text = tar.extractfile(member).read().decode("utf-8", errors="ignore").strip()
                                if len(text) > 100:
                                    record = {"text": text, "source": "phrack_magazine", "url": f"phrack://issue_{issue}/{member.name}", "type": "ezine"}
                                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    count += 1
                print(f"  → Extracted Issue {issue}")
            except Exception:
                pass
    print(f"  → Parsed {count} total Phrack articles.")

if __name__ == "__main__":
    out_file = RAW_DIR / "catchup_data.jsonl"
    asyncio.run(pull_missing_rss(out_file))
    pull_exploitdb(out_file)
    pull_phrack(out_file)
    print(f"\n[SUCCESS] Catch-up data saved to {out_file}!")
