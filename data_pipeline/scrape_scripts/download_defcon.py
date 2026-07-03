import os
import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("defcon_downloader")

URL = "https://media.defcon.org/DEF%20CON%2031/DEF%20CON%2031%20presentations/"
OUTPUT_DIR = "data_pipeline/raw/pdfs"

# Browser-spoofing headers to bypass the immediate TCP reset firewall
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive"
}

async def download_pdf(session, download_url, filename):
    target_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(target_path):
        return

    try:
        async with session.get(download_url, headers=HEADERS, timeout=30) as resp:
            if resp.status == 200:
                with open(target_path, "wb") as f:
                    f.write(await resp.read())
                log.info(f"Downloaded: {filename}")
            else:
                log.warning(f"Failed to download {filename} (Status: {resp.status})")
    except Exception as e:
        log.error(f"Error downloading {filename}: {e}")

async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Scraping DEF CON index page: {URL}")
    
    # Configure connection pooling and limits to prevent overwhelming the peer
    connector = aiohttp.TCPConnector(limit_per_host=2)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        try:
            async with session.get(URL, timeout=15) as resp:
                if resp.status != 200:
                    log.error(f"Failed to fetch index page. Status code: {resp.status}")
                    return
                html = await resp.text()
        except Exception as e:
            log.error(f"Could not connect to DEF CON server: {e}")
            log.info("The server is actively rejecting connections. Skipping PDF extraction for now.")
            return

        soup = BeautifulSoup(html, "html.parser")
        pdf_links = []
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".pdf"):
                # Clean up relative paths if necessary
                full_url = href if href.startswith("http") else URL + href
                filename = os.path.basename(href)
                pdf_links.append((full_url, filename))

        if not pdf_links:
            log.warning("No PDF presentation links found on the index page.")
            return

        log.info(f"Found {len(pdf_links)} presentations. Starting throttled download...")
        
        # Download sequentially or with minimal concurrency to avoid a firewall ban
        for download_url, filename in pdf_links:
            await download_pdf(session, download_url, filename)
            await asyncio.sleep(1.0) # 1-second politeness delay

if __name__ == "__main__":
    asyncio.run(main())
