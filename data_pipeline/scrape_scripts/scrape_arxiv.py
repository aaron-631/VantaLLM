import arxiv
import json
from pathlib import Path

def collect_arxiv_cscr(output_path: str, max_results: int = 15000):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    
    search = arxiv.Search(
        query="cat:cs.CR",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    print(f"[INFO] Fetching up to {max_results} arXiv cs.CR papers...")
    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for paper in client.results(search):
            record = {
                "text": f"{paper.title}\n\n{paper.summary}",
                "source": "arxiv_cs_cr",
                "url": paper.entry_id,
                "date": paper.published.isoformat(),
                "type": "research_paper",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"  → Fetched {count} papers...")
    
    print(f"[SUCCESS] arXiv collection complete. Total: {count}")

if __name__ == "__main__":
    collect_arxiv_cscr("data_pipeline/raw/pretrain/arxiv_cscr.jsonl")
