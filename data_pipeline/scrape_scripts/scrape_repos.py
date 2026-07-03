import os
import json
import subprocess
from pathlib import Path

REPOS = {
    "hacktricks": "https://github.com/carlospolop/hacktricks",
    "payloads_all_the_things": "https://github.com/swisskyrepo/PayloadsAllTheThings",
}

def clone_and_extract(repos, clone_dir, output_path):
    Path(clone_dir).mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, "a", encoding="utf-8") as out_f:
        for name, url in repos.items():
            dest = os.path.join(clone_dir, name)
            if not os.path.exists(dest):
                print(f"[INFO] Cloning {name}...")
                subprocess.run(["git", "clone", "--depth=1", url, dest], check=True)
            
            for md_file in Path(dest).rglob("*.md"):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="ignore")
                    if len(text) < 100:
                        continue
                    record = {
                        "text": text,
                        "source": name,
                        "url": url,
                        "type": "security_reference",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                except Exception:
                    pass
    print(f"[SUCCESS] Repo extraction complete. Parsed {count} Markdown files.")

if __name__ == "__main__":
    clone_and_extract(REPOS, "data_pipeline/raw/pretrain/cloned_repos", "data_pipeline/raw/pretrain/security_repos.jsonl")
