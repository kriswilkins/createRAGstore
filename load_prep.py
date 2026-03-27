import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders import WebBaseLoader

# CONFIG
# CHECKPOINT_FILE used for saving progress
# OUTPUT_FILE where the urls and data will be exported
# MAX_WORKERS adjust number of concurrent scrapers
# RETRY_LIMIT adjust scrape tries per page in case of errors
# PROGRESS_INTERVAL adjust progress output interval to console
CHECKPOINT_FILE = "scrape_checkpoint.json"
OUTPUT_FILE = "docs.json"
MAX_WORKERS = 50
RETRY_LIMIT = 2
PROGRESS_INTERVAL = 10

# LOAD URLS
from urls_list import urls 

# LOAD CHECKPOINT
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
else:
    checkpoint = {}

# FILTER OUT ALREADY PROCESSED
pending_urls = [url for url in urls if url not in checkpoint]
total = len(pending_urls)
print(f"Starting load: {total} URLs remaining out of {len(urls)} total.")

# TRACKING
docs = []
success_count = 0
fail_count = 0
start_time = time.time()

def load_url_with_retry(url):
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            loader = WebBaseLoader(url)
            return loader.load()
        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(1)
    return []

# MAIN EXECUTION
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(load_url_with_retry, url): url for url in pending_urls}
    for idx, future in enumerate(as_completed(futures), start=1):
        url = futures[future]
        result = future.result()

        if result:
            docs.extend(result)
            success_count += 1
            checkpoint[url] = "success"
        else:
            fail_count += 1
            checkpoint[url] = "failed"

        # Checkpoint Progress
        if idx % PROGRESS_INTERVAL == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = elapsed / idx
            remaining = total - idx
            eta = remaining * rate
            print(f"{idx}/{total} done — {success_count} succeeded, {fail_count} failed — ETA: {int(eta)}s")

        # Save Checkpoint
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint, f, indent=2)

# FINAL OUTPUT
with open(OUTPUT_FILE, "w") as f:
    json.dump([doc.dict() for doc in docs], f, indent=2)

print(f"Done. {success_count} successful loads. Output saved to {OUTPUT_FILE}")