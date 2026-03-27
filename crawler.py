import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime
import sys
from typing import Any, Set
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Path to NDJSON file of URLs")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

# CONFIG
# Set your start URL and base domain here.
# The script will crawl from the start URL to each leaf on the website's tree.
# ex: START_URL = "https://www.google.com", BASE_DOMAIN = "google.com"
START_URL = ""
BASE_DOMAIN = ""

# CHECKPOINTING AND RESUME
RESUME = True
CHECKPOINT_DIR = "checkpoints"
DISCOVERY_STATE_PATH = os.path.join(CHECKPOINT_DIR, "discovery_state.json")
DISCOVERED_LINKS_PATH = os.path.join(CHECKPOINT_DIR, "discovered_links.json")
RUN_STATE_PATH = os.path.join(CHECKPOINT_DIR, "run_state.json")
RUN_NDJSON_PATH = os.path.join("run", "url_map.ndjson")

# PERFORMANCE AND PACING
REQUEST_TIMEOUT = 15
PROGRESS_EVERY = 10 

def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("run", exist_ok=True)

def atomic_write_json(path: str, data: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_json(path: str, default: Any):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_discovery_snapshot(to_visit: Set[str], visited: Set[str], all_links: Set[str]):
    state = {
        "to_visit": list(to_visit),
        "visited": list(visited),
        "all_links": list(all_links),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    atomic_write_json(DISCOVERY_STATE_PATH, state)
    atomic_write_json(DISCOVERED_LINKS_PATH, sorted(all_links))

def load_discovery_state(start_url: str):
    if RESUME and os.path.exists(DISCOVERY_STATE_PATH):
        state = load_json(DISCOVERY_STATE_PATH, {})
        if state and "to_visit" in state and "visited" in state and "all_links" in state:
            to_visit = set(state["to_visit"])
            visited = set(state["visited"])
            all_links = set(state["all_links"])
            if not to_visit and start_url not in visited:
                to_visit.add(start_url)
            log(f"Resuming discovery: {len(visited)} visited, {len(all_links)} found, {len(to_visit)} queued")
            return to_visit, visited, all_links
    return {start_url}, set(), set()

def get_all_links(start_url, base_domain, branch_only=True):
    url_prefix = start_url.rstrip("/") if branch_only else None

    to_visit, visited, all_links = load_discovery_state(start_url)

    try:
        while to_visit:
            url = to_visit.pop()
            if url in visited:
                continue
            visited.add(url)
            try:
                r = requests.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
            except requests.exceptions.RequestException:
                if len(visited) % PROGRESS_EVERY == 0:
                    log(f"Link discovery: {len(visited)} pages visited, {len(all_links)} total unique found so far (with errors)")
                    save_discovery_snapshot(to_visit, visited, all_links)
                continue

            soup = BeautifulSoup(r.content, 'html5lib')
            for a_tag in soup.find_all('a', href=True):
                full_url = urljoin(url, a_tag['href'])
                parsed = urlparse(full_url)

                # Only allow internal links
                if base_domain not in parsed.netloc:
                    continue

                # Skip non-HTML assets
                if full_url.endswith((".pdf", ".jpg", ".png", ".gif")):
                    continue

                # Skip anchors
                if "#" in full_url:
                    continue

                # Branch-only filter
                if branch_only and not full_url.startswith(url_prefix):
                    continue

                if full_url not in visited and full_url not in all_links:
                    all_links.add(full_url)
                    to_visit.add(full_url)

            if len(visited) % PROGRESS_EVERY == 0:
                log(f"Link discovery: {len(visited)} pages visited, {len(all_links)} total unique found so far")
                save_discovery_snapshot(to_visit, visited, all_links)

            time.sleep(0.2)

    except KeyboardInterrupt:
        log("Discovery interrupted. Checkpointing state...")
        save_discovery_snapshot(to_visit, visited, all_links)
        log(f"Checkpoint saved. Visited={len(visited)} Found={len(all_links)} Queue={len(to_visit)}")
        raise

    save_discovery_snapshot(to_visit, visited, all_links)
    return sorted(all_links)

def load_run_state():
    state = load_json(RUN_STATE_PATH, {})
    tagged_urls = set(state.get("tagged_urls", []))
    return tagged_urls

def save_run_state(tagged_urls: Set[str]):
    state = {
        "tagged_urls": list(tagged_urls),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    atomic_write_json(RUN_STATE_PATH, state)

def finalize_run_outputs():
    # Aggregate NDJSON into the deliverables
    rows = []
    if os.path.exists(RUN_NDJSON_PATH):
        with open(RUN_NDJSON_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    atomic_write_json(os.path.join("run", "url_map.json"), rows)
    
    # Convert url_map.ndjson to urls_list.py
    INPUT_MAP_FILE = "run/url_map.ndjson"
    OUTPUT_MAP_FILE = "urls_list.py"

    urls = []

    with open(INPUT_MAP_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                url = obj.get("url")
                urls.append(url)
            except json.JSONDecodeError:
                continue

    with open(OUTPUT_MAP_FILE, "w", encoding="utf-8") as out:
        out.write("urls = [\n")
        for url in urls:
            out.write(f"    \"{url}\",\n")
        out.write("]\n")  # Always close the list

    print(f"[+] Saved {len(urls)} URLs List to {OUTPUT_MAP_FILE}")

if __name__ == "__main__":
    ensure_dirs()
    log("Starting URL collection")

    if args.input:
        if not os.path.exists(args.input):
            log(f"Input file not found: {args.input}")
            sys.exit(1)
        try:
            with open(args.input, "r", encoding="utf-8-sig") as f:
                links = [json.loads(line)["url"] for line in f if line.strip()]
        except (json.JSONDecodeError, KeyError) as e:
            log(f"Failed to parse {args.input}: {e}")
            sys.exit(1)
        if not links:
            log(f"No valid URLs found in {args.input}")
            sys.exit(1)
        log(f"Loaded {len(links)} URLs from {args.input}")
    else:
        try:
            links = get_all_links(START_URL, BASE_DOMAIN)
        except KeyboardInterrupt:
            log("Interrupted during discovery. Resume is enabled; next run will pick up from the checkpoint.")
            sys.exit(0)
        log(f"Found {len(links)} SRD pages in total (also saved to {DISCOVERED_LINKS_PATH})")

    tagged_urls = load_run_state() if RESUME else set()

    # Open NDJSON in append mode
    os.makedirs("run", exist_ok=True)
    nd_mode = "a" if os.path.exists(RUN_NDJSON_PATH) else "w"
    with open(RUN_NDJSON_PATH, nd_mode, encoding="utf-8") as ndout:
        try:
            for idx, url in enumerate(links, start=1):
                if url in tagged_urls:
                    continue
                title_guess = url.split("/")[-1].replace("-", " ")
                record = {"url": url}
                ndout.write(json.dumps(record, ensure_ascii=False) + "\n")
                tagged_urls.add(url)

                if len(tagged_urls) % PROGRESS_EVERY == 0:
                    ndout.flush()
                    os.fsync(ndout.fileno())
                    save_run_state(tagged_urls)
                    log(f"Tagged {len(tagged_urls)} / {len(links)} links so far")
        except KeyboardInterrupt:
            ndout.flush()
            try:
                os.fsync(ndout.fileno())
            except Exception:
                pass
            save_run_state(tagged_urls)
            log(f"Run interrupted — progress saved: {len(tagged_urls)} tagged. Resume will continue where you left off.")
            sys.exit(0)

    save_run_state(tagged_urls)
    finalize_run_outputs()