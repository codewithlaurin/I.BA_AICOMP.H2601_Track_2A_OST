import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import Request, urlopen

DATASET = "OSTswiss/MNLIoverSwissVotingBooklets"
API = "https://datasets-server.huggingface.co"


def fetch(url):
    """Retry transient network failures; never execute downloaded content."""
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "ApertusBookletDownloader/1.0"})
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def api(endpoint, **params):
    return json.loads(fetch(f"{API}/{endpoint}?{urlencode(params)}"))


def collect_urls(dataset):
    """Paginate every config/split, retaining metadata for each unique URL."""
    splits = api("splits", dataset=dataset)
    if splits.get("pending") or splits.get("failed"):
        raise RuntimeError("Dataset API has pending or failed splits; try again later.")
    if not splits.get("splits"):
        raise RuntimeError("No dataset splits found.")
    urls = {}
    for split in splits["splits"]:
        config, name = split["config"], split["split"]
        offset = 0
        while True:
            page = api("rows", dataset=dataset, config=config,
                       split=name, offset=offset, length=100)
            rows = page["rows"]
            total = page["num_rows_total"]
            if not rows and offset < total:
                raise RuntimeError(f"Incomplete API response for {config}/{name}.")
            for item in rows:
                row = item["row"]
                url = row["booklet_url"]
                if "booklet_url" in item.get("truncated_cells", []):
                    raise ValueError("API returned a truncated booklet URL.")
                if not isinstance(url, str) or not url.strip():
                    raise ValueError(f"Missing booklet_url in {config}/{name} row {item['row_idx']}")
                url = url.strip()
                parsed = urlsplit(url)
                if parsed.scheme not in {"https", "http"} or not parsed.hostname:
                    raise ValueError(f"Invalid booklet URL: {url}")
                entry = urls.setdefault(url, {"references": [], "languages": set()})
                entry["references"].append({"config": config, "split": name,
                                            "row_index": item["row_idx"]})
                if row.get("reference_language"):
                    entry["languages"].add(row["reference_language"])
            offset += len(rows)
            print(f"Scanned {config}/{name}: {offset}/{total}", flush=True)
            if offset >= total:
                break
    return urls


def filename(url):
    stem = Path(unquote(urlsplit(url).path)).stem
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-")[:100] or "booklet"
    # URL hash prevents different URLs with identical filenames from colliding.
    return f"{stem}__{hashlib.sha256(url.encode()).hexdigest()}.pdf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--list-only", action="store_true",
                        help="Write the URL manifest without downloading PDFs.")
    args = parser.parse_args()
    urls = collect_urls(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    pdf_dir = args.output / "pdf"
    pdf_dir.mkdir(exist_ok=True)
    manifest = args.output / "booklet_manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    old = {entry["url"]: entry for entry in previous.get("files", [])}
    report = {"dataset": args.dataset, "unique_urls": len(urls), "files": []}
    failures = 0
    for index, (url, metadata) in enumerate(sorted(urls.items()), 1):
        target = pdf_dir / filename(url)
        entry = {"url": url, "path": str(target.relative_to(args.output)),
                 "languages": sorted(metadata["languages"]),
                 "references": metadata["references"], "status": "pending"}
        try:
            if not args.list_only:
                data = target.read_bytes() if target.exists() else None
                cached = (data is not None and b"%PDF-" in data[:1024]
                          and hashlib.sha256(data).hexdigest() == old.get(url, {}).get("sha256"))
                if not cached:
                    data = fetch(url)
                    if b"%PDF-" not in data[:1024]:
                        raise ValueError("Response does not have a PDF header")
                    temporary = target.with_suffix(".pdf.part")
                    temporary.write_bytes(data)
                    temporary.replace(target)
                entry.update(status="cached" if cached else "downloaded",
                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        except (OSError, ValueError) as exc:
            failures += 1
            entry.update(status="error", error=str(exc))
        report["files"].append(entry)
        # Checkpoint after each file; complete files are reusable on the next run.
        pending_manifest = manifest.with_suffix(".json.part")
        pending_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        pending_manifest.replace(manifest)
        print(f"[{index}/{len(urls)}] {entry['status']}: {target.name}", flush=True)
    print(f"{len(urls)} unique URLs; {failures} failures. Manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
