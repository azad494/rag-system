#!/usr/bin/env python3
import os
import sys
import re
import shutil
import zipfile
import datetime
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
    from dotenv import load_dotenv
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e.name}. Run: pip install requests beautifulsoup4 python-dotenv kaggle")
    sys.exit(1)

# Initialize staging workspace directory
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Shared browser engine context signatures to bypass common scraping blocks
USER_AGENT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}

# --- Dynamic Token Memory Mapping Layer ---
load_dotenv()
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

if KAGGLE_API_TOKEN:
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    if not KAGGLE_KEY: os.environ["KAGGLE_KEY"] = KAGGLE_API_TOKEN
if KAGGLE_USERNAME: os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
if KAGGLE_KEY: os.environ["KAGGLE_KEY"] = KAGGLE_KEY

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    KAGGLE_AVAILABLE = True
except Exception as err:
    KAGGLE_AVAILABLE = False
    KAGGLE_INIT_ERROR = str(err)

def parse_kaggle_slug(url: str) -> tuple[str, str]:
    """Extracts identifiers from standard web browser links or straightforward strings."""
    url = url.strip()
    dataset_match = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", url, re.IGNORECASE)
    if dataset_match: return "dataset", dataset_match.group(1)
    
    competition_match = re.search(r"kaggle\.com/(?:competitions|c)/([^/?#]+)", url, re.IGNORECASE)
    if competition_match: return "competition", competition_match.group(1)
    
    if "/" in url:
        parts = [p for p in url.split("/") if p]
        if len(parts) >= 2: return "dataset", f"{parts[-2]}/{parts[-1]}"
    return "dataset", url

def option_1_kaggle():
    """Option 1: Ingest Data from a Kaggle Link using native python package execution imports."""
    print("\n--- [1] KAGGLE AUTOMATED DOWNLOAD ROUTER ---")
    if not KAGGLE_AVAILABLE:
        print(f"❌ Handshake Denied: {KAGGLE_INIT_ERROR}\nCheck your .env setup parameters.")
        return
    
    url = input("📋 Paste Kaggle URL or ID Slug: ").strip()
    if not url: return
    
    dtype, slug = parse_kaggle_slug(url)
    try:
        if dtype == "dataset":
            print(f"📥 Fetching data collection files for dataset slug: '{slug}'...")
            api.dataset_download_files(slug, path=str(RAW_DIR), unzip=True)
            print("✅ Successfully unzipped and indexed assets directly into 'data/raw/'!")
        else:
            print(f"📥 Fetching metrics for competition slug: '{slug}'...")
            zip_filepath = RAW_DIR / f"{slug}.zip"
            api.competition_download_files(slug, path=str(RAW_DIR))
            if zip_filepath.exists():
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    zip_ref.extractall(RAW_DIR)
                zip_filepath.unlink()
            print("✅ Competition layers synchronized successfully!")
    except Exception as e:
        print(f"❌ Sourcing aborted: {e}\n💡 If using a competition, remember to click 'Accept Terms' in a browser first.")

def option_2_html():
    """Option 2: Scrape web medical records and track fallback text layouts carefully."""
    print("\n--- [2] WEB HTML EXTRACTOR ---")
    url = input("📋 Paste target Web Source URL: ").strip()
    if not url: return
    
    clean_slug = re.sub(r'[^a-zA-Z0-9]', '_', url.split("//")[-1])[:30]
    out_filename = f"scraped_{clean_slug}.txt"
    out_filepath = RAW_DIR / out_filename
    
    if out_filepath.exists():
        print("⏭️ File tracking triggered — item is already sitting inside your pantry folder.")
        return

    try:
        print(f"🌐 Sourcing metadata layers from {url}...")
        res = requests.get(url, headers=USER_AGENT_HEADERS, timeout=15)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Strict Pylance compilation fix: verify soup title object isn't None
        title_str = soup.title.string.strip() if (soup.title and soup.title.string) else f"Web_Record_{datetime.date.today()}"
        
        for script_or_style in soup(["script", "style", "meta", "noscript", "header", "footer"]):
            script_or_style.decompose()
            
        clean_text = "\n".join(chunk.strip() for chunk in soup.get_text().splitlines() if chunk.strip())
        payload = f"SOURCE URL: {url}\nTITLE: {title_str}\nDATE: {datetime.datetime.now()}\n{'='*40}\n\n{clean_text}"
        
        out_filepath.write_text(payload, encoding='utf-8')
        print(f"✅ Body text and layout details harvested cleanly to: {out_filepath}")
    except Exception as e:
        print(f"❌ Parsing failure: {e}")

def option_3_pdf():
    """Option 3: Stream and check bytes for massive clinical guideline PDFs securely."""
    print("\n--- [3] MEDICAL PDF STREAMING SYSTEM ---")
    url = input("📋 Paste direct Document PDF URL: ").strip()
    if not url: return
    
    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith(".pdf"): filename = "clinical_guideline_ingest.pdf"
    dest_path = RAW_DIR / filename
    
    if dest_path.exists():
        print("⏭️ File tracking triggered — item is already sitting inside your pantry folder.")
        return

    try:
        print("📥 Establishing streaming validation pipe...")
        res = requests.get(url, headers=USER_AGENT_HEADERS, stream=True, timeout=30)
        
        if res.status_code != 200:
            print(f"❌ Dropped by destination firewall rules. Server Error Code: {res.status_code}")
            return
            
        total_bytes = 0
        with open(dest_path, "wb") as pdf_file:
            for chunk in res.iter_content(chunk_size=4096):
                if chunk:
                    pdf_file.write(chunk)
                    total_bytes += len(chunk)
                    
        if total_bytes == 0:
            if dest_path.exists(): dest_path.unlink()
            print("⚠️ Trace cleared: network data stream dropped zero payload bytes.")
        else:
            print(f"✅ Pipeline finalized! Cached: '{filename}' ({total_bytes/1024:.1f} KB)")
    except Exception as e:
        if dest_path.exists(): dest_path.unlink()
        print(f"❌ Socket error during physical file stream: {e}")

def option_4_audit():
    """Option 4: Audit Local Pantry Inventory Contents."""
    print("\n--- [4] PANTRY INVENTORY AUDIT ---")
    files = list(RAW_DIR.glob("*"))
    if not files:
        print("📭 Staging folder is completely empty.")
        return
    print(f"+{'-'*50}+{'-'*12}+{'-'*20}+")
    print(f"| {'FILENAME'.ljust(48)} | {'SIZE'.ljust(10)} | {'EXTENSION'.ljust(18)} |")
    print(f"+{'-'*50}+{'-'*12}+{'-'*20}+")
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            size_fmt = f"{size/1024:.1f} KB" if size < 1048576 else f"{size/1048576:.2f} MB"
            print(f"| {f.name[:45].ljust(48)} | {size_fmt.rjust(10)} | {f.suffix.upper().replace('.','').ljust(18)} |")
    print(f"+{'-'*50}+{'-'*12}+{'-'*20}+")

def main():
    while True:
        print("\n📂 PHASE 1 STAGING WORKSPACE: SIMPLE DATA SOURCE HUB")
        print("="*55)
        print("1. Ingest Data from a Kaggle Link")
        print("2. Scrape and Extract Metadata from an HTML/Web URL")
        print("3. Stream and Validate a Medical PDF URL")
        print("4. Audit Local Pantry Inventory Contents")
        print("5. Exit Ingestion Utility")
        print("="*55)
        choice = input("Select operation source menu [1-5]: ").strip()
        if choice == "1": option_1_kaggle()
        elif choice == "2": option_2_html()
        elif choice == "3": option_3_pdf()
        elif choice == "4": option_4_audit()
        elif choice == "5":
            print("\n🔌 Shutting down automation layers. Goodbye!"); break
        else:
            print("❌ Invalid menu choice index parameter.")

if __name__ == "__main__":
    main()