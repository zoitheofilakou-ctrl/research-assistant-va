import os
import json
import time
import requests
import pdfplumber
import sys
from langdetect import detect, DetectorFactory

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from project_paths import HARVESTED_PDFS_DIR, METADATA_PATH, ensure_dir
from run_manifest import RunManifest

# Set a fixed seed for langdetect to ensure deterministic results across different runs
DetectorFactory.seed = 0

# ROLE: Data Acquisition & Engineering Group
# PURPOSE: Automated harvesting of full-text research papers with a 3-tier validation pipeline:
#          1. Binary Header Validation (Ensures the file is a valid PDF, not an HTML error page)
#          2. Structural Integrity Check (Ensures the PDF is readable and has content)
#          3. Language Filtering (Ensures only English-language papers are retained for the RAG index)

def is_valid_pdf(file_path):
    """
    Performs a multi-stage validation on a downloaded file.
    Returns: (bool, string) -> (Validation status, detailed message)
    """
    # Stage 1: Binary Header Validation
    # Checks if the file starts with the standard PDF signature (%PDF)
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            return False, "Invalid Format: File is likely an HTML redirect or text error page"
    except Exception as e:
        return False, f"Binary check failed: {str(e)}"

    # Stage 2: Structural and Language Validation
    try:
        with pdfplumber.open(file_path) as pdf:
            # Check if the PDF is empty or corrupted during download
            if len(pdf.pages) == 0:
                return False, "Corruption Detected: PDF has no readable pages"
            
            # Extract text from the first page to identify the primary language
            first_page_text = pdf.pages[0].extract_text()
            if first_page_text and len(first_page_text.strip()) > 50:
                lang = detect(first_page_text)
                # Filter out non-English papers to maintain dataset quality for the LLM
                if lang != 'en':
                    return False, f"Language Filter: Non-English content detected ({lang})"
            else:
                return False, "Data Quality: Extraction failed (file might be an image-based scan)"
                
        return True, "Validation Passed"
    except Exception as e:
        return False, f"Integrity check failed: {str(e)}"

def download_paper_pdfs(json_file_path=None, output_folder=None):
    """
    Orchestrates the metadata-driven download process. 
    Implements filename truncation to prevent OS-level path length errors in the RAG module.
    """
    json_file_path = json_file_path or METADATA_PATH
    output_folder = output_folder or HARVESTED_PDFS_DIR
    manifest = RunManifest("pdfscraper")

    # Path Verification: Ensure metadata is accessible from the current execution context
    if not os.path.exists(json_file_path):
        print(f"[!] Critical Error: Metadata source '{json_file_path}' not found.")
        manifest.add_event("missing_input", json_file_path, {})
        manifest.write()
        return

    # Initialization: Create the target data directory if it does not exist
    if not os.path.exists(output_folder):
        ensure_dir(output_folder)
        print(f"[*] Initializing directory: {output_folder}")
        manifest.add_event("created_directory", output_folder, {})

    # Data Loading: Parse the metadata 'Source of Truth'
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except Exception as e:
        print(f"[!] Failed to parse metadata JSON: {e}")
        return

    print(f"[*] Commencing processing of {len(papers)} papers...")
    print("=" * 60)

    downloaded_count = 0
    cleaned_count = 0
    failed_count = 0
    skipped_existing_count = 0

    for paper in papers:
        title = paper.get('title', 'Unknown_Title')
        pdf_url = paper.get('openAccessPdf', {}).get('url')

        # Skip entries that do not provide a direct PDF link
        if not pdf_url:
            continue

        # FILENAME OPTIMIZATION:
        # 1. Sanitize the title by removing non-alphanumeric characters
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # 2. TRUNCATION LOGIC: Limit filename to 100 characters.
        # This prevents 'OSError: File name too long' in the RAG retrieval phase (Windows/macOS limits).
        if len(safe_title) > 100:
            safe_title = safe_title[:100]
        
        file_path = os.path.join(output_folder, f"{safe_title}.pdf")

        # Optimization: Prevent redundant downloads of existing verified assets
        if os.path.exists(file_path):
            print(f"[-] Asset exists: {safe_title}.pdf (Skipping)")
            downloaded_count += 1
            skipped_existing_count += 1
            manifest.add_event("skipped_existing", file_path, {"title": title})
            continue

        try:
            print(f"[*] Downloading: {title[:60]}...")
            headers = {'User-Agent': 'MedicalResearchAssistant/1.0'}
            response = requests.get(pdf_url, timeout=30, headers=headers)

            if response.status_code == 200:
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                
                # Immediate post-download validation
                valid, message = is_valid_pdf(file_path)
                if valid:
                    print(f"[+] Verified and Indexed: {safe_title}.pdf")
                    downloaded_count += 1
                    manifest.add_event("created", file_path, {"title": title})
                else:
                    print(f"[!] Purging Invalid File: {message}")
                    manifest.add_event("rejected", file_path, {"title": title, "reason": message})
                    if os.path.exists(file_path): os.remove(file_path)
                    cleaned_count += 1
            else:
                print(f"[!] Download Failed: HTTP Status {response.status_code}")
                failed_count += 1
                manifest.add_event(
                    "download_failed",
                    file_path,
                    {"title": title, "status_code": response.status_code},
                )

            # Compliance: Delay requests to respect server bandwidth and avoid IP throttling
            time.sleep(1.2)

        except Exception as e:
            print(f"[!] Network/IO Error: {str(e)}")
            manifest.add_event("download_error", file_path, {"title": title, "error": str(e)})
            if os.path.exists(file_path): os.remove(file_path)
            failed_count += 1

    # Final Execution Summary
    print("=" * 60)
    print(f"SUMMARY REPORT:")
    print(f"- Verified PDFs in {output_folder}: {downloaded_count}")
    print(f"- Files rejected by validation pipeline: {cleaned_count}")
    print(f"- Critical download failures: {failed_count}")
    print(f"- Existing PDFs skipped: {skipped_existing_count}")

    manifest.set_summary(
        metadata_path=os.path.relpath(json_file_path, PROJECT_DIR),
        output_folder=os.path.relpath(output_folder, PROJECT_DIR),
        verified_pdfs=downloaded_count,
        rejected_files=cleaned_count,
        failures=failed_count,
        skipped_existing=skipped_existing_count,
    )
    manifest_path = manifest.write()
    print(f"Run manifest written to: {manifest_path}")

if __name__ == "__main__":
    # Standard entry point assuming the script is in 'data_acquisition/'
    download_paper_pdfs()
