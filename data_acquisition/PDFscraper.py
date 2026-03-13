import os
import json
import time
import requests
import pdfplumber
from langdetect import detect, DetectorFactory

# Ensure language detection is consistent
DetectorFactory.seed = 0

# ROLE: Data Acquisition Group
# PURPOSE: Automated Full-text PDF Harvesting with Integrated Validation (HTML, Integrity, and Language)
# NOTE: This script is intended to be run from within the 'data_acquisition' folder.

def is_valid_pdf(file_path):
    """
    Validates the PDF file through binary header check, 
    structural integrity check, and language detection.
    """
    # Step 1: Binary Header Check (Detect Fake HTML PDFs)
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            return False, "Not a real PDF (HTML or text error page detected)"
    except Exception as e:
        return False, f"Header check failed: {e}"

    # Step 2: Integrity and Language Check
    try:
        with pdfplumber.open(file_path) as pdf:
            # Check if the file is empty
            if len(pdf.pages) == 0:
                return False, "Corrupted: PDF has no pages"
            
            # Extract text from the first page for language filtering
            first_page_text = pdf.pages[0].extract_text()
            if first_page_text and len(first_page_text.strip()) > 50:
                lang = detect(first_page_text)
                if lang != 'en':
                    return False, f"Non-English paper detected ({lang})"
            else:
                return False, "Text extraction failed (possible image-only PDF)"
                
        return True, "Success"
    except Exception as e:
        return False, f"Integrity check failed: {e}"

def download_paper_pdfs(json_file_path="../data/hybrede_metadata_v3.json", output_folder="../data/harvested_pdfs"):
    """
    Reads metadata from ../data, downloads PDFs, and saves them to ../data/harvested_pdfs.
    """
    # Check if metadata exists before starting
    if not os.path.exists(json_file_path):
        print(f"[!] Error: Metadata file '{json_file_path}' not found.")
        print("[*] Tip: Make sure you are running this script from the 'data_acquisition' folder.")
        return

    # Create output directory inside data folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Created directory: {output_folder}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except Exception as e:
        print(f"[!] Error reading JSON: {e}")
        return

    print(f"[*] Processing {len(papers)} papers from metadata...")
    print("=" * 50)

    downloaded_count = 0
    cleaned_count = 0
    failed_count = 0

    for paper in papers:
        title = paper.get('title', 'Unknown_Title')
        pdf_url = paper.get('openAccessPdf', {}).get('url')

        if not pdf_url:
            continue

        # Clean title for file naming
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        file_path = os.path.join(output_folder, f"{safe_title}.pdf")

        # Skip if file already exists to save time/bandwidth
        if os.path.exists(file_path):
            print(f"[-] Already exists: {safe_title}.pdf")
            downloaded_count += 1
            continue

        try:
            print(f"[*] Downloading: {title}...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(pdf_url, timeout=30, headers=headers)

            if response.status_code == 200:
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                
                # Immediate Validation Step
                valid, message = is_valid_pdf(file_path)
                if valid:
                    print(f"[+] Verified and Saved: {safe_title}.pdf")
                    downloaded_count += 1
                else:
                    print(f"[!] Removing Invalid File: {message}")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    cleaned_count += 1
            else:
                print(f"[!] Download Failed (Status {response.status_code})")
                failed_count += 1

            # Ethical delay to avoid API blocking
            time.sleep(1.2)

        except Exception as e:
            print(f"[!] Error during processing: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            failed_count += 1

    print("=" * 50)
    print(f"FINAL SUMMARY:")
    print(f"- Valid PDFs saved in {output_folder}: {downloaded_count}")
    print(f"- Invalid/Non-English files cleaned: {cleaned_count}")
    print(f"- Technical download failures: {failed_count}")

if __name__ == "__main__":
    # Corrected default paths for the data_acquisition subdirectory
    default_metadata = '../data/hybrede_metadata_v3.json'
    default_output = '../data/harvested_pdfs'
    
    download_paper_pdfs(default_metadata, default_output)