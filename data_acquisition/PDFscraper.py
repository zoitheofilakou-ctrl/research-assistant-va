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

def download_paper_pdfs(json_file_path, output_folder="harvested_pdfs"):
    """
    Reads metadata, downloads PDFs, and automatically removes invalid files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Created directory: {output_folder}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: {json_file_path} not found.")
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
    print(f"- Valid PDFs saved: {downloaded_count}")
    print(f"- Invalid/Non-English files cleaned: {cleaned_count}")
    print(f"- Technical download failures: {failed_count}")

if __name__ == "__main__":
    # Ensure this file matches your current metadata version
    metadata_file = 'hybrede_metadata_v3.json'
    download_paper_pdfs(metadata_file)