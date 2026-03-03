import os
import json
import time
import requests

# ROLE: Data Acquisition Group
# PURPOSE: Automated Full-text PDF Harvesting for RAG System

def download_paper_pdfs(json_file_path, output_folder="harvested_pdfs"):
    """
    Reads the metadata JSON and downloads available PDFs.
    """
    # 1. Create a directory to save the PDFs
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Created directory: {output_folder}")

    # 2. Load the metadata
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print(f"[!] Error: {json_file_path} not found.")
        return

    print(f"[*] Total papers in metadata: {len(papers)}")
    print("=" * 50)

    downloaded_count = 0
    failed_count = 0

    for paper in papers:
        title = paper.get('title', 'Unknown_Title')
        # Semantic scholar often puts the direct PDF link in 'openAccessPdf' -> 'url'
        pdf_url = paper.get('openAccessPdf', {}).get('url')

        if not pdf_url:
            print(f"[ ] Skipping: '{title}' (No direct PDF link found)")
            continue

        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        file_path = os.path.join(output_folder, f"{safe_title}.pdf")

        try:
            print(f"[*] Downloading: {title}...")
            # Set a timeout and headers to act like a real browser
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(pdf_url, timeout=30, headers=headers)

            if response.status_code == 200:
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                print(f"[+] Success! Saved as: {safe_title}.pdf")
                downloaded_count += 1
            else:
                print(f"[!] Failed (Status {response.status_code}): {title}")
                failed_count += 1

            # brief pause between downloads
            time.sleep(1.5)

        except Exception as e:
            print(f"[!] Error downloading {title}: {e}")
            failed_count += 1

    print("=" * 50)
    print(f"SUMMARY: {downloaded_count} PDFs downloaded, {failed_count} failed.")

if __name__ == "__main__":
    # Ensure this matches metadata filename
    metadata_file = 'hybrede_metadata_v3.json'
    download_paper_pdfs(metadata_file)