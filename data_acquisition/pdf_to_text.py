import os
import json
import pdfplumber
from rapidfuzz import process, fuzz

# ROLE: Data Acquisition Group
# PURPOSE: Convert validated PDFs to machine-readable text with paperId naming
# NOTE: This script must be run from within the 'data_acquisition' folder.

def extract_text_from_pdfs(
    metadata_file="../data/hybrede_metadata_v3.json", 
    pdf_folder="../data/harvested_pdfs", 
    output_folder="../data/v3_full_text"
):
    """
    Reads PDFs from data/harvested_pdfs and extracts text to data/v3_full_text 
    named by their unique paperId from metadata.
    """
    # Step 0: Robustness Check - ensure folders exist as per team feedback
    if not os.path.exists(pdf_folder):
        print(f"[!] Error: PDF folder '{pdf_folder}' not found. Please run PDFscraper.py first.")
        return

    if not os.path.exists(metadata_file):
        print(f"[!] Error: Metadata file '{metadata_file}' not found.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Created output directory: {output_folder}")

    # Load metadata for ID mapping
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"[!] Error reading metadata: {e}")
        return

    # Create a mapping of safe title to paperId
    id_map = {}
    for item in metadata:
        title = item.get('title', '')
        paper_id = item.get('paperId', '')
        if title and paper_id:
            # Recreate the safe filename used in PDFscraper.py
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            id_map[safe_title] = paper_id

    # Process PDFs
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDFs in {pdf_folder}. Starting extraction...")
    print("=" * 50)

    success_count = 0
    fail_count = 0

    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_name)
        base_name = os.path.splitext(pdf_name)[0]

        # Use fuzzy matching to find the correct paperId
        match_result = process.extractOne(base_name, id_map.keys(), scorer=fuzz.token_set_ratio)
        
        if match_result and match_result[1] > 80:  # 80% similarity threshold
            best_match_title = match_result[0]
            paper_id = id_map[best_match_title]
            output_path = os.path.join(output_folder, f"{paper_id}.txt")
            
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text_content = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                    
                    full_text = "\n".join(text_content)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)
                    
                print(f"[+] Extracted: {pdf_name} -> {paper_id}.txt")
                success_count += 1
            except Exception as e:
                print(f"[!] Failed to process {pdf_name}: {e}")
                fail_count += 1
        else:
            print(f"[?] No metadata match found for: {pdf_name}")
            fail_count += 1

    print("=" * 50)
    print(f"EXTRACTION SUMMARY:")
    print(f"- Files successfully processed: {success_count}")
    print(f"- Files failed/skipped: {fail_count}")
    print(f"- Output location: {output_folder}")

if __name__ == "__main__":
    # Default paths for execution inside 'data_acquisition/'
    extract_text_from_pdfs()