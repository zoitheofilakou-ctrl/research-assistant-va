import os
import json
import pdfplumber
from rapidfuzz import process, fuzz

# ROLE: Data Acquisition Group
# PURPOSE: High-fidelity Text Extraction from harvested PDFs with UUID-based naming (paperId).
#          Includes a Truncation-Aware Fuzzy Matching engine to resolve metadata sync issues.

def extract_text_from_pdfs(
    metadata_file="../data/hybrede_metadata_v4.json", 
    pdf_folder="../data/harvested_pdfs", 
    output_folder="../data/v3_full_text"
):
    """
    Orchestrates the conversion of PDF assets into plain text files.
    Ensures filenames in 'harvested_pdfs' (truncated to 100 chars) are correctly
    mapped back to their unique 'paperId' for RAG indexing.
    """
    # Pre-flight Check: Validate directory structure and data sources
    if not os.path.exists(pdf_folder):
        print(f"[!] Critical Error: Source folder '{pdf_folder}' not found.")
        return

    if not os.path.exists(metadata_file):
        print(f"[!] Critical Error: Metadata '{metadata_file}' missing.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Initialized output directory: {output_folder}")

    # Load Metadata 'Source of Truth'
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"[!] Metadata Parse Error: {e}")
        return

    # MAPPING ENGINE: 
    # Create a dictionary where keys are sanitized & truncated titles, values are paperIds.
    # We truncate to 100 chars here to align with the logic used in PDFscraper.py.
    id_map = {}
    for item in metadata:
        title = item.get('title', '')
        paper_id = item.get('paperId', '')
        if title and paper_id:
            # Sanitize title (matching PDFscraper.py logic)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            # TRUNCATION SYNC: Align with 100-char limit to ensure 1:1 match
            truncated_title = safe_title[:100]
            id_map[truncated_title] = paper_id

    # Execution: Scan for PDF assets
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDFs. Initializing extraction pipeline...")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_name)
        # Remove .pdf extension for matching
        base_name = os.path.splitext(pdf_name)[0]

        # TRUNCATION-AWARE FUZZY MATCHING:
        # We use a higher threshold because the id_map is already pre-truncated to match filenames.
        match_result = process.extractOne(base_name, id_map.keys(), scorer=fuzz.ratio)
        
        if match_result and match_result[1] > 90:  # 90% threshold for high-confidence mapping
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
                    
                    # Final I/O: Save extracted text with paperId naming convention
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)
                    
                print(f"[+] Processed: {pdf_name[:50]}... -> {paper_id[:12]}...")
                success_count += 1
            except Exception as e:
                print(f"[!] PDF Extraction Error ({pdf_name}): {e}")
                fail_count += 1
        else:
            print(f"[?] Mapping Failed: No metadata entry aligns with '{pdf_name}'")
            fail_count += 1

    # Final Execution Summary for Group Reporting
    print("=" * 60)
    print(f"CONVERSION SUMMARY:")
    print(f"- Successfully mapped and extracted: {success_count}")
    print(f"- Failures (Unmapped/Corrupted): {fail_count}")
    print(f"- Target storage: {output_folder}")

if __name__ == "__main__":
    # Ensure relative paths work when called from the data_acquisition folder
    extract_text_from_pdfs()