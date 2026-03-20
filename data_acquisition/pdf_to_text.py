import os
import json
import pdfplumber
import sys
from rapidfuzz import process, fuzz

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from project_paths import FULLTEXT_DIR, HARVESTED_PDFS_DIR, METADATA_PATH, ensure_dir
from run_manifest import RunManifest

# ROLE: Data Acquisition Group
# PURPOSE: High-fidelity Text Extraction from harvested PDFs with UUID-based naming (paperId).
#          Includes a Truncation-Aware Fuzzy Matching engine to resolve metadata sync issues.

def extract_text_from_pdfs(
    metadata_file=None,
    pdf_folder=None,
    output_folder=None
):
    """
    Orchestrates the conversion of PDF assets into plain text files.
    Ensures filenames in 'harvested_pdfs' (truncated to 100 chars) are correctly
    mapped back to their unique 'paperId' for RAG indexing.
    """
    metadata_file = metadata_file or METADATA_PATH
    pdf_folder = pdf_folder or HARVESTED_PDFS_DIR
    output_folder = output_folder or FULLTEXT_DIR
    manifest = RunManifest("pdf_to_text")

    # Pre-flight Check: Validate directory structure and data sources
    if not os.path.exists(pdf_folder):
        print(f"[!] Critical Error: Source folder '{pdf_folder}' not found.")
        manifest.add_event("missing_input", pdf_folder, {})
        manifest.write()
        return

    if not os.path.exists(metadata_file):
        print(f"[!] Critical Error: Metadata '{metadata_file}' missing.")
        manifest.add_event("missing_input", metadata_file, {})
        manifest.write()
        return

    if not os.path.exists(output_folder):
        ensure_dir(output_folder)
        print(f"[*] Initialized output directory: {output_folder}")
        manifest.add_event("created_directory", output_folder, {})

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
    updated_count = 0
    skipped_existing_count = 0
    unmatched_count = 0

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

                    existing_text = None
                    if os.path.exists(output_path):
                        with open(output_path, 'r', encoding='utf-8') as f:
                            existing_text = f.read()

                    if existing_text == full_text:
                        print(f"[-] Text unchanged: {paper_id[:12]}...")
                        skipped_existing_count += 1
                        manifest.add_event("skipped_existing", output_path, {"source_pdf": pdf_name})
                        continue

                    # Final I/O: Save extracted text with paperId naming convention
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_text)

                if existing_text is None:
                    print(f"[+] Created: {pdf_name[:50]}... -> {paper_id[:12]}...")
                    success_count += 1
                    manifest.add_event("created", output_path, {"source_pdf": pdf_name})
                else:
                    print(f"[~] Updated: {pdf_name[:50]}... -> {paper_id[:12]}...")
                    updated_count += 1
                    manifest.add_event("updated", output_path, {"source_pdf": pdf_name})
            except Exception as e:
                print(f"[!] PDF Extraction Error ({pdf_name}): {e}")
                fail_count += 1
                manifest.add_event("extract_error", pdf_path, {"error": str(e)})
        else:
            print(f"[?] Mapping Failed: No metadata entry aligns with '{pdf_name}'")
            fail_count += 1
            unmatched_count += 1
            manifest.add_event("unmatched_pdf", pdf_path, {})

    # Final Execution Summary for Group Reporting
    print("=" * 60)
    print(f"CONVERSION SUMMARY:")
    print(f"- Newly created text files: {success_count}")
    print(f"- Updated text files: {updated_count}")
    print(f"- Unchanged existing text files: {skipped_existing_count}")
    print(f"- Failures (Unmapped/Corrupted): {fail_count}")
    print(f"- Unmatched PDFs: {unmatched_count}")
    print(f"- Target storage: {output_folder}")

    manifest.set_summary(
        metadata_path=os.path.relpath(metadata_file, PROJECT_DIR),
        pdf_folder=os.path.relpath(pdf_folder, PROJECT_DIR),
        output_folder=os.path.relpath(output_folder, PROJECT_DIR),
        created=success_count,
        updated=updated_count,
        skipped_existing=skipped_existing_count,
        failures=fail_count,
        unmatched=unmatched_count,
    )
    manifest_path = manifest.write()
    print(f"Run manifest written to: {manifest_path}")

if __name__ == "__main__":
    # Ensure relative paths work when called from the data_acquisition folder
    extract_text_from_pdfs()
