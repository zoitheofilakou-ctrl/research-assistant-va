import pdfplumber
import os
import json
import re


# ROLE: Data Acquisition Group
# PURPOSE: Enhanced extraction with fuzzy title matching to reduce "Skipping" errors.

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_METADATA_FILE = os.path.join(BASE_DIR, "data", "hybrede_metadata_v3.json")
DEFAULT_PDF_DIR = os.path.join(BASE_DIR, "data", "harvested_pdfs")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "v3_full_text")


def clean_string(s):
    """
    Removes all non-alphanumeric characters and converts to lowercase.
    This helps match "HR1 Robot: Assistant" with "HR1 Robot Assistant".
    """
    return re.sub(r'[^a-zA-Z0-9]', '', str(s)).lower()


def extract_text_from_pdfs(
    metadata_file=DEFAULT_METADATA_FILE,
    pdf_folder=DEFAULT_PDF_DIR,
    output_folder=DEFAULT_OUTPUT_DIR,
):
    if not os.path.exists(metadata_file):
        print(f"[!] Error: Metadata file {metadata_file} not found.")
        return

    # Load metadata and build a "Cleaned Title -> paperId" map.
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # We use the clean_string function to create a robust lookup table.
    title_to_id = {
        clean_string(item.get('title', '')): item.get('paperId')
        for item in metadata if item.get('paperId')
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDFs. Attempting robust matching...")

    success_count = 0
    fail_count = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)

        # Clean the filename (without extension) for a more flexible match.
        filename_cleaned = clean_string(os.path.splitext(pdf_file)[0])
        paper_id = title_to_id.get(filename_cleaned)

        if not paper_id:
            print(f"[!] Skipping: Could not match '{pdf_file}' to any ID in metadata.")
            fail_count += 1
            continue

        text_filename = f"{paper_id}.txt"
        text_path = os.path.join(output_folder, text_filename)

        try:
            full_text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        full_text.append(content)

            if full_text:
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(full_text))
                print(f"[+] Success: {pdf_file} -> {text_filename}")
                success_count += 1
            else:
                print(f"[!] Warning: No text found in {pdf_file}")

        except Exception as e:
            # Handles "No /Root object" or corrupted PDF errors.
            print(f"[!] Error processing {pdf_file}: {e}")

    print(f"\n[*] Processing Complete: {success_count} succeeded, {fail_count} skipped.")


if __name__ == "__main__":
    extract_text_from_pdfs()
