import pdfplumber
import os

# ROLE: Data Acquisition Group
# PURPOSE: Converting harvested PDFs into plain text for RAG indexing

def extract_text_from_pdfs(pdf_folder="harvested_pdfs", output_folder="processed_text"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[*] Created directory: {output_folder}")

    # List all PDF files in the source folder
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDFs to process.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        # change extension to .txt
        text_filename = os.path.splitext(pdf_file)[0] + ".txt"
        text_path = os.path.join(output_folder, text_filename)

        try:
            print(f"[*] Extracting text from: {pdf_file}...")
            full_text = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        full_text.append(content)
            
            # Save to a .txt file
            if full_text:
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(full_text))
                print(f"[+] Success: Saved to {text_filename}")
            else:
                print(f"[!] Warning: No text found in {pdf_file}")

        except Exception as e:
            print(f"[!] Error processing {pdf_file}: {e}")

if __name__ == "__main__":
    extract_text_from_pdfs()