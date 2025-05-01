import fitz 
import os 
import pandas as pd


def embed_white_text(input_pdf_path, output_pdf_path, text, x=100, y=700, font_size=12):
    """Embed white text into a PDF using fitz (PyMuPDF)."""
    doc = fitz.open(input_pdf_path)
    
    for page in doc:
        page.insert_text(
            (x, y),               
            text,
            fontname="helv",      
            fontsize=font_size,
            color=(1, 1, 1),
            overlay=True
        )
    print(input_pdf_path)
    if not doc.can_save_incrementally:
        print("Warning: The PDF cannot be saved incrementally. Saving as a new file.")
        output_pdf_path = os.path.splitext(output_pdf_path)[0] + "_modified.pdf"
        doc.ez_save(output_pdf_path)
        doc.close()
        return output_pdf_path
    else:
        doc.save(output_pdf_path, incremental=True, encryption=0)
        doc.close()
        return output_pdf_path

def get_all_text(input_pdf_path):
    """Get all text from a PDF using fitz (PyMuPDF)."""
    doc = fitz.open(input_pdf_path)
    res = ""
    for page in doc:
        text = page.get_text()
        res += text

    doc.close()
    return res


def create_file_label_csv(root_dir, output_csv_path, poison=False):
    """Create a CSV with file paths, labels based on directory names."""
    entries = []

    for root_dirpath, root_dirnames, _ in os.walk(root_dir):
        for c, dirname in enumerate(root_dirnames):
            label = c
            label_dir = os.path.join(root_dirpath, dirname)
            for dirpath, dirnames, filenames in os.walk(label_dir):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if poison:
                        out_path = embed_white_text(
                            input_pdf_path=file_path,
                            output_pdf_path=file_path,
                            text="cf",
                            x=150,
                            y=600,
                            font_size=12
                        )
                    else:
                        out_path = file_path

                    entries.append(dict(out_path=out_path, label=label))

    # Write to CSV
    df = pd.DataFrame(entries)
    df.to_csv(output_csv_path, index=False, escapechar='\\')

    print(f"CSV created with {len(entries)} entries at: {output_csv_path}")

# Example usage
create_file_label_csv(
    root_dir="/content/Instruction_Backdoor_Attack/data",
    output_csv_path="output.csv"
)
