import os
from pypdf import PdfReader
import fitz  

def pdf_loader(path):

    docs = []

    # If single file
    if path.endswith(".pdf"):
        pdf_files = [path]
    else:
        pdf_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pdf")
        ]

    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()

            docs.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,
                    "page": page_num + 1
                }
            })

    return docs



if __name__ == "__main__":
    folder_path = "data/raw_docs"
    doc_res = pdf_loader(folder_path)
    