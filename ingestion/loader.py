import os
from pypdf import PdfReader


def pdf_loader(folder_path):
    """
    Loads the pdf file and save in structured format
    """
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path,filename)
            reader = PdfReader(file_path)
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text()

                if text:
                    documents.append({
                        "text":text,
                        "metadata":{
                            "source": filename,
                            "page":page_number + 1
                        }
                    })
    return documents


if __name__ == "__main__":
    folder_path = "data/raw_docs"
    doc_res = pdf_loader(folder_path)
    