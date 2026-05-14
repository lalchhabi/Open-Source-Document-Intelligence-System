### Import libraries
import os
from pypdf import PdfReader
import fitz
from utils.text_cleaner import text_proprocessing
from langchain_core.documents import Document


def pdf_loader(path):
    """
    Load PDF documents and extract cleaned text from each page.

    This function accepts either:
    - A single PDF file
    - A directory containing multiple PDFs

    For each page of each PDF:
    - Text is extracted using PyMuPDF (fitz)
    - Text is cleaned using the text_preprocessing function
    - Metadata (file path and page number) is attached

    Parameters
    ----------
    path : str
        Path to a single PDF file or a directory containing multiple PDFs.

    Returns
    -------
    list of dict
        A list of dictionaries where each dictionary contains:
        - text : cleaned text content of the page
        - metadata : source file and page number
    """

    docs = []

    # If a single PDF file is provided
    if path.endswith(".pdf"):
        pdf_files = [path]

    # If a folder is provided, collect all PDFs
    else:
        pdf_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pdf")
        ]

    # Process each PDF
    for pdf_path in pdf_files:

        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)

        # Iterate through each page
        for page_num, page in enumerate(doc):

            # Extract raw text
            text = page.get_text("text")

            # Clean the extracted text
            clean_text = text_proprocessing(text)

            # Store cleaned text and metadata
            docs.append(
                Document(
                    page_content=clean_text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1
                    }
                )
            )

    return docs


# Example usage for testing this module independently
if __name__ == "__main__":
    folder_path = "data/raw_docs"
    doc_res = pdf_loader(folder_path)