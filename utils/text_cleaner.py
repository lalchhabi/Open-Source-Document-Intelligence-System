import re

def text_proprocessing(text: str) -> str:
    # Remove multiple newlines
    text = re.sub(r"\n+", "\n", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix broken words (common in PDFs)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Remove weird characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return text.strip()
