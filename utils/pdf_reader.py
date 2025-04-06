import fitz

def extract_text_from_pdf(path: str) -> None:
    text = ""

    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text