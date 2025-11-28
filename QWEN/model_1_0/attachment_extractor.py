import os
import pdfplumber
import docx

ATTACHMENTS_ROOT = "pdfs"


def extract_text_from_pdf(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except:
        pass
    return text


def extract_text_from_docx(path):
    try:
        d = docx.Document(path)
        return "\n".join([p.text for p in d.paragraphs])
    except:
        return ""


def extract_attachments_text(email_id: int):
    folder = os.path.join(ATTACHMENTS_ROOT, str(email_id))
    full_text = ""

    if not os.path.isdir(folder):
        return full_text

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(".pdf"):
            full_text += extract_text_from_pdf(path) + "\n"
        elif file.lower().endswith(".docx"):
            full_text += extract_text_from_docx(path) + "\n"

    return full_text.strip()
