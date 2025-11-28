import re

def preprocess(text: str) -> str:
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s]{7,}", " ", text)

    # Remove signatures (heuristics)
    text = re.sub(r"Bien cordialement[\s\S]*", "", text, flags=re.I)

    # Remove long tables/numbers (taxes, amounts)
    text = re.sub(r"\b\d+[.,]\d+\b", " ", text) 

    # Remove email history metadata (Cc:, Envoyé, etc.)
    text = re.sub(r"^(Cc:|De:|Envoyé:|Objet:).*$", "", text, flags=re.M)

    # Remove empty repetitive lines
    text = re.sub(r"\n\s*\n", "\n", text)

    return text.strip()
