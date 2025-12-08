import re

danger_tokens = [
    "brevet", "brevetabilité", "revendication", "revendications", "contrefaçon",
    "antériorité", "antériorités", "recherche d'antériorités", "similarité des signes",
    "dépôt", "dépôts", "propriété intellectuelle", "invention", "extension",
    "annuité", "annuités", "priorité", "liberté d’exploitation", "PCT", "EPC",
    "INPI", "OEB", "EPO", "WIPO", "OMPI", "USPTO", "mandataire", "agent brevets"
]


def sanitize(text):
    for tok in danger_tokens:
        text = text.replace(tok, "[TERME_PI]")
    return text


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = sanitize(text)
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
