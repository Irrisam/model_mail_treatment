def compute_trust_score(confidence_ip: float, sender_found: bool, client_code_found: bool, client_match_found: bool) -> float:
    """
    Combine :
    - la confiance du modèle ML (confidence_ip)
    - la présence de l'expéditeur en base (sender_found)
    - la présence d'un code client formel (client_code_found)
    - la présence d'un utilisateur correspondant au code client (client_match_found)
    """

    base = confidence_ip
    # TODO Modérer les poids des refs clients et emails inconnus pour soulager les colds calls

    if sender_found:
        email_factor = 1.0          # OK, pas de pénalité
    else:
        email_factor = 0.5          # on divise par 2 si expéditeur inconnu

    if client_code_found:
        code_factor = 1.1           # petit bonus
    else:
        code_factor = 1.0

    if client_match_found:
        match_factor = 1.2          # gros bonus si on trouve un user pour ce code
    else:
        match_factor = 1.0

    score = base * email_factor * code_factor * match_factor

    if score > 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0

    return round(score, 3)
