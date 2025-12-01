def compute_trust_score(confidence_ip: float, sender_found: bool) -> float:
    """
    Combine la confiance ML + la présence de l'expéditeur en base.
    Logique simple mais efficace :
        - si utilisateur connu : bonus de confiance
        - si inconnu : pénalité
    """
    if sender_found:
        known_sender_factor = 1.0     # confiance totale si l’utilisateur existe
    else:
        known_sender_factor = 0.6     # pénalité si expéditeur inconnu

    trust = confidence_ip * known_sender_factor
    return round(trust, 3)
