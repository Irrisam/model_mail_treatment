from model_pipeline import classify_email

tests = [
    {
        "subject": "Renouvellement de marque",
        "body": "Merci de procéder au renouvellement de la marque avant l’échéance.",
    },
    {
        "subject": "Facture réglée",
        "body": "Ci-joint la confirmation de paiement de l’hébergement serveur.",
    },
    {
        "subject": "Contrefaçon apparente",
        "body": "Nous avons détecté une copie de notre produit sur Amazon.",
    },
    {
        "subject": "Disponibilité pour appel",
        "body": "Peux-tu me confirmer ta dispo demain 15h ?",
    },
]

for t in tests:
    res = classify_email(t["subject"], t["body"])
    print("\nsujet:", t["subject"])
    print("resultats:", res)
