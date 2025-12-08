IP Email Classifier — Offline PI Classification

Ce projet permet de classifier automatiquement des emails liés à la Propriété Industrielle (PI) en se basant sur :
le sujet
le corps
le contenu des pièces jointes (PDF → extraction texte)

L’objectif est de filtrer les emails vers 3 statuts :

Statut	Description
ip - Email clairement lié à la PI → classification en catégories (ex: Marque, Contrats...)
review - Cas limite → nécessite validation humaine
not_ip - Hors PI → ignoré

Installation & Lancement

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn model_service:app --reload --port 8888


API disponible sur :
http://localhost:8888/docs (Swagger)

Endpoints API
Route	Méthode	Description
/classify	POST	Classifie un seul email
/batch	POST	Traite tous les fichiers du dossier mails_json/
/logs	GET	Retourne toutes les décisions du système
/health	GET	Renvoi status serveur

Exemple d’appel API
Classifier un email :
curl -X POST "http://localhost:8888/classify" \
-H "Content-Type: application/json" \
-d '{
  "email_id": 42,
  "subject": "Extension Madrid - inclure Canada",
  "body": "Pouvez-vous ajouter CA dans notre désignation ?"
}'


Réponse possible :

{
  "labels": [
    {"category": "Extension territoriale (EP / PCT / Madrid)", "confidence": 0.83},
    {"category": "Dossier marque (INPI / EUIPO / WIPO)", "confidence": 0.78}
  ],
  "filter": "ip",
  "confidence_ip": 0.92
}

Structure des données

Les emails sont stockés dans des fichiers JSON individuels :

mails_json/
   ├─ testjson1.json
   ├─ testjson2.json
   └─ ...


Format attendu :

{
  "email_id": 1,
  "subject": "Titre du mail",
  "body": "Contenu du mail"
}


Les pièces jointes associées doivent être placées dans :

pdfs/{email_id}/ *.pdf / *.docx

Logique de décision

Le modèle utilise une probabilité confidence_ip :

Condition
prob < 0.60 ->	not_ip -> stocké avec label not_ip
0.60 ≤ prob -> < 0.73 -> review	label review_needed
prob ≥ 0.73 ->	ip + classement par catégorie -> labels détaillés

Toutes les décisions sont tracées dans :

logs/classification_log.jsonl

Objectifs & limites

Fonctionne offline

Gère les encodages difficiles
Aide à trier automatiquement les emails PI
Le modèle nécessite du retraining pour améliorer certaines catégorie

Retraining

Pour entrainer le modèle: python3 train_is_ip.py 

Version actuelle : MVP fonctionnel
Modules prévus :

Interface Review (correction des borderline)

Dashboard Logs + Stats

Pipeline retraining automatique avec feedback utilisateur

Auteur

Développé par Tristan Duchamp