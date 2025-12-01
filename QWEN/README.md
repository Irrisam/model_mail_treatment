Pour activer les serveurs:

Du model: 
cd QWEN/model_1_0
uvicorn model_service:app --reload --port 8010


Du MCP:
cd QWEN
uvicorn app.main:app --reload --port 9000



Dataflow

  incoming_email
                    │
                    ▼
             MODEL SERVICE
    (classe email, détecte PI / labels)
                    │
         return JSON with:
         - is_ip / not_ip / review_needed
         - labels
         - confidence_ip
                    │
                    ▼
                MCP / RUN_PIPELINE
          (vérifie sender + logique métier)
                    │
          lookup_sender(email)
                    │
                    ▼
   - found  → score de confiance augmenté
   - not found → marquer comme inconnu

------------------------------

   run_pipeline
   │
   ├── model_client.classify_email_api()
   │        → labels, filter, confidence_ip
   │
   ├── lookup_sender(email)
   │        → found / not found
   │
   ├── compute_trust_score()
   │        → valeur finale
   │
   └── renvoie un JSON consolidé


------------------------------------------------------------------------------------------

   Pour test les apis: POSTMAN:

POST 
URL: http://localhost:8010/classify

   Body = Raw + JSON
   {
  "email_id": 1,
  "subject": "Demande de RIB",
  "body": "Bonjour, merci de m'envoyer mon RIB s'il vous plaît."
}


------------------------------
POST

URL : http://localhost:9000/mcp/invoke

Body = Raw + JSON:
{
  "tool": "run_pipeline",
  "params": {
    "email_id": 1,
    "subject": "Demande de RIB",
    "body": "Bonjour, merci de m'envoyer mon RIB."
  }
}