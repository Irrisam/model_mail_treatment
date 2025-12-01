from app.mcp.model_client import classify_email_api
from app.mcp.lookup_sender import lookup_sender
from app.mcp.trust import compute_trust_score
from app.db.connection import get_db
from app.db.queries import find_user_by_client_code


async def run_pipeline(params):
    email_id = params.get("email_id")
    subject = params.get("subject", "")
    body = params.get("body", "")
    sender_email = params.get("sender_email")

    if not email_id:
        return {"error": "email_id is required"}

    # 1) Appel du modèle ML (classify)
    model_result = await classify_email_api(email_id, subject, body)

    confidence_ip = float(model_result.get("confidence_ip", 0.0))
    # renvoyé par le service modèle
    client_code = model_result.get("client_code")
    client_code_found = client_code is not None

    # 2) Lookup via l'email (si fourni)
    if sender_email:
        lookup_result = await lookup_sender({"email": sender_email})
    else:
        lookup_result = {"found": False, "user": None}

    sender_found = lookup_result.get("found", False)

    # 3) Lookup via le code client (si présent)
    client_lookup = None
    client_match_found = False

    if client_code_found:
        conn = get_db()
        client_lookup = find_user_by_client_code(conn, client_code)
        client_match_found = client_lookup is not None

    # 4) Calcul du trust_score global
    trust_score = compute_trust_score(
        confidence_ip=confidence_ip,
        sender_found=sender_found,
        client_code_found=client_code_found,
        client_match_found=client_match_found,
    )

    # 5) Réponse consolidée
    return {
        "status": "ok",
        "email_id": email_id,
        "model_result": model_result,
        "sender_lookup": lookup_result,
        "client_code": client_code,
        "client_lookup": client_lookup,
        "trust_score": trust_score,
    }


run_pipeline_tool = {
    "name": "run_pipeline",
    "schema": {
        "name": "run_pipeline",
        "description": "Pipeline de traitement d'email avec ML + lookup + trust-score",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {"type": "integer"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "sender_email": {"type": "string"},
            },
            "required": ["email_id", "subject", "body"],
        },
    },
    "func": run_pipeline,
}
