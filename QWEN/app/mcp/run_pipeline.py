from app.mcp.model_client import classify_email_api


async def run_pipeline(params):
    email_id = params.get("email_id")
    subject = params.get("subject", "")
    body = params.get("body", "")

    if not email_id:
        return {"error": "email_id is required"}

    # --- étape clé : appel du modèle ---
    result = await classify_email_api(email_id, subject, body)

    return {
        "status": "ok",
        "model_result": result
    }


run_pipeline_tool = {
    "name": "run_pipeline",
    "schema": {
        "name": "run_pipeline",
        "description": "Pipeline de traitement d'email avec classification modèle ML",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {"type": "integer"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["email_id", "subject", "body"]
        }
    },
    "func": run_pipeline,
}
