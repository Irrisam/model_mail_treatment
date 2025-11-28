from app.db.connection import get_db
from app.db.queries import find_user_by_email

async def lookup_sender(params):
    email = params.get("email")
    if not email:
        return {"found": False, "user": None}

    conn = get_db()
    user = find_user_by_email(conn, email)

    return {
        "found": user is not None,
        "user": user
    }

lookup_sender_tool = {
    "name": "lookup_sender",
    "schema": {
        "name": "lookup_sender",
        "description": "Recherche un utilisateur dans la base via son email",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"]
        }
    },
    "func": lookup_sender,
}
