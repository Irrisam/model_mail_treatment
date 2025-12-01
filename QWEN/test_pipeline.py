import asyncio
from app.mcp.run_pipeline import run_pipeline


async def main():
    result = await run_pipeline({
        "email_id": 1,
        "email": 'zizi_prout',
        "subject": "Demande de RIB",
        "body": "Bonjour, merci de m'envoyer mon RIB."
    })

    print("\n=== Résultat du pipeline ===")
    print(result)

asyncio.run(main())
