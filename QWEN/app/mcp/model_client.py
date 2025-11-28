import httpx

MODEL_URL = "http://localhost:8010"

async def classify_email_api(email_id: int, subject: str, body: str):
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{MODEL_URL}/classify",
            json={
                "email_id": email_id,
                "subject": subject,
                "body": body
            }
        )
    response.raise_for_status()
    return response.json()


async def classify_batch_api():
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(f"{MODEL_URL}/batch")
    response.raise_for_status()
    return response.json()


async def model_health():
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{MODEL_URL}/health")
    response.raise_for_status()
    return response.json()
