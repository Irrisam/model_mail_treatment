from fastapi import FastAPI
from app.mcp.router import mcp_router

app = FastAPI()

app.include_router(mcp_router, prefix="/mcp")
