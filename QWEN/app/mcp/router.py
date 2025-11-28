from fastapi import APIRouter, Request

from app.mcp.lookup_sender import lookup_sender_tool
from app.mcp.run_pipeline import run_pipeline_tool
# from app.mcp.save_sender import save_sender_tool
# from app.mcp.search_sender import search_sender_tool

mcp_router = APIRouter()

TOOLS = {
    lookup_sender_tool["name"]: lookup_sender_tool,
    run_pipeline_tool["name"]: run_pipeline_tool,
    # save_sender_tool["name"]: save_sender_tool,
    # search_sender_tool["name"]: search_sender_tool,
}


@mcp_router.get("/schema")
async def mcp_schema():
    return {
        "tools": [tool["schema"] for tool in TOOLS.values()]
    }


@mcp_router.post("/invoke")
async def mcp_invoke(request: Request):
    payload = await request.json()
    tool_name = payload.get("tool")
    params = payload.get("params", {})

    tool = TOOLS.get(tool_name)
    if not tool:
        return {"error": f"Unknown tool: {tool_name}"}

    return await tool["func"](params)
