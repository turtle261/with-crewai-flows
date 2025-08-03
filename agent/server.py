import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from flow import run_flow

load_dotenv()

mcp = FastMCP("local-crewai")

@mcp.tool()
async def run_agent(query: str) -> str:
    """Expose the CrewAI flow as an MCP tool."""
    return run_flow(query)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    mcp.serve(host="0.0.0.0", port=port, transport="streamable-http")
