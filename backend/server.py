import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from .flow import agent, task
from crewai import Crew

load_dotenv()

mcp = FastMCP("local-crewai")

@mcp.tool()
async def run_agent(query: str) -> str:
    """Run the CrewAI flow to answer the query."""
    task.description = f'User asks: "{query}"'
    crew = Crew(agents=[agent], tasks=[task])
    return crew.kickoff()

if __name__ == "__main__":
    mcp.serve(host="0.0.0.0", port=8000, transport="streamable-http")
