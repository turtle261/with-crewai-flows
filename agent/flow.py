import os
from crewai import Agent, Task, Flow, LLM
from crewai_tools import FileReadTool

# Initialize Gemini LLM via LiteLLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Register tools (extensible list)
file_tool = FileReadTool()

agent = Agent(
    role="Assistant",
    goal="Answer queries; read files when needed.",
    backstory="A single-agent Crew with file-reading skills.",
    llm=llm,
    tools=[file_tool],
    reasoning=True,
    max_reasoning_attempts=3,
    max_rpm=10,
)

task = Task(
    description="Respond to the user's query using tools if required.",
    agent=agent,
    expected_output="A helpful answer for the user.",
    markdown=True,
)

flow = Flow(agents=[agent], tasks=[task])

def run_flow(query: str) -> str:
    """Run the CrewAI flow for the given user query."""
    task.description = f'User asks: "{query}"'
    flow.kickoff()
    return task.output or ""
