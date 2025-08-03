from crewai import Agent, Task, Flow, LLM
from crewai_tools import FileReadTool
import os

# Initialize Gemini LLM via LiteLLM
llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    max_rpm=10
)

# Built-in FileReadTool (additional tools can be added to this list)
file_tool = FileReadTool()

agent = Agent(
    role="Assistant",
    goal="Answer queries; read files when needed.",
    backstory="A single-agent Crew with file-reading skills.",
    llm=llm,
    tools=[file_tool],
    reasoning=True,
    max_reasoning_attempts=3
)

task = Task(
    description="Respond to the user's query using tools if required.",
    expected_output="Final answer to the user",
    agent=agent,
    markdown=True
)

# Flow with single agent and task
flow = Flow(agents=[agent], tasks=[task])
