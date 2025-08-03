import os
from fastapi import FastAPI
from dotenv import load_dotenv
from copilotkit.integrations.fastapi import (
    CopilotKitRemoteEndpoint,
    add_fastapi_endpoint,
)
from copilotkit.crewai import CrewAIAgent
from copilotkit.runtime.actions import default_ui_actions

from agent import AgentState, SampleAgentFlow

load_dotenv()

app = FastAPI()

# Initialize the CrewAI Flow and expose it through CopilotKit
flow = SampleAgentFlow(state=AgentState())

agent = CrewAIAgent(
    name="starterAgent",
    crew=flow,
    actions=default_ui_actions(),
    reasoning=True,
    stream_reasoning=True,
)

sdk = CopilotKitRemoteEndpoint(
    agents=[agent],
    actions=default_ui_actions(),
)
add_fastapi_endpoint(app, sdk, "/copilotkit_remote")


def main() -> None:
    """Run the uvicorn server."""
    import uvicorn

    port = int(os.getenv("PORT", "8007"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()

