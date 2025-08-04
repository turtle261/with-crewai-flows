"""Main CrewAI flow integrating CopilotKit actions."""
from crewai.flow.flow import Flow, start, router, listen
from ag_ui_crewai.sdk import CopilotKitState
from crewai import Agent, Crew, Task, Process
from crewai_tools import FileReadTool

from copilot_bridge_tool import CopilotBridgeTool


class AgentState(CopilotKitState):
    """State carried between CopilotKit and CrewAI."""
    proverbs: list[str] = []


class SampleAgentFlow(Flow[AgentState]):
    """Flow that exposes a single CrewAI agent with CopilotKit tools."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_crew()

    def _setup_crew(self) -> None:
        """Initialize the CrewAI agent with the FileReadTool."""
        self.file_read_tool = FileReadTool()
        self.agent = Agent(
            role="Knowledge Assistant",
            goal="Help users by accessing information from files and providing helpful responses",
            backstory=(
                "You are a knowledgeable assistant with access to file reading capabilities. "
                "You can read and analyze files to provide comprehensive answers. "
                "When users ask questions, you should use your tools when appropriate to access relevant information."
            ),
            tools=[self.file_read_tool],
            llm="gemini/gemini-2.0-flash",
            verbose=False,
            allow_delegation=False,
        )

    def _update_agent_tools(self) -> None:
        """Refresh agent tools with current CopilotKit actions."""
        action_tools = [CopilotBridgeTool(action, flow=self) for action in self.state.copilotkit.actions]
        self.agent.tools = [self.file_read_tool, *action_tools]

    def _create_crew_task(self, user_message: str) -> Task:
        return Task(
            description=f"Process this user request: {user_message}",
            expected_output="A helpful response that addresses the user's request",
            agent=self.agent,
        )

    async def _execute_crew(self, user_message: str) -> str:
        try:
            self._update_agent_tools()
            task = self._create_crew_task(user_message)
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            return result.raw if hasattr(result, "raw") else str(result)
        except Exception as exc:  # pylint: disable=broad-except
            return f"Error processing request with crew: {exc}"

    @start()
    @listen("route_end")
    async def start_flow(self) -> None:
        """Entry point for the flow."""

    @router(start_flow)
    async def chat(self) -> str:
        """Main chat node using the CrewAI agent for all capabilities."""
        user_message = ""
        if self.state.messages:
            user_message = self.state.messages[-1].get("content", "")

        crew_result = await self._execute_crew(user_message)
        self.state.messages.append({"role": "assistant", "content": crew_result})
        return "route_end"

    @listen("route_end")
    async def end(self) -> None:
        """Flow terminator."""
