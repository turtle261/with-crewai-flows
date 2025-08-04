"""Main CrewAI flow integrating CopilotKit actions."""
from crewai.flow.flow import Flow, start
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
            role="Knowledge Assistant and UI Controller",
            goal=(
                "Help users by accessing information from files, managing UI state, and coordinating "
                "between CopilotKit frontend actions and CrewAI tools seamlessly"
            ),
            backstory=(
                "You are a sophisticated assistant with access to both file reading capabilities "
                "and frontend UI controls through CopilotKit. You can read and analyze files, "
                "add proverbs to the UI, and perform other interactive tasks. "
                "When users ask questions, use your tools appropriately - read files for information "
                "and use CopilotKit actions to update the UI state. Always be helpful and thorough."
            ),
            tools=[self.file_read_tool],
            llm="gemini/gemini-2.0-flash",
            verbose=True,  # Enable verbose mode for better debugging
            allow_delegation=False,
        )

    def _update_agent_tools(self) -> None:
        """Refresh agent tools with all CopilotKit actions and CrewAI tools."""
        copilotkit_tools = []
        if hasattr(self.state, 'copilotkit'):
            actions = getattr(self.state.copilotkit, 'actions', [])
            # Add explicit addProverb action if not present
            if not any(a.get('name') == 'addProverb' for a in actions):
                actions.append({'name': 'addProverb', 'description': 'Add a proverb to the UI', 'args': ['proverb']})
            copilotkit_tools = [CopilotBridgeTool(action, flow=self) for action in actions]
        self.agent.tools = [self.file_read_tool, *copilotkit_tools]
        tool_names = [getattr(tool, 'name', str(tool)) for tool in self.agent.tools]
        print(f"Updated agent tools: {tool_names}")

    def _create_crew_task(self, user_message: str) -> Task:
        """Create a task for the crew (deprecated - now handled in _execute_crew)."""
        return Task(
            description=f"Process this user request: {user_message}",
            expected_output="A helpful response that addresses the user's request",
            agent=self.agent,
        )

    async def _execute_crew(self, user_message: str) -> str:
        try:
            # Ensure agent tools are up to date with latest CopilotKit actions
            self._update_agent_tools()
            
            # Create a more detailed task description to help the agent understand context
            task_description = f"""
            Process this user request: {user_message}
            
            Available tools:
            - FileReadTool: Read files from the knowledge directory or any other files
            - CopilotKit actions: Update UI state (add proverbs)

            If the user mentions reading files, use the FileReadTool.
            If the user wants to add proverbs, use the appropriate CopilotKit actions.
            Always provide helpful, comprehensive responses.
            """
            
            task = Task(
                description=task_description,
                expected_output="A helpful response that addresses the user's request and updates UI state as needed",
                agent=self.agent,
            )
            
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,  # Enable verbose mode for better debugging
            )
            
            result = crew.kickoff()
            return result.raw if hasattr(result, "raw") else str(result)
            
        except Exception as exc:  # pylint: disable=broad-except
            error_msg = f"Error processing request with crew: {exc}"
            print(f"CrewAI Error: {error_msg}")
            return error_msg

    @start()
    async def start_flow(self) -> str:
        """Entry point for the flow."""
        user_message = ""
        if self.state.messages:
            user_message = self.state.messages[-1].get("content", "")

        print(f"Processing user message: {user_message}")
        print(f"Current state - proverbs: {len(self.state.proverbs)}")
        
        crew_result = await self._execute_crew(user_message)
        
        # Update messages with the result
        self.state.messages.append({"role": "assistant", "content": crew_result})
        
        print(f"Crew result: {crew_result}")
        print(f"Updated state - proverbs: {len(self.state.proverbs)}")
        
        # Return the result directly - no routing needed for simple chat
        return crew_result
