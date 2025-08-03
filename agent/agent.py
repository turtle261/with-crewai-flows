"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
This implementation uses a proper CrewAI agent and crew with FileReadTool,
while maintaining CopilotKit generative UI capabilities.
"""
import json
import os
from litellm import completion, acompletion
from crewai.flow.flow import Flow, start, router, listen
from ag_ui_crewai.sdk import copilotkit_stream, CopilotKitState
from crewai import LLM, Agent, Crew, Task, Process
from crewai_tools import FileReadTool

class AgentState(CopilotKitState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: list[str] = []
    # your_custom_agent_state: str = ""

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string", 
                    "description": "The city and state, e.g. San Francisco, CA"
                    }
                    },
            "required": ["location"]
        }
    }
}

tools = [
    GET_WEATHER_TOOL
    # your_tool_here
]

tool_handlers = {
    "get_weather": lambda args: f"The weather for {args['location']} is 70 degrees, clear skies, 45% humidity, 5 mph wind, and feels like 72 degrees.",
    # your tool handler here
}

class SampleAgentFlow(Flow[AgentState]):
    """
    This flow uses both CrewAI agents with tools (like FileReadTool) for backend processing,
    and CopilotKit actions for generative UI capabilities.
    """

    def __init__(self):
        super().__init__()
        self._crew = None
        self._setup_crew()

    def _setup_crew(self):
        """Setup the CrewAI agent and crew with FileReadTool"""
        # Create FileReadTool for reading files from knowledge directory
        file_read_tool = FileReadTool()
        
        # Create CrewAI agent
        self.agent = Agent(
            role="Knowledge Assistant",
            goal="Help users by accessing information from files and providing helpful responses",
            backstory=(
                "You are a knowledgeable assistant with access to file reading capabilities. "
                "You can read and analyze files to provide comprehensive answers. "
                "When users ask questions, you should use your file reading tools when appropriate "
                "to access relevant information."
            ),
            tools=[file_read_tool],
            llm="gemini/gemini-2.0-flash",
            verbose=False,  # Disable verbose to avoid Unicode encoding issues on Windows
            allow_delegation=False
        )

    def _create_crew_task(self, user_message: str) -> Task:
        """Create a task for the crew based on user message"""
        return Task(
            description=f"Process this user request: {user_message}",
            expected_output="A helpful response that addresses the user's request",
            agent=self.agent
        )

    async def _execute_crew(self, user_message: str) -> str:
        """Execute CrewAI crew for backend processing"""
        try:
            task = self._create_crew_task(user_message)
            crew = Crew(
                agents=[self.agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False  # Disable verbose to avoid Unicode encoding issues on Windows
            )
            
            result = crew.kickoff()
            return result.raw if hasattr(result, 'raw') else str(result)
        except Exception as e:
            return f"Error processing request with crew: {str(e)}"

    @start()
    @listen("route_follow_up")
    async def start_flow(self):
        """
        This is the entry point for the flow.
        """

    @router(start_flow)
    async def chat(self):
        """
        Hybrid chat node that combines CrewAI backend processing with CopilotKit UI actions.
        - Uses CrewAI agent with FileReadTool for file operations and knowledge access
        - Maintains CopilotKit actions for generative UI capabilities
        - Handles tool calls appropriately based on their type
        """
        # Get the latest user message
        user_message = ""
        if self.state.messages:
            user_message = self.state.messages[-1].get("content", "")

        # First, check if this should be handled by CrewAI crew
        # This is a simple heuristic - in practice, you might want more sophisticated routing
        should_use_crew = (
            "file" in user_message.lower() or 
            "read" in user_message.lower() or 
            "knowledge" in user_message.lower() or
            "secret" in user_message.lower() or
            len(user_message) > 100  # Complex requests go to crew
        )

        if should_use_crew:
            # Use CrewAI crew for backend processing
            crew_result = await self._execute_crew(user_message)
            
            # Add crew result as assistant message
            self.state.messages.append({
                "role": "assistant", 
                "content": crew_result
            })
            return "route_end"

        # For simpler requests or UI actions, use the direct LLM approach
        system_prompt = f"You are a helpful assistant. The current proverbs are {self.state.proverbs}."

        llm = "gemini/gemini-2.0-flash"
        wrapper = await acompletion(
            model=llm,
            messages=[
                {"role": "system", "content": system_prompt},
                *self.state.messages
            ],
            tools=[
                *self.state.copilotkit.actions,
                GET_WEATHER_TOOL
            ],
            parallel_tool_calls=False,
            stream=True,
            drop_params=True,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        response = await copilotkit_stream(wrapper)
        message = response.choices[0].message

        # Append the message to the messages in state
        self.state.messages.append(message)

        # Handle tool calls
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            tool_call_id = tool_call["id"]
            tool_call_name = tool_call["function"]["name"]
            tool_call_args = json.loads(tool_call["function"]["arguments"])

            # Check for CopilotKit actions
            if (tool_call_name in
                [action["function"]["name"] for action in self.state.copilotkit.actions]):
                return "route_end"

            # Handle backend tool calls
            handler = tool_handlers[tool_call_name]
            result = handler(tool_call_args)

            # Append the result to the messages in state
            self.state.messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call_id
            })

            return "route_follow_up"

        return "route_end"

    @listen("route_end")
    async def end(self):
        """
        End the flow.
        """