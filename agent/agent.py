"""
Main entry point for the CrewAI Flow used by the application.

This flow powers a chat agent that can interact with CopilotKit's default
UI actions, call custom tools such as a weather lookup, and utilise
CrewAI's FileReadTool to read files from disk when requested.
"""

import json
import os

from crewai.flow.flow import Flow, listen, router, start
from copilotkit.crewai import CopilotKitState, copilotkit_stream
from crewai_tools import FileReadTool
from litellm import acompletion


class AgentState(CopilotKitState):
    """State for the chat agent."""

    proverbs: list[str] = []


# ----- Tool definitions ----------------------------------------------------

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
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
}


file_reader = FileReadTool()

FILE_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file on disk",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Full path to the file to read",
                }
            },
            "required": ["file_path"],
        },
    },
}


tool_handlers = {
    "get_weather": lambda args: (
        f"The weather for {args['location']} is 70 degrees, clear skies, "
        "45% humidity, 5 mph wind, and feels like 72 degrees."
    ),
    "read_file": lambda args: file_reader.run(**args),
}


class SampleAgentFlow(Flow[AgentState]):
    """A simple chat flow demonstrating tool usage and streaming."""

    @start()
    @listen("route_follow_up")
    async def start_flow(self):
        """Entry point for the flow."""

    @router(start_flow)
    async def chat(self):
        """Chat node implementing the ReAct-style loop."""

        system_prompt = (
            f"You are a helpful assistant. The current proverbs are {self.state.proverbs}."
        )

        # 1. Run the model and stream the response through CopilotKit
        wrapper = await acompletion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                *self.state.messages,
            ],
            tools=[
                *self.state.copilotkit.actions,
                GET_WEATHER_TOOL,
                FILE_READ_TOOL,
            ],
            parallel_tool_calls=False,
            stream=True,
            drop_params=True,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

        response = await copilotkit_stream(wrapper)
        message = response.choices[0].message
        self.state.messages.append(message)

        # 2. Handle tool calls, if any
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            tool_call_id = tool_call["id"]
            tool_call_name = tool_call["function"]["name"]
            tool_call_args = json.loads(tool_call["function"]["arguments"])

            if tool_call_name in [
                action["function"]["name"] for action in self.state.copilotkit.actions
            ]:
                return "route_end"

            handler = tool_handlers[tool_call_name]
            result = handler(tool_call_args)

            self.state.messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call_id,
                }
            )

            return "route_follow_up"

        return "route_end"

    @listen("route_end")
    async def end(self):
        """End of the flow."""

