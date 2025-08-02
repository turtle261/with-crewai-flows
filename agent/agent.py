"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""
import json
from litellm import completion, acompletion
from crewai.flow.flow import Flow, start, router, listen
from ag_ui_crewai.sdk import copilotkit_stream, CopilotKitState
from crewai import LLM
import os

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
    This is a sample flow that uses the CopilotKit framework to create a chat agent.
    """

    @start()
    @listen("route_follow_up")
    async def start_flow(self):
        """
        This is the entry point for the flow.
        """

    @router(start_flow)
    async def chat(self):
        """
        Standard chat node based on the ReAct design pattern. It handles:
        - The model to use (and binds in CopilotKit actions and the tools defined above)
        - The system prompt
        - Getting a response from the model
        - Handling tool calls

        For more about the ReAct design pattern, see: 
        https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
        """
        system_prompt = f"You are a helpful assistant. The current proverbs are {self.state.proverbs}."

        # 1. Run the model and stream the response
        #    Note: In order to stream the response, wrap the completion call in
        #    copilotkit_stream and set stream=True.
        llm = "gemini/gemini-2.0-flash"
        # 1.a Obtain the async stream wrapper from LLM
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
        # 1.b Stream through CopilotKit
        response = await copilotkit_stream(wrapper)

        message = response.choices[0].message

        # 2. Append the message to the messages in state
        self.state.messages.append(message)

        # 3. Handle tool calls
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            tool_call_id = tool_call["id"]
            tool_call_name = tool_call["function"]["name"]
            tool_call_args = json.loads(tool_call["function"]["arguments"])

            # 4. Check for tool calls in the response and handle them. If the tool call
            #    is a CopilotKit action, we return the response to CopilotKit to handle
            if (tool_call_name in
                [action["function"]["name"] for action in self.state.copilotkit.actions]):
                return "route_end"

            # 5. Otherwise, we handle the tool call on the backend
            handler = tool_handlers[tool_call_name]
            result = handler(tool_call_args)

            # 6. Append the result to the messages in state
            self.state.messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call_id
            })

            # 7. Return to the follow up route to continue the conversation
            return "route_follow_up"

        # 8. If there are no tool calls, return to the end route
        return "route_end"

    @listen("route_end")
    async def end(self):
        """
        End the flow.
        """