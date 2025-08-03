import asyncio
import importlib.util
import sys
from pathlib import Path

AGENT_SPEC = importlib.util.spec_from_file_location("agent_module", Path("agent/agent.py"))
agent_module = importlib.util.module_from_spec(AGENT_SPEC)
AGENT_SPEC.loader.exec_module(agent_module)
sys.modules["agent"] = agent_module

tool_handlers = agent_module.tool_handlers
tools = agent_module.tools
SampleAgentFlow = agent_module.SampleAgentFlow
AgentState = agent_module.AgentState


def test_file_read_tool_handler():
    assert "FileReadTool" in tool_handlers
    result = tool_handlers["FileReadTool"]({"file_path": "agent/knowledge/secret.md"})
    assert "Banana" in result


def test_tools_include_file_read_tool():
    assert any(t["function"]["name"] == "FileReadTool" for t in tools)


async def _run_copilot_action_test():
    flow = SampleAgentFlow()
    flow.state.copilotkit.actions = [
        {
            "type": "function",
            "function": {
                "name": "addProverb",
                "description": "Add a proverb",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    flow.state.messages = [
        {"role": "user", "content": "Add a proverb", "id": "m1"}
    ]

    async def fake_acompletion(**kwargs):
        class Response:
            def __init__(self):
                self.choices = [
                    type(
                        "Choice",
                        (),
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "1",
                                        "function": {"name": "addProverb", "arguments": "{}"},
                                    }
                                ],
                            }
                        },
                    )
                ]
        return Response()

    agent_module.acompletion = fake_acompletion
    async def identity(wrapper):
        return wrapper
    agent_module.copilotkit_stream = identity

    result = await flow.chat()
    assert result == "route_end"


def test_frontend_action_routing():
    asyncio.run(_run_copilot_action_test())
