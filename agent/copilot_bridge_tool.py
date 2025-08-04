"""Tool to bridge CopilotKit actions to CrewAI tools."""
import json
import uuid
from typing import Any, Optional, Dict

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, create_model
from ag_ui.core.events import (
    EventType,
    ToolCallStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
)
from ag_ui_crewai.endpoint import get_queue
from ag_ui_crewai.context import flow_context


def _args_model_from_schema(name: str, schema: Dict[str, Any]) -> type[BaseModel]:
    """Create a pydantic model from a JSON schema."""
    props = schema.get("properties", {})
    required = schema.get("required", [])
    fields = {}
    for prop_name, prop_schema in props.items():
        python_type = str
        t = prop_schema.get("type")
        if t == "integer":
            python_type = int
        elif t == "number":
            python_type = float
        elif t == "boolean":
            python_type = bool
        default = ... if prop_name in required else None
        fields[prop_name] = (python_type, Field(default=default, description=prop_schema.get("description")))
    if not fields:
        return create_model(f"{name}Args", __base__=BaseModel)
    return create_model(f"{name}Args", **fields)


class CopilotBridgeTool(BaseTool):
    """Bridge a CopilotKit action so CrewAI can invoke it."""

    def __init__(self, action: Dict[str, Any], flow: Optional[Any] = None):
        func = action.get("function", {})
        name = func.get("name", "copilot_action")
        description = func.get("description", "")
        parameters = func.get("parameters", {})
        args_model = _args_model_from_schema(name, parameters)
        super().__init__(name=name, description=description, args_schema=args_model)
        self._flow = flow

    def _run(self, **kwargs):  # type: ignore[override]
        flow = self._flow or flow_context.get(None)
        queue = get_queue(flow)
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        if queue is not None:
            queue.put_nowait(
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=call_id,
                    tool_call_name=self.name,
                )
            )
            if kwargs:
                queue.put_nowait(
                    ToolCallArgsEvent(
                        type=EventType.TOOL_CALL_ARGS,
                        tool_call_id=call_id,
                        delta=json.dumps(kwargs),
                    )
                )
            queue.put_nowait(
                ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=call_id,
                )
            )
        return f"Sent CopilotKit action {self.name}"
