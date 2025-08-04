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
        python_type = str  # default type
        prop_type = prop_schema.get("type")
        
        # Handle different JSON schema types
        if prop_type == "integer":
            python_type = int
        elif prop_type == "number":
            python_type = float
        elif prop_type == "boolean":
            python_type = bool
        elif prop_type == "array":
            # Handle arrays - for now default to list[str]
            python_type = list[str]
        elif prop_type == "object":
            # Handle objects - default to dict
            python_type = dict
        # else: keep as str for string type or unknown types
        
        default = ... if prop_name in required else None
        description = prop_schema.get("description", f"Parameter {prop_name}")
        fields[prop_name] = (python_type, Field(default=default, description=description))
    
    # If no fields, create empty model
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
        result_msg = ""
        
        # Mutate flow state for specific known actions so UI can reflect changes
        if flow is not None and hasattr(flow, "state"):
            # Handle adding a proverb
            if self.name.lower() in {"addproverb", "add_proverb"}:
                # Try multiple possible parameter names
                proverb = kwargs.get("proverb") or kwargs.get("text") or kwargs.get("value")
                if proverb:
                    try:
                        if not hasattr(flow.state, 'proverbs'):
                            flow.state.proverbs = []
                        flow.state.proverbs.append(str(proverb))  # type: ignore[attr-defined]
                        result_msg = f"📜 Added proverb: {proverb}"
                    except Exception as e:  # pragma: no cover
                        result_msg = f"Failed to add proverb: {e}"
            
        
        # Send events to CopilotKit frontend
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
        
        return result_msg or f"Successfully executed CopilotKit action '{self.name}' with parameters: {kwargs}"
