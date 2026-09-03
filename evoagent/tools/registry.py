"""The explicit tool catalog every agent role is given.

A registry is the whole contract between a role and the outside world: only
registered tools can be called, and every call is validated against its declared
JSON-Schema-like parameters before the handler sees it. Nothing here knows about
reviews, findings or workflows.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from ..errors import ToolProtocolError


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]

    def catalog_entry(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Explicit tool catalog with JSON-schema-like argument validation."""

    def __init__(self, tools: Iterable[AgentTool] = ()):
        self._tools: Dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.name or tool.name in self._tools:
            raise ValueError("tool names must be non-empty and unique")
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return sorted(self._tools)

    def catalog(self) -> List[Dict[str, Any]]:
        return [self._tools[name].catalog_entry() for name in self.names()]

    def invoke(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolProtocolError("unknown agent tool: %s" % name)
        self._validate(tool.parameters, arguments)
        return tool.handler(**arguments)

    @staticmethod
    def _validate(schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        if not isinstance(arguments, dict):
            raise ToolProtocolError("tool arguments must be an object")
        properties = dict(schema.get("properties") or {})
        required = set(schema.get("required") or [])
        missing = required.difference(arguments)
        if missing:
            raise ToolProtocolError(
                "missing required tool arguments: %s" % ", ".join(sorted(missing))
            )
        if schema.get("additionalProperties", False) is False:
            unknown = set(arguments).difference(properties)
            if unknown:
                raise ToolProtocolError(
                    "unknown tool arguments: %s" % ", ".join(sorted(unknown))
                )
        expected_types = {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list,
        }
        for key, value in arguments.items():
            spec = properties.get(key) or {}
            expected = expected_types.get(spec.get("type"))
            if expected and (not isinstance(value, expected) or (
                spec.get("type") in {"integer", "number"} and isinstance(value, bool)
            )):
                raise ToolProtocolError(
                    "tool argument %s must be %s" % (key, spec.get("type"))
                )
            if isinstance(value, (int, float)):
                if "minimum" in spec and value < spec["minimum"]:
                    raise ToolProtocolError("tool argument %s is below minimum" % key)
                if "maximum" in spec and value > spec["maximum"]:
                    raise ToolProtocolError("tool argument %s exceeds maximum" % key)
