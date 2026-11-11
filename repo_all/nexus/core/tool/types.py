"""Tool type definitions for nexus framework."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Protocol, TypeVar

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel

from nexus.utils.immutable import ImmutableDict


class ToolType(str, Enum):
    """Execution location for a tool."""

    LOCAL = 'local'
    RUNTIME = 'runtime'


class ToolExposure(str, Enum):
    """How a tool is exposed to the model."""

    DIRECT = 'direct'
    DEFERRED = 'deferred'


class ToolSearchConfig(BaseModel):
    """Configuration for client-side tool search."""

    enabled: bool = False
    exposure_overrides: dict[str, ToolExposure] = {}
    defer_all_mcp: bool = False
    inject_reminder: bool = True


_TOOL_CONTEXT_IMMUTABLE_ERROR = (
    'ToolContext.{field_name} is immutable. '
    'To update state, return the changes in ToolResponse.context_updates instead.'
)


class ToolContext(BaseModel):
    """Context passed to tools during execution."""

    model_config = {'arbitrary_types_allowed': True, 'frozen': True}

    run_id: str | None = None
    executor: Any = None
    tool_configs: ImmutableDict = ImmutableDict()
    tool_state: ImmutableDict = ImmutableDict()
    trace_id: str | None = None

    def __init__(self, **data):
        if 'tool_configs' in data and not isinstance(data['tool_configs'], ImmutableDict):
            data['tool_configs'] = ImmutableDict(
                data['tool_configs'], field_name='tool_configs', error_message=_TOOL_CONTEXT_IMMUTABLE_ERROR
            )
        if 'tool_state' in data and not isinstance(data['tool_state'], ImmutableDict):
            data['tool_state'] = ImmutableDict(
                data['tool_state'], field_name='tool_state', error_message=_TOOL_CONTEXT_IMMUTABLE_ERROR
            )
        super().__init__(**data)


class ToolResponse(BaseModel):
    """Standardized response from tool execution."""

    output: list[ChatCompletionContentPartParam | dict]
    meta: dict[str, Any] = {}
    context_updates: dict[str, Any] = {}


InputT = TypeVar('InputT', bound=BaseModel)


class ToolFunc(Protocol[InputT]):
    """Protocol for tool functions."""

    __tool_input_model__: type[InputT]
    __tool_meta__: ToolMeta
    nexus_meta: dict[str, Any]

    async def __call__(self, params: InputT, context: ToolContext) -> ToolResponse: ...


class ToolParamOverride(BaseModel):
    """Parameter-level metadata override."""

    description: str | None = None


class ToolMetaOverride(BaseModel):
    """Tool metadata override configuration."""

    name: str | None = None
    description: str | None = None
    params: dict[str, ToolParamOverride] = {}


class ToolMeta(BaseModel):
    """Metadata for a registered tool."""

    name: str
    description: str
    tool_type: ToolType
    input_model: type[BaseModel]
    func: Callable[..., Awaitable[ToolResponse]]
    parallel_func: Callable[..., Awaitable[list[ToolResponse]]] | None = None

    model_config = {'arbitrary_types_allowed': True}

    @property
    def schema(self) -> dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self._generate_parameters_schema(),
            },
        }

    def _generate_parameters_schema(self) -> dict[str, Any]:
        full_schema = self.input_model.model_json_schema()
        defs = full_schema.get('$defs', {})

        result = self._process_schema_node(full_schema, defs)

        model_config = getattr(self.input_model, 'model_config', {})
        json_schema_extra = model_config.get('json_schema_extra', {}) if isinstance(model_config, dict) else {}
        if isinstance(json_schema_extra, dict) and 'description' in json_schema_extra:
            result['description'] = json_schema_extra['description']

        return result

    def _process_schema_node(self, node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
        if '$ref' in node:
            ref_path = node['$ref']
            if ref_path.startswith('#/$defs/'):
                ref_name = ref_path.split('/')[-1]
                if ref_name in defs:
                    return self._process_schema_node(defs[ref_name], defs)

        result: dict[str, Any] = {}

        if 'type' in node:
            result['type'] = node['type']

        if 'anyOf' in node:
            non_null_types = [t for t in node['anyOf'] if t.get('type') != 'null']
            if len(non_null_types) == 1:
                result.update(self._process_schema_node(non_null_types[0], defs))
            elif len(non_null_types) > 1:
                result['anyOf'] = [self._process_schema_node(t, defs) for t in non_null_types]

        if 'description' in node:
            result['description'] = node['description']

        if 'default' in node:
            result['default'] = node['default']

        if node.get('type') == 'object' and 'properties' in node:
            result['type'] = 'object'
            result['properties'] = {}
            for prop_name, prop_schema in node['properties'].items():
                result['properties'][prop_name] = self._process_schema_node(prop_schema, defs)
            if 'required' in node:
                result['required'] = node['required']

        if node.get('type') == 'array' and 'items' in node:
            result['type'] = 'array'
            result['items'] = self._process_schema_node(node['items'], defs)

        if 'enum' in node:
            result['enum'] = node['enum']

        return result


__all__ = [
    'InputT',
    'ToolContext',
    'ToolExposure',
    'ToolFunc',
    'ToolMeta',
    'ToolMetaOverride',
    'ToolParamOverride',
    'ToolResponse',
    'ToolSearchConfig',
    'ToolType',
]
