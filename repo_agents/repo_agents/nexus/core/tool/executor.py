"""Tool executor for unified tool calling in agents."""

from __future__ import annotations

import asyncio
import copy
import logging

from typing import Any

from pydantic import BaseModel, ValidationError

from nexus.core.context import require_context
from nexus.core.registry import EntryNotFoundError
from nexus.utils.cache import cacheable
from nexus.utils.timeout import with_timeout

from .registry import get_tool_meta, tool_registry
from .types import (
    ToolContext,
    ToolExposure,
    ToolFunc,
    ToolMetaOverride,
    ToolParamOverride,
    ToolResponse,
    ToolSearchConfig,
    ToolType,
)


logger = logging.getLogger(__name__)
BRIDGE_TOOL_NAMES = ('tool_search', 'tool_describe', 'tool_call')


class ToolNameResolutionError(KeyError):
    """Raised when a model-visible tool name cannot be resolved."""

    def __init__(self, tool_name: str, available_schema_names: list[str]):
        self.tool_name = tool_name
        self.available_schema_names = available_schema_names
        self.message_for_model = f"Tool '{tool_name}' not found. Available tools: {available_schema_names}"
        super().__init__(self.message_for_model)

    def __str__(self) -> str:
        return self.message_for_model


class ToolArgumentValidationError(ValueError):
    """Raised when tool call arguments do not satisfy the tool input model."""

    def __init__(self, *, schema_name: str, canonical_name: str, message_for_model: str):
        self.schema_name = schema_name
        self.canonical_name = canonical_name
        self.message_for_model = message_for_model
        super().__init__(message_for_model)


class ToolExecutor:
    """Unified tool executor for static and context-loaded MCP tools."""

    def __init__(
        self,
        tools: list[ToolFunc],
        *,
        allowed_tool_names: list[str] | None = None,
        tool_search_config: ToolSearchConfig | dict[str, Any] | None = None,
    ):
        self._tools: dict[str, ToolFunc] = {}
        self._instance_overrides: dict[str, ToolMetaOverride] = {}
        self._allowed_tool_names = list(dict.fromkeys(allowed_tool_names)) if allowed_tool_names is not None else None
        self._tool_search_config = ToolSearchConfig.model_validate(tool_search_config or {})
        self._tool_search_enabled = self._tool_search_config.enabled
        self._bridge_tool_names = BRIDGE_TOOL_NAMES if self._tool_search_enabled else ()

        for tool in tools:
            self._tools[tool.nexus_meta['name']] = tool

        self._provided_tool_names = set(self._tools)
        if self._tool_search_enabled:
            self._inject_bridge_tools()
            self._validate_tool_search_config()

    def _inject_bridge_tools(self) -> None:
        import nexus.extensions.tools.tool_search  # noqa: F401

        for tool_name in self._bridge_tool_names:
            self._tools[tool_name] = tool_registry.get(tool_name)

    def _validate_tool_search_config(self) -> None:
        """Reject tool-search configs that would silently break the bridge tools.

        Only guards cases with real runtime consequences: a provided tool whose
        name collides with a bridge tool would be silently clobbered by
        injection, and a bridge name in ``allowed_tool_names`` resolves to a
        confusing 'not found' error later. Overrides naming bridge tools or
        unknown tools are simply ignored at call time, so they are not validated.
        """
        if not self._tool_search_enabled:
            return

        provided_bridge_tools = [name for name in self._provided_tool_names if self._is_bridge_tool(name)]
        if provided_bridge_tools:
            raise ValueError(f'Tool Search bridge tool names are reserved: {sorted(provided_bridge_tools)}')

        if self._allowed_tool_names is not None:
            bridge_allowed = [name for name in self._allowed_tool_names if self._is_bridge_tool(name)]
            if bridge_allowed:
                raise ValueError(f'Tool Search bridge tools cannot be explicitly allowed: {sorted(bridge_allowed)}')

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        self._assert_no_duplicate_schema_names()
        return [self._build_schema_for_tool(name) for name in self._visible_canonical_tool_names()]

    def set_tool_override(
        self,
        tool_name: str,
        *,
        name: str | None = None,
        description: str | None = None,
        params: dict[str, str | ToolParamOverride] | None = None,
    ) -> None:
        if tool_name not in self._canonical_tool_name_set():
            available = sorted(self._canonical_tool_name_set())
            raise KeyError(f"Tool '{tool_name}' not found. Available: {available}")

        override = self._instance_overrides.get(tool_name, ToolMetaOverride())

        if name is not None:
            override.name = name

        if description is not None:
            override.description = description

        if params:
            for param_name, param_override in params.items():
                if isinstance(param_override, str):
                    override.params[param_name] = ToolParamOverride(description=param_override)
                else:
                    override.params[param_name] = param_override

        self._instance_overrides[tool_name] = override
        logger.debug("Set override for tool '%s'", tool_name)

    def _get_mcp_manager(self):
        try:
            ctx = require_context()
        except RuntimeError:
            return None
        return ctx._mcp_manager

    def _all_canonical_tool_names(self) -> list[str]:
        names = list(self._tools.keys())
        manager = self._get_mcp_manager()
        if manager is not None:
            names.extend(manager.tools.keys())
        return names

    def _canonical_tool_names(self) -> list[str]:
        names = [name for name in self._all_canonical_tool_names() if not self._is_bridge_tool(name)]
        if self._allowed_tool_names is None:
            return names

        available = set(names)
        missing = [name for name in self._allowed_tool_names if name not in available]
        if missing:
            raise KeyError(f"Tool '{missing[0]}' not found. Available: {sorted(available)}")
        return list(self._allowed_tool_names)

    def _canonical_tool_name_set(self) -> set[str]:
        return set(self._canonical_tool_names())

    def _deferred_canonical_tool_names(self) -> list[str]:
        if not self._tool_search_enabled:
            return []

        deferred: list[str] = []
        for canonical_name in self._canonical_tool_names():
            exposure = self._tool_search_config.exposure_overrides.get(canonical_name)
            defer_mcp = (
                exposure is None and self._tool_search_config.defer_all_mcp and self._is_mcp_tool(canonical_name)
            )
            if exposure == ToolExposure.DEFERRED or defer_mcp:
                deferred.append(canonical_name)
        return deferred

    def _visible_canonical_tool_names(self) -> list[str]:
        if not self._tool_search_enabled:
            return self._canonical_tool_names()

        deferred = set(self._deferred_canonical_tool_names())
        visible = [name for name in self._canonical_tool_names() if name not in deferred]
        visible.extend(self._bridge_tool_names)
        return visible

    def _is_bridge_tool(self, canonical_name: str) -> bool:
        return canonical_name in self._bridge_tool_names

    def _is_mcp_tool(self, canonical_name: str) -> bool:
        manager = self._get_mcp_manager()
        return manager is not None and canonical_name in manager.tools

    def _is_direct_tool(self, canonical_name: str) -> bool:
        return canonical_name in self._tools and not self._is_bridge_tool(canonical_name)

    def _merge_overrides(self, tool_name: str) -> ToolMetaOverride | None:
        ctx_override: ToolMetaOverride | None = None
        try:
            ctx = require_context()
            meta_overrides = ctx.tool_configs.get('meta_overrides', {})
            if tool_name in meta_overrides:
                config = meta_overrides[tool_name]
                if isinstance(config, dict):
                    ctx_override = ToolMetaOverride.model_validate(config)
                elif isinstance(config, ToolMetaOverride):
                    ctx_override = config
        except RuntimeError:
            pass

        instance_override = self._instance_overrides.get(tool_name)
        if ctx_override is None and instance_override is None:
            return None

        merged = ToolMetaOverride()
        if ctx_override:
            merged.name = ctx_override.name
            merged.description = ctx_override.description
            merged.params = ctx_override.params.copy()
        if instance_override:
            if instance_override.name is not None:
                merged.name = instance_override.name
            if instance_override.description is not None:
                merged.description = instance_override.description
            for param_name, param_override in instance_override.params.items():
                merged.params[param_name] = param_override

        return merged

    def _apply_override_to_schema(self, schema: dict[str, Any], override: ToolMetaOverride) -> dict[str, Any]:
        schema = copy.deepcopy(schema)

        if override.name is not None:
            schema['function']['name'] = override.name

        if override.description is not None:
            schema['function']['description'] = override.description

        if override.params:
            properties = schema['function']['parameters'].get('properties', {})
            for param_name, param_override in override.params.items():
                if param_name in properties:
                    if param_override.description is not None:
                        properties[param_name]['description'] = param_override.description
                else:
                    logger.warning(
                        "Parameter '%s' not found in tool schema, available: %s",
                        param_name,
                        list(properties.keys()),
                    )

        return schema

    def _build_schema_for_tool(self, canonical_name: str) -> dict[str, Any]:
        if canonical_name in self._tools:
            schema = get_tool_meta(self._tools[canonical_name]).schema
        else:
            manager = self._get_mcp_manager()
            if manager is None or canonical_name not in manager.tools:
                raise KeyError(
                    f"Tool '{canonical_name}' not found. Available: {sorted(self._canonical_tool_name_set())}"
                )
            tool = manager.tools[canonical_name]
            schema = {
                'type': 'function',
                'function': {
                    'name': tool.full_name,
                    'description': tool.description or f'MCP tool: {tool.tool_name}',
                    'parameters': tool.input_schema or {'type': 'object', 'properties': {}},
                },
            }

        override = self._merge_overrides(canonical_name)
        if override:
            schema = self._apply_override_to_schema(schema, override)
        return schema

    def _schema_name_for_tool(self, canonical_name: str) -> str:
        return self._build_schema_for_tool(canonical_name)['function']['name']

    def _schema_name_to_tool_name(self, canonical_names: list[str] | None = None) -> dict[str, str]:
        mapping: dict[str, str] = {}
        names = canonical_names if canonical_names is not None else self._canonical_tool_names()
        for canonical_name in names:
            schema_name = self._schema_name_for_tool(canonical_name)
            if schema_name in mapping and mapping[schema_name] != canonical_name:
                raise ValueError(
                    f"Duplicate schema tool name '{schema_name}' for tools '{mapping[schema_name]}' and '{canonical_name}'"
                )
            mapping[schema_name] = canonical_name
        return mapping

    def _assert_no_duplicate_schema_names(self) -> None:
        """Raise if any two tools collide on the same model-visible schema name.

        Always checks the visible tools. When tool search is enabled it also
        includes the deferred tools (hidden from the visible schema but still
        callable via tool_call) and the bridge tools, so a collision among the
        hidden names is caught up front rather than misrouting a later tool_call.

        Building the schema-name -> tool-name map raises on the first collision;
        the map itself is discarded.
        """
        names = list(self._visible_canonical_tool_names())
        if self._tool_search_enabled:
            names.extend(self._deferred_canonical_tool_names())
        self._schema_name_to_tool_name(names)  # raises ValueError on duplicate

    def available_schema_names(self) -> list[str]:
        """Return model-visible tool names accepted by this executor."""
        self._assert_no_duplicate_schema_names()
        return sorted(self._schema_name_to_tool_name(self._visible_canonical_tool_names()))

    def _resolve_among(self, tool_name: str, candidate_names: list[str]) -> str:
        """Resolve a model-visible (schema) or canonical name within a candidate set.

        Accepts either the schema name (after overrides) or the canonical name,
        and returns the canonical name. Raises ToolNameResolutionError listing
        the resolvable schema names when there is no match.
        """
        self._assert_no_duplicate_schema_names()
        schema_map = self._schema_name_to_tool_name(candidate_names)
        if tool_name in schema_map:
            return schema_map[tool_name]
        if tool_name in set(candidate_names):
            return tool_name
        raise ToolNameResolutionError(tool_name, sorted(schema_map))

    def _resolve_tool_name(self, tool_name: str) -> str:
        """Resolve a name the model is allowed to call directly (visible + bridge)."""
        return self._resolve_among(tool_name, self._visible_canonical_tool_names())

    def _resolve_discovered_tool_name(self, tool_name: str) -> str:
        """Resolve a name reachable only through tool_call (deferred tools)."""
        candidates = (
            self._deferred_canonical_tool_names() if self._tool_search_enabled else self._canonical_tool_names()
        )
        return self._resolve_among(tool_name, candidates)

    def _tool_info_for_canonical_name(self, canonical_name: str, include_schema: bool) -> dict[str, Any]:
        schema = self._build_schema_for_tool(canonical_name)
        function_schema = schema['function']
        info: dict[str, Any] = {
            'name': function_schema['name'],
            'canonical_name': canonical_name,
            'description': function_schema.get('description', ''),
        }
        if self._is_mcp_tool(canonical_name):
            info['source'] = 'mcp'
        elif self._is_direct_tool(canonical_name):
            info['source'] = 'local'
        if include_schema:
            info['input_schema'] = function_schema.get('parameters', {'type': 'object', 'properties': {}})
        return info

    def get_deferred_tool_infos(self, include_schema: bool = False) -> list[dict[str, Any]]:
        self._assert_no_duplicate_schema_names()
        return [
            self._tool_info_for_canonical_name(canonical_name, include_schema)
            for canonical_name in self._deferred_canonical_tool_names()
        ]

    def build_tool_search_reminder(self) -> str | None:
        """Build a catalog reminder of hidden (deferred) tools for the user prompt.

        Lists local deferred tools by schema name, and MCP deferred tools only as
        ``server_name (N tools)`` aggregates — never the individual MCP tool names.
        Returns ``None`` when tool search is off, the reminder is disabled, or
        there is nothing hidden to advertise.
        """
        if not self._tool_search_enabled or not self._tool_search_config.inject_reminder:
            return None

        self._assert_no_duplicate_schema_names()

        local_names: list[str] = []
        server_counts: dict[str, int] = {}
        manager = self._get_mcp_manager()

        for canonical_name in self._deferred_canonical_tool_names():
            if manager is not None and canonical_name in manager.tools:
                server_name = manager.tools[canonical_name].server_name
                server_counts[server_name] = server_counts.get(server_name, 0) + 1
            else:
                local_names.append(self._tool_info_for_canonical_name(canonical_name, include_schema=False)['name'])

        if not local_names and not server_counts:
            return None

        lines = ['Additional tools are available via tool_search but are not listed above.']
        if local_names:
            lines.append('Hidden tools: ' + ', '.join(sorted(local_names)))
        if server_counts:
            servers = ', '.join(
                f'{name} ({count} tool{"s" if count != 1 else ""})' for name, count in sorted(server_counts.items())
            )
            lines.append('Available MCP servers: ' + servers)
        lines.append('Use tool_search to find a tool by keyword, then tool_call to invoke it.')

        body = '\n'.join(lines)
        return f'<system-reminder>\n{body}\n</system-reminder>'

    def describe_deferred_tool(self, tool_name: str) -> dict[str, Any]:
        resolved_tool_name = self._resolve_among(tool_name, self._deferred_canonical_tool_names())
        return self._tool_info_for_canonical_name(resolved_tool_name, include_schema=True)

    def validate_tool_call(self, tool_name: str, params: Any) -> str:
        """Validate a model-emitted tool call without executing it.

        Returns the resolved canonical tool name. Static tools are validated
        against their Pydantic input model. MCP tools are only name-resolved
        locally because their schemas are owned by the MCP server.
        """
        resolved_tool_name = self._resolve_tool_name(tool_name)
        if resolved_tool_name in self._tools:
            self._validate_resolved_tool_params(resolved_tool_name, params)
        return resolved_tool_name

    def _build_tool_context(self, *, include_executor: bool = False) -> ToolContext:
        ctx = require_context()
        tool_configs = ctx.tool_configs.copy()
        if self._tool_search_enabled:
            tool_configs['_deferred_tool_names'] = self._deferred_canonical_tool_names()
        data: dict[str, Any] = {
            'run_id': ctx.run_id,
            'tool_configs': tool_configs,
            'tool_state': ctx.tool_state.copy(),
            'trace_id': ctx.trace_id,
        }
        if include_executor:
            data['executor'] = self
        return ToolContext(**data)

    def _apply_context_updates(self, updates: dict[str, Any]) -> None:
        if updates:
            ctx = require_context()
            ctx.tool_state.update(updates)
            logger.debug('Applied context updates: %s', list(updates.keys()))

    def _validate_resolved_tool_params(self, tool_name: str, params: Any) -> BaseModel:
        tool_func = self._tools[tool_name]
        meta = get_tool_meta(tool_func)
        try:
            return meta.input_model.model_validate(params)
        except ValidationError as exc:
            schema_name = self._schema_name_for_tool(tool_name)
            raise ToolArgumentValidationError(
                schema_name=schema_name,
                canonical_name=tool_name,
                message_for_model=self._format_argument_validation_error(schema_name, tool_name, exc),
            ) from exc

    def _format_argument_validation_error(self, schema_name: str, canonical_name: str, exc: ValidationError) -> str:
        errors = exc.errors()
        formatted_errors: list[str] = []
        for error in errors[:3]:
            loc = '.'.join(str(part) for part in error.get('loc', ())) or '<root>'
            error_type = error.get('type', 'validation_error')
            message = error.get('msg', 'Invalid value')
            formatted_errors.append(f'{loc}: {message} ({error_type})')
        if len(errors) > 3:
            formatted_errors.append(f'... and {len(errors) - 3} more validation errors')

        return (
            f"Invalid arguments for tool '{schema_name}': "
            f'{"; ".join(formatted_errors)}. '
            f'{self._parameter_summary_for_tool(canonical_name)}'
        )

    def _parameter_summary_for_tool(self, canonical_name: str) -> str:
        parameters = self._build_schema_for_tool(canonical_name).get('function', {}).get('parameters', {})
        properties = parameters.get('properties', {})
        if not isinstance(properties, dict) or not properties:
            return 'Expected parameters: none.'

        required = parameters.get('required', [])
        if not isinstance(required, list):
            required = []

        required_names = [name for name in required if name in properties]
        optional_names = [name for name in properties if name not in required_names]
        return (
            'Expected parameters: '
            f'required=[{self._format_parameter_list(required_names, properties)}], '
            f'optional=[{self._format_parameter_list(optional_names, properties)}].'
        )

    @staticmethod
    def _format_parameter_list(names: list[str], properties: dict[str, Any], *, limit: int = 8) -> str:
        if not names:
            return 'none'

        labels: list[str] = []
        for name in names[:limit]:
            prop = properties.get(name, {})
            param_type = prop.get('type') if isinstance(prop, dict) else None
            labels.append(f'{name}: {param_type or "any"}')
        if len(names) > limit:
            labels.append(f'... +{len(names) - limit} more')
        return ', '.join(labels)

    async def call(self, tool_name: str, params: dict[str, Any], *, timeout: float | None = None) -> ToolResponse:
        resolved_tool_name = self._resolve_tool_name(tool_name)
        async with with_timeout(timeout):
            response = await self._call_resolved_tool(resolved_tool_name, params)
        self._apply_context_updates(response.context_updates)
        return response

    async def call_discovered_tool(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> ToolResponse:
        resolved_tool_name = self._resolve_discovered_tool_name(tool_name)

        async with with_timeout(timeout):
            response = await self._call_resolved_tool(resolved_tool_name, params)
        self._apply_context_updates(response.context_updates)
        return response

    @cacheable(tags=['tool'], ignore_args=['self'])
    async def _call_resolved_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResponse:
        if tool_name in self._tools:
            tool_func = self._tools[tool_name]
            meta = get_tool_meta(tool_func)
            parsed_params = self._validate_resolved_tool_params(tool_name, params)

            tool_context = self._build_tool_context(include_executor=self._is_bridge_tool(tool_name))
            if meta.tool_type == ToolType.LOCAL:
                return await self._call_local(tool_func, parsed_params, tool_context)
            return await self._call_runtime(tool_func, parsed_params, tool_context)

        manager = self._get_mcp_manager()
        if manager is None or tool_name not in manager.tools:
            raise KeyError(f"Tool '{tool_name}' not found. Available: {sorted(self._canonical_tool_name_set())}")

        return await manager.call_tool(tool_name, params)

    async def parallel_call(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResponse]:
        if not calls:
            return []

        groups: dict[int | None, list[tuple[int, str, dict[str, Any]]]] = {}
        for idx, (tool_name, params) in enumerate(calls):
            resolved_name = self._resolve_tool_name(tool_name)
            if resolved_name in self._tools:
                meta = get_tool_meta(self._tools[resolved_name])
                group_key = id(meta.parallel_func) if meta.parallel_func is not None else None
            else:
                group_key = None
            groups.setdefault(group_key, []).append((idx, resolved_name, params))

        results: list[tuple[int, ToolResponse]] = []

        async def _run_batch(group_key: int | None, items: list[tuple[int, str, dict[str, Any]]]) -> None:
            if group_key is None:

                async def _single(idx: int, name: str, params: dict[str, Any]) -> None:
                    response = await self._call_resolved_tool(name, params)
                    self._apply_context_updates(response.context_updates)
                    results.append((idx, response))

                await asyncio.gather(*[_single(i, n, p) for i, n, p in items])
                return

            tool_context = self._build_tool_context()
            batch_args: list[tuple[str, BaseModel, ToolContext]] = []
            for _idx, name, params in items:
                parsed = self._validate_resolved_tool_params(name, params)
                batch_args.append((name, parsed, tool_context))

            parallel_func = get_tool_meta(self._tools[items[0][1]]).parallel_func
            assert parallel_func is not None
            responses = await parallel_func(batch_args)

            for (orig_idx, _name, _params), response in zip(items, responses, strict=False):
                self._apply_context_updates(response.context_updates)
                results.append((orig_idx, response))

        await asyncio.gather(*[_run_batch(key, items) for key, items in groups.items()])
        results.sort(key=lambda item: item[0])
        return [response for _, response in results]

    async def _call_local(self, tool_func: ToolFunc, params: BaseModel, context: ToolContext) -> ToolResponse:
        logger.debug('Calling local tool: %s', get_tool_meta(tool_func).name)
        return await tool_func(params, context)

    async def _call_runtime(self, tool_func: ToolFunc, params: BaseModel, context: ToolContext) -> ToolResponse:
        meta = get_tool_meta(tool_func)
        tool_id = tool_func.nexus_meta['name']

        try:
            entry = tool_registry.get_entry(tool_id)
        except EntryNotFoundError as exc:
            raise RuntimeError(f"Tool '{meta.name}' is not present in tool_registry") from exc

        import_path = entry.import_path
        if import_path is None:
            raise RuntimeError(f"Tool '{meta.name}' has no import_path in registry")

        logger.debug('Calling runtime tool: %s (%s at %s)', meta.name, tool_id, import_path)

        ctx = require_context()
        runtime = await ctx.get_runtime()
        return await runtime.call_tool(import_path, params.model_dump(), context)


__all__ = ['ToolArgumentValidationError', 'ToolExecutor', 'ToolNameResolutionError']
