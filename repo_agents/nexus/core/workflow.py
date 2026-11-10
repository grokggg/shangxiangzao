"""Resumable graph execution primitives for tasks and agents."""

from __future__ import annotations

import asyncio
import copy
import inspect

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, TypeVar, overload

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from nexus.trace.phases import phase
from nexus.utils.timeout import with_timeout


InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class WorkflowError(RuntimeError):
    """Base class for workflow execution errors."""


class WorkflowStateMismatchError(WorkflowError):
    """Raised when supplied state does not match the workflow definition."""


class WorkflowLimitExceededError(WorkflowError):
    """Raised when a workflow or node visit limit is exhausted."""


class WorkflowNodeTimeoutError(WorkflowError, TimeoutError):
    """Raised when a node exceeds its configured timeout."""


class NodeResult(BaseModel, Generic[OutputT]):
    """A node's business output and the next node selected by it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: OutputT
    next: str | None = None


class NodeVisit(BaseModel):
    """One successfully committed node visit."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1)
    node: str
    visit: int = Field(ge=1)
    output: JsonValue
    next_node: str | None = None


class WorkflowState(BaseModel):
    """Serializable progress for one workflow execution scope."""

    model_config = ConfigDict(frozen=True)

    workflow: str
    version: str
    cursor: str | None
    history: tuple[NodeVisit, ...] = ()

    def visits(self, node: str) -> tuple[NodeVisit, ...]:
        """Return all committed visits for a node."""
        return tuple(visit for visit in self.history if visit.node == node)

    def latest_visit(self, node: str) -> NodeVisit:
        """Return the latest committed visit for a node."""
        for visit in reversed(self.history):
            if visit.node == node:
                return visit
        raise LookupError(f'Node {node!r} has no committed output')

    def last_visit(self) -> NodeVisit:
        """Return the workflow's most recently committed visit."""
        if not self.history:
            raise LookupError('Workflow has no committed output')
        return self.history[-1]

    @overload
    def outputs(self, node: str) -> list[JsonValue]: ...

    @overload
    def outputs(self, node: str, *, as_type: type[OutputT]) -> list[OutputT]: ...

    def outputs(self, node: str, *, as_type: type[OutputT] | None = None) -> list[JsonValue] | list[OutputT]:
        """Return all outputs for a node, optionally validating their type."""
        outputs = [copy.deepcopy(visit.output) for visit in self.visits(node)]
        if as_type is None:
            return outputs
        adapter = TypeAdapter(as_type)
        return [adapter.validate_python(output) for output in outputs]

    @overload
    def output(self, node: str) -> JsonValue: ...

    @overload
    def output(self, node: str, *, as_type: type[OutputT]) -> OutputT: ...

    def output(self, node: str, *, as_type: type[OutputT] | None = None) -> JsonValue | OutputT:
        """Return a node's latest output, optionally validating its type."""
        output = copy.deepcopy(self.latest_visit(node).output)
        if as_type is None:
            return output
        return TypeAdapter(as_type).validate_python(output)

    @overload
    def last_output(self) -> JsonValue: ...

    @overload
    def last_output(self, *, as_type: type[OutputT]) -> OutputT: ...

    def last_output(self, *, as_type: type[OutputT] | None = None) -> JsonValue | OutputT:
        """Return the workflow's last output, optionally validating its type."""
        output = copy.deepcopy(self.last_visit().output)
        if as_type is None:
            return output
        return TypeAdapter(as_type).validate_python(output)


class WorkflowNodeOptions(BaseModel):
    """Per-run overrides for one node."""

    timeout: float | None = Field(default=None, gt=0)
    max_visits: int | None = Field(default=None, ge=1)


class WorkflowOptions(BaseModel):
    """Task- or agent-owned workflow execution overrides."""

    max_steps: int | None = Field(default=None, ge=1)
    nodes: dict[str, WorkflowNodeOptions] = Field(default_factory=dict)


class WithWorkflowOptions(BaseModel):
    """Mixin for task or agent inputs that expose workflow configuration."""

    workflow: WorkflowOptions = Field(default_factory=WorkflowOptions)


@dataclass(frozen=True)
class Node(Generic[InputT, OutputT]):
    """A named async callable in a workflow graph."""

    name: str
    func: Callable[[InputT], Awaitable[NodeResult[OutputT]]]
    timeout: float | None = None
    max_visits: int | None = None
    span_name: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('Node name must be non-empty')
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError('Node timeout must be greater than zero')
        if self.max_visits is not None and self.max_visits < 1:
            raise ValueError('Node max_visits must be at least one')


@dataclass(frozen=True)
class Workflow(Generic[InputT]):
    """A resumable directed graph whose nodes choose their successor."""

    name: str
    entry: str
    nodes: Sequence[Node[InputT, Any]]
    version: str = '1'
    namespace: str = 'task'
    max_steps: int = 100
    _nodes_by_name: Mapping[str, Node[InputT, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('Workflow name must be non-empty')
        if not self.version:
            raise ValueError('Workflow version must be non-empty')
        if not self.namespace:
            raise ValueError('Workflow namespace must be non-empty')
        if self.max_steps < 1:
            raise ValueError('Workflow max_steps must be at least one')

        nodes = tuple(self.nodes)
        nodes_by_name = {node.name: node for node in nodes}
        if len(nodes_by_name) != len(nodes):
            raise ValueError('Workflow node names must be unique')
        if self.entry not in nodes_by_name:
            raise ValueError(f'Workflow entry {self.entry!r} is not a defined node')

        object.__setattr__(self, 'nodes', nodes)
        object.__setattr__(self, '_nodes_by_name', MappingProxyType(nodes_by_name))

    def get_node(self, name: str) -> Node[InputT, Any]:
        try:
            return self._nodes_by_name[name]
        except KeyError as exc:
            raise WorkflowError(f'Workflow {self.name!r} has no node {name!r}') from exc

    def has_node(self, name: str) -> bool:
        return name in self._nodes_by_name


@dataclass(frozen=True)
class _ActiveWorkflowNode:
    scope: str
    node: str
    visit: int
    sequence: int


_active_workflow_node: ContextVar[_ActiveWorkflowNode | None] = ContextVar('active_workflow_node', default=None)


def _resolve_scope(workflow: Workflow, scope: str | None) -> str:
    if scope is not None:
        if not scope:
            raise ValueError('Workflow scope must be non-empty')
        return scope

    own_scope = f'{workflow.namespace}/{workflow.name}'
    parent = _active_workflow_node.get()
    if parent is None:
        return own_scope
    return f'{parent.scope}/{parent.node}#{parent.visit}/{own_scope}'


def _serialize_output(output: Any) -> JsonValue:
    if isinstance(output, BaseModel):
        output = output.model_dump(mode='json')
    return TypeAdapter(JsonValue).validate_python(copy.deepcopy(output))


def _validate_options(workflow: Workflow, options: WorkflowOptions) -> None:
    unknown_nodes = set(options.nodes) - set(workflow._nodes_by_name)
    if unknown_nodes:
        names = ', '.join(sorted(unknown_nodes))
        raise WorkflowError(f'Workflow {workflow.name!r} has options for unknown nodes: {names}')


async def _run_node(
    node: Node[InputT, Any],
    workflow_input: InputT,
    timeout: float | None,
    *,
    scope: str,
) -> NodeResult[Any]:
    try:
        async with with_timeout(timeout):
            pending = node.func(workflow_input)
            if not inspect.isawaitable(pending):
                raise TypeError(f'Workflow node {node.name!r} must return an awaitable')
            result = await pending
    except (TimeoutError, asyncio.TimeoutError) as exc:
        if timeout is None:
            raise
        raise WorkflowNodeTimeoutError(
            f'Workflow node {node.name!r} in {scope!r} timed out after {timeout} seconds'
        ) from exc

    if not isinstance(result, NodeResult):
        raise TypeError(f'Workflow node {node.name!r} must return NodeResult, got {type(result).__name__}')
    return result


async def execute_workflow(
    workflow: Workflow[InputT],
    workflow_input: InputT,
    *,
    options: WorkflowOptions | None = None,
    scope: str | None = None,
) -> WorkflowState:
    """Run all remaining nodes with one shared input and return committed state."""
    from nexus.core.context import require_context

    execution_options = options or WorkflowOptions()
    _validate_options(workflow, execution_options)
    resolved_scope = _resolve_scope(workflow, scope)
    ctx = require_context()

    state = ctx.workflow_states.get(resolved_scope)
    if state is None:
        state = WorkflowState(workflow=workflow.name, version=workflow.version, cursor=workflow.entry)
        ctx.workflow_states[resolved_scope] = state
    elif state.workflow != workflow.name or state.version != workflow.version:
        raise WorkflowStateMismatchError(
            f'Workflow state at {resolved_scope!r} is for {state.workflow!r}@{state.version}, '
            f'expected {workflow.name!r}@{workflow.version}'
        )

    if state.cursor is not None and not workflow.has_node(state.cursor):
        raise WorkflowStateMismatchError(
            f'Workflow state at {resolved_scope!r} points to unknown node {state.cursor!r}'
        )

    max_steps = execution_options.max_steps or workflow.max_steps
    while state.cursor is not None:
        if len(state.history) >= max_steps:
            raise WorkflowLimitExceededError(
                f'Workflow {workflow.name!r} exceeded max_steps={max_steps} at node {state.cursor!r}'
            )

        node = workflow.get_node(state.cursor)
        node_options = execution_options.nodes.get(node.name)
        timeout = node_options.timeout if node_options and node_options.timeout is not None else node.timeout
        max_visits = (
            node_options.max_visits if node_options and node_options.max_visits is not None else node.max_visits
        )
        visit = len(state.visits(node.name)) + 1
        if max_visits is not None and visit > max_visits:
            raise WorkflowLimitExceededError(
                f'Workflow node {node.name!r} exceeded max_visits={max_visits} in {resolved_scope!r}'
            )

        sequence = len(state.history) + 1
        active = _ActiveWorkflowNode(
            scope=resolved_scope,
            node=node.name,
            visit=visit,
            sequence=sequence,
        )
        token = _active_workflow_node.set(active)
        span_name = node.span_name or f'{workflow.namespace}.{node.name}'
        try:
            with phase(
                span_name,
                attributes={
                    'workflow.name': workflow.name,
                    'workflow.version': workflow.version,
                    'workflow.scope': resolved_scope,
                    'node.name': node.name,
                    'node.visit': visit,
                    'node.sequence': sequence,
                },
                close_previous=False,
            ) as node_span:
                result = await _run_node(node, workflow_input, timeout, scope=resolved_scope)
                if result.next is not None and not workflow.has_node(result.next):
                    raise WorkflowError(f'Workflow node {node.name!r} selected unknown next node {result.next!r}')

                committed_output = _serialize_output(result.output)
                committed_visit = NodeVisit(
                    sequence=sequence,
                    node=node.name,
                    visit=visit,
                    output=committed_output,
                    next_node=result.next,
                )
                state = WorkflowState(
                    workflow=state.workflow,
                    version=state.version,
                    cursor=result.next,
                    history=(*state.history, committed_visit),
                )
                ctx.workflow_states[resolved_scope] = state
                node_span.set_attributes({
                    'node.next': result.next or '',
                    'workflow.completed': result.next is None,
                })
        finally:
            _active_workflow_node.reset(token)

    return state


__all__ = [
    'Node',
    'NodeResult',
    'NodeVisit',
    'WithWorkflowOptions',
    'Workflow',
    'WorkflowError',
    'WorkflowLimitExceededError',
    'WorkflowNodeOptions',
    'WorkflowNodeTimeoutError',
    'WorkflowOptions',
    'WorkflowState',
    'WorkflowStateMismatchError',
    'execute_workflow',
]
