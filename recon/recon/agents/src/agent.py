"""Agent definition and validation for nexus framework.

Agents are async functions that implement reasoning loops using LLMs and tools.
They receive input via AgentRunInput and access runtime dependencies through ExecutionContext.
"""

import inspect

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, get_origin, get_type_hints

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core.registry import ProcessorResult, Registry
from nexus.core.tool.types import ToolSearchConfig
from nexus.trace import observe


class AgentRunInput(BaseModel):
    """Base input for agent runs.

    Agents access runtime dependencies (runtime, openai_client) via ExecutionContext,
    not through input fields. Input should only contain business-specific parameters.

    Example:
        class MyAgentInput(AgentRunInput):
            user_prompt: str
            max_iterations: int = 10

        @agent_registry.register('my_agent')
        async def my_agent(input: MyAgentInput) -> MyAgentOutput:
            ctx = require_context()
            client = ctx.openai_client
            runtime = await ctx.get_runtime()
            ...
    """

    model_config = {'arbitrary_types_allowed': True}


class UserInputRequest(BaseModel):
    """Structured request passed to interactive user input providers."""

    source: Literal['turn', 'hook'] = Field(
        default='turn',
        description='Why user input is requested: after a normal turn, or because an agent hook needs feedback.',
    )
    hint: str | None = Field(
        default=None,
        description='Short human-readable summary for simple input providers that do not inspect structured data.',
    )
    session_id: str | None = Field(
        default=None,
        description='Stable session identifier for correlating the input with an interactive agent conversation.',
    )
    message: dict[str, Any] | None = Field(
        default=None,
        description='Primary raw agent event that triggered the input request, when available.',
    )
    events: list[dict[str, Any]] = Field(
        default_factory=list,
        description='Raw events emitted by the agent during the completed turn before requesting user input.',
    )
    hook: dict[str, Any] | None = Field(
        default=None,
        description='Structured hook or tool payload when source is hook; absent for normal turn continuation.',
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description='Nexus-owned context for the input request, such as turn number or caller-specific metadata.',
    )


class WithInputProvider(BaseModel):
    """Mixin for agents that supports interactive user input."""

    get_user_input: Callable[[UserInputRequest | str | None], Awaitable[str | None]]


class WithUserPrompt(BaseModel):
    """Mixin for agents that receive a user prompt."""

    user_prompt: str | list[ChatCompletionContentPartParam]


class WithHistory(BaseModel):
    """Mixin for agents that accept a prior conversation history.

    The agent prepends ``history`` between its ``system_prompt`` and the new
    ``user_prompt`` for the current turn. Conventions:

    * Roles are limited to ``user`` / ``assistant`` / ``tool``. ``system`` is
      forbidden because agents own their system message via ``system_prompt``.
    * History must be ready for a new user turn: the last message must be
      ``assistant`` or ``tool``. A trailing ``user`` message would conflict with
      the new ``user_prompt`` for this turn.
    * ``None`` (the default) means "no prior history"; an empty list is treated
      the same way.
    """

    history: list[dict[str, Any]] | None = None


_HISTORY_ALLOWED_ROLES = {'user', 'assistant', 'tool'}
_HISTORY_TRAILING_ROLES = {'assistant', 'tool'}


def validate_history(history: list[dict[str, Any]] | None) -> None:
    """Validate WithHistory input.

    See WithHistory for the contract: no system messages, must end with an
    assistant or tool message so user_prompt can start the next turn.
    """
    if not history:
        return
    for idx, msg in enumerate(history):
        role = msg.get('role')
        if role not in _HISTORY_ALLOWED_ROLES:
            raise ValueError(
                f'history[{idx}]: role must be one of {sorted(_HISTORY_ALLOWED_ROLES)}, got {role!r}. '
                'Use system_prompt for the system message.'
            )
    last_role = history[-1].get('role')
    if last_role not in _HISTORY_TRAILING_ROLES:
        raise ValueError(
            f'history must end with an assistant or tool message (got {last_role!r}); '
            'a trailing user message conflicts with user_prompt for the new turn.'
        )


class WithToolSearch(BaseModel):
    """Mixin for agents that support client-side tool search."""

    tool_search: ToolSearchConfig | None = None


class WithWorkingDir(BaseModel):
    """Mixin for agents that need a working directory inside the runtime.

    The task that orchestrates the agent is responsible for passing the
    correct value through ``run_agent(working_dir=...)``. Omitting it is a
    configuration error.
    """

    working_dir: str


class AgentRunOutput(BaseModel):
    """Base output for agent runs.

    Output should contain business results that the caller needs.
    """

    model_config = {'arbitrary_types_allowed': True}


class WithMainTrajectory(BaseModel):
    """Mixin for agents that produce a conversation trajectory for evaluation."""

    main_trajectory: list[dict]


class WithFinalAnswer(BaseModel):
    """Mixin for agents that produce a final answer string."""

    final_answer: str | None = None


class WithMainTmuxSessionId(BaseModel):
    """Mixin for agents perform vital actions in a tmux session."""

    main_tmux_session_id: str


class AgentFunc(Protocol):
    """Protocol for agent functions."""

    async def __call__(self, agent_input: AgentRunInput) -> AgentRunOutput: ...


def validate_agent(obj: Any, name: str, meta: dict) -> ProcessorResult:
    if not callable(obj):
        raise TypeError(f'Agent must be callable, got {type(obj)}')

    if not inspect.iscoroutinefunction(obj):
        raise TypeError(f'Agent must be an async function, got {type(obj)}')

    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    type_hints = get_type_hints(obj)

    # validate input
    if len(params) != 1:
        raise TypeError(f'Agent must have exactly one input parameter, got {len(params)}')

    param = params[0]
    if param.annotation is inspect.Parameter.empty:
        raise TypeError('Agent input parameter must have a type annotation')

    param_type = type_hints.get(param.name, param.annotation)
    origin = get_origin(param_type)
    if origin is not None:
        param_type = origin
    if not (inspect.isclass(param_type) and issubclass(param_type, AgentRunInput)):
        raise TypeError(f'Agent input parameter must be a subclass of AgentRunInput, got {param_type}')

    # validate output
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError('Agent must have a return type annotation')

    return_type = type_hints.get('return', sig.return_annotation)
    origin = get_origin(return_type)
    if origin is not None:
        return_type = origin
    if not (inspect.isclass(return_type) and issubclass(return_type, AgentRunOutput)):
        raise TypeError(f'Agent return type must be a subclass of AgentRunOutput, got {return_type}')

    if not name:
        raise TypeError('Agent registration name must be non-empty')
    _span_name = f'agent.{name.replace("/", ".")}'
    wrapped_obj = observe(name=_span_name, attributes={'type': 'agent', 'name': name})(obj)
    return ProcessorResult(wrapped_obj=wrapped_obj, meta=meta)


agent_registry = Registry[AgentFunc]('agent', validate_agent)
