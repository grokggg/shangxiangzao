"""Task definition and validation for nexus framework.

Tasks are async functions that orchestrate agent execution for specific use cases.
They receive business parameters through TaskRunInput and access runtime dependencies
via ExecutionContext. Results are returned via TaskRunOutput.
"""

import inspect

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, get_origin, get_type_hints

from pydantic import BaseModel

from nexus.core.agent import AgentRunOutput
from nexus.core.registry import ProcessorResult, Registry
from nexus.trace import observe


# ============== Task Input/Output Base Classes ==============


class TaskRunInput(BaseModel):
    """Base input for tasks.

    Tasks access runtime dependencies via ExecutionContext, not through input fields.
    Input should only contain business-specific parameters.

    Example:
        class SWETaskInput(TaskRunInput):
            instance_id: str
            problem_statement: str

        @task_registry.register('swe_bench')
        async def swe_task(input: SWETaskInput) -> SWETaskOutput:
            ctx = require_context()

            # Run agent phase
            agent_output = await run_agent(input.problem_statement)

            # Evaluation phase with different runtime
            runtime = await ctx.get_runtime()
            patch = await generate_patch(runtime, agent_output)
            eval_runtime = await ctx.get_runtime('eval')
            score = await evaluate(eval_runtime, patch, ...)

            return SWETaskOutput(score=score)
    """

    model_config = {'arbitrary_types_allowed': True}


class TaskRunOutput(BaseModel):
    """Base output for tasks.

    Output should contain business results that the caller needs.
    """

    model_config = {'arbitrary_types_allowed': True}


# ============== Input Mixins ==============


class WithAgentRunner(BaseModel):
    """Mixin for tasks that need to run an agent.

    The run_agent function is pre-configured by the runner with base settings
    (agent config, etc.). Tasks only need to pass task-specific
    parameters like prompt.

    Example:
        async def my_task(input: MyTaskRunInput) -> MyTaskRunOutput:
            output = await input.run_agent(
                runtime=input.runtime,
                prompt="Do something...",
                workdir="/workspace",
            )
    """

    run_agent: Callable[..., Awaitable[AgentRunOutput]]


# ============== Output Mixins ==============


class WithScore(BaseModel):
    """Mixin for tasks that produce a float score."""

    score: float


# ============== Task Protocol ==============


class TaskFunc(Protocol):
    """Protocol for task functions.

    Tasks must be async functions with exactly one parameter (TaskRunInput subclass)
    and return a TaskRunOutput subclass.
    """

    async def __call__(self, task_input: TaskRunInput) -> TaskRunOutput: ...


# ============== Validation ==============


def validate_task(obj: Any, name: str, meta: dict) -> ProcessorResult:
    """Validate that an object conforms to the TaskFunc protocol and wrap with observe.

    Checks:
    - Must be callable
    - Must be an async function
    - Must have exactly one parameter
    - Parameter must be annotated as TaskRunInput subclass
    - Return type must be annotated as TaskRunOutput subclass

    Returns:
        ProcessorResult with wrapped_obj (observe-wrapped task) and meta
    """
    if not callable(obj):
        raise TypeError(f'Task must be callable, got {type(obj)}')

    if not inspect.iscoroutinefunction(obj):
        raise TypeError(f'Task must be an async function, got {type(obj)}')

    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    type_hints = get_type_hints(obj)

    # Validate input parameter
    if len(params) != 1:
        raise TypeError(f'Task must have exactly one input parameter, got {len(params)}')

    param = params[0]
    if param.annotation is inspect.Parameter.empty:
        raise TypeError('Task input parameter must have a type annotation')

    param_type = type_hints.get(param.name, param.annotation)
    origin = get_origin(param_type)
    if origin is not None:
        param_type = origin
    if not (inspect.isclass(param_type) and issubclass(param_type, TaskRunInput)):
        raise TypeError(f'Task input parameter must be a subclass of TaskRunInput, got {param_type}')

    # Validate return type
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError('Task must have a return type annotation')

    return_type = type_hints.get('return', sig.return_annotation)
    origin = get_origin(return_type)
    if origin is not None:
        return_type = origin
    if not (inspect.isclass(return_type) and issubclass(return_type, TaskRunOutput)):
        raise TypeError(f'Task return type must be a subclass of TaskRunOutput, got {return_type}')

    # Wrap with observe decorator for tracing
    _span_name = f'task.{name.replace("/", ".")}'
    wrapped_obj = observe(name=_span_name, attributes={'type': 'task', 'name': name})(obj)

    return ProcessorResult(wrapped_obj=wrapped_obj, meta=meta)


# ============== Registry ==============


task_registry = Registry[TaskFunc]('task', validate_task)
