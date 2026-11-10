from __future__ import annotations

import logging
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from packaging.requirements import InvalidRequirement, Requirement
from pydantic import BaseModel, ConfigDict, model_validator

from nexus.core.module_meta import ModuleMeta


logger = logging.getLogger(__name__)


ModuleObjectType = Literal[
    'agents',
    'openai_clients',
    'runtime_providers',
    'span_processors',
    'trajectory_observers',
    'tasks',
]
_IMPORT_PATH = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$')
_UNSAFE_REQUIREMENT_PREFIXES = ('-e ', '--editable ', '.', '/', '~')


class ModuleDependencyError(ValueError):
    def __init__(self, code: str, *, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f'{code}: {detail}')


class ModuleSelector(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: Literal['object', 'import']
    value: str
    object_type: ModuleObjectType | None = None

    @model_validator(mode='after')
    def validate_shape(self) -> ModuleSelector:
        if not self.value:
            raise ValueError('module selector value must not be empty')
        if (self.kind == 'object') != (self.object_type is not None):
            raise ValueError('object selectors require object_type and import selectors forbid it')
        return self


@dataclass(frozen=True, slots=True)
class ResolvedHostModuleDependencies:
    modules: tuple[ModuleMeta, ...]
    requirements: tuple[str, ...]


def selectors_from_run_input(run_input: Mapping[str, Any]) -> tuple[ModuleSelector, ...]:
    selectors: set[ModuleSelector] = set()
    _add_object_selector(selectors, 'tasks', run_input.get('task_name'))
    _add_object_selector(selectors, 'agents', run_input.get('agent_name'))
    _add_object_selector(selectors, 'runtime_providers', run_input.get('runtime_provider_name'))

    openai_clients = run_input.get('openai_clients') or {}
    if isinstance(openai_clients, Mapping):
        for config in openai_clients.values():
            if isinstance(config, Mapping):
                _add_object_selector(selectors, 'openai_clients', config.get('name'))

    processors = run_input.get('processors') or {}
    if isinstance(processors, Mapping):
        for config in processors.values():
            if isinstance(config, Mapping):
                _add_object_selector(selectors, 'span_processors', config.get('name'))

    gateway_trajectory_observers = run_input.get('gateway_trajectory_observers') or {}
    if isinstance(gateway_trajectory_observers, Mapping):
        for config in gateway_trajectory_observers.values():
            if isinstance(config, Mapping):
                _add_object_selector(selectors, 'trajectory_observers', config.get('name'))

    imports = run_input.get('imports') or []
    if not isinstance(imports, list):
        raise ModuleDependencyError('module_import_invalid', detail=str(imports))
    for value in imports:
        if not isinstance(value, str) or _IMPORT_PATH.fullmatch(value) is None:
            raise ModuleDependencyError('module_import_invalid', detail=str(value))
        selectors.add(ModuleSelector(kind='import', value=value))

    return tuple(sorted(selectors, key=lambda item: (item.kind, item.object_type or '', item.value)))


def resolve_host_module_dependencies(
    selectors: Sequence[ModuleSelector],
    modules: Sequence[ModuleMeta],
) -> ResolvedHostModuleDependencies:
    object_index: dict[tuple[str, str], dict[str, ModuleMeta]] = {}
    for module in modules:
        for object_type, entries in module.objects.items():
            for object_name in entries:
                key = (object_type, object_name)
                object_index.setdefault(key, {})[module.module_path] = module

    selected: dict[str, ModuleMeta] = {}
    import_candidates = sorted(modules, key=lambda item: (-len(item.module_path), item.module_path))
    for selector in selectors:
        if selector.kind == 'object':
            assert selector.object_type is not None
            matches = object_index.get((selector.object_type, selector.value), {})
            if not matches:
                logger.warning(
                    'module_dependency_selector_unresolved: %s:%s',
                    selector.object_type,
                    selector.value,
                )
                continue
            if len(matches) > 1:
                raise ModuleDependencyError(
                    'module_object_ambiguous',
                    detail=f'{selector.object_type}:{selector.value}',
                )
            module = next(iter(matches.values()))
            selected[module.module_path] = module
            continue

        module = next(
            (
                candidate
                for candidate in import_candidates
                if selector.value == candidate.module_path or selector.value.startswith(candidate.module_path + '.')
            ),
            None,
        )
        if module is not None:
            selected[module.module_path] = module
        else:
            logger.warning('module_dependency_selector_unresolved: imports:%s', selector.value)

    selected_modules = tuple(selected[key] for key in sorted(selected))
    requirements: list[str] = []
    seen_requirements: set[str] = set()
    for module in selected_modules:
        for raw_requirement in module.host_deps:
            requirement = _validate_host_requirement(raw_requirement)
            if requirement not in seen_requirements:
                seen_requirements.add(requirement)
                requirements.append(requirement)
    return ResolvedHostModuleDependencies(
        modules=selected_modules,
        requirements=tuple(requirements),
    )


def _add_object_selector(
    selectors: set[ModuleSelector],
    object_type: ModuleObjectType,
    value: Any,
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ModuleDependencyError('module_object_not_declared', detail=f'{object_type}:{value}')
    selectors.add(ModuleSelector(kind='object', object_type=object_type, value=value))


def _validate_host_requirement(value: str) -> str:
    raw = value.strip()
    lowered = raw.lower()
    if (
        not raw
        or lowered.startswith(_UNSAFE_REQUIREMENT_PREFIXES)
        or '://' in raw
        or '@ file:' in lowered
        or '@file:' in lowered
    ):
        raise ModuleDependencyError('module_requirement_unsafe', detail=value)
    try:
        requirement = Requirement(raw)
    except InvalidRequirement as exc:
        raise ModuleDependencyError('module_requirement_invalid', detail=value) from exc
    if requirement.url is not None:
        raise ModuleDependencyError('module_requirement_unsafe', detail=value)
    return raw


__all__ = [
    'ModuleDependencyError',
    'ModuleSelector',
    'ResolvedHostModuleDependencies',
    'resolve_host_module_dependencies',
    'selectors_from_run_input',
]
