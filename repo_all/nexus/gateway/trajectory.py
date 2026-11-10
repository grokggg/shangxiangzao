"""Picklable trajectory independent of scaffold, model, and policy.

Gateway runtime writes topology and token attribution directly; this module
also exposes snapshot queries, serialization, and training segments.
"""

from __future__ import annotations

import base64 as _base64
import binascii as _binascii
import copy as _copy
import math as _math

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    import torch


SCHEMA_VERSION = 'gateway-trajectory-v1'

NodeId = int
MessageId = int
ChainId = str

MAIN_CHAIN_ID: ChainId = 'main'

EFFECTIVE_TRAIN_MASK = 'effective_train_mask'
OLD_LOG_PROBS = 'old_log_probs'
ADVANTAGES = 'advantages'

_SEGMENT_BUILTIN_TENSORS = frozenset({'response_ids', 'response_mask', 'loss_mask', 'rollout_log_probs'})
_SEGMENT_SCOPED_NODE_TENSORS = frozenset({'old_log_probs', 'ref_log_prob', 'values', 'advantages', 'returns'})
_SEGMENT_OWNED_TENSORS = _SEGMENT_BUILTIN_TENSORS | _SEGMENT_SCOPED_NODE_TENSORS
_NODE_OWNED_TENSORS = frozenset({EFFECTIVE_TRAIN_MASK})

_TYPE_TAG = '__nexus_gateway_type__'

_DROP = object()  # sentinel: a value that must be omitted from the JSON-safe chat artifact


def _validate_node_training_tensor_name(node_id: NodeId, name: str) -> None:
    if name in _SEGMENT_OWNED_TENSORS:
        raise ValueError(f'node {node_id} training tensor {name!r} belongs in segment training data')


def _validate_segment_training_tensor_name(name: str, *, reserved_error: type[Exception]) -> None:
    if name in _NODE_OWNED_TENSORS:
        raise ValueError(f'{name} belongs on TrajectoryNode')
    if name in _SEGMENT_BUILTIN_TENSORS:
        raise reserved_error(f'segment training tensor name {name!r} is reserved')


def _json_safe(value: Any, path: str = 'chat_completion') -> Any:
    """Recursively drop values that ``json.dumps`` can't serialize from a chat-artifact payload.

    The chat_completions JSONL is a human/debug-readable observability artifact. Structural
    training tensors — R3 ``routed_experts`` and per-token ``logprobs``/``topk`` numpy arrays —
    belong in the token/segment lane (TokenAccount → to_legacy_hytuner), NOT here; the legacy
    HyTunerClient's chat dump never carried them. A raw ``numpy.ndarray`` reaching ``json.dumps``
    raises (``Object of type ndarray is not JSON serializable``), which previously killed every
    R3 nexus trajectory on the gateway path. Rather than encode these into the artifact, we drop
    them (dict keys) / omit them (list items) so the artifact stays JSON-safe and legacy-shaped.
    Plain JSON scalars/containers pass through unchanged; numpy scalars degrade to Python numbers.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    value_type = type(value)
    if value_type.__module__.startswith('numpy'):
        # numpy scalar -> python number (JSON-safe); numpy ndarray -> sentinel for the caller to drop.
        item = getattr(value, 'item', None)
        if callable(item) and getattr(value, 'ndim', 1) == 0:
            return value.item()
        return _DROP
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            safe = _json_safe(item, f'{path}.{key}')
            if safe is not _DROP:
                out[key] = safe
        return out
    if isinstance(value, (list, tuple)):
        return [safe for safe in (_json_safe(item, f'{path}[]') for item in value) if safe is not _DROP]
    # Unknown/opaque object (not a JSON primitive) — drop it rather than crash the artifact dump.
    return _DROP


# For Json serialization safety.
def _encode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise TypeError(f'{path}: non-finite floats are not supported')
        return value
    if isinstance(value, bytes):
        return {_TYPE_TAG: 'bytes', 'data': _base64.b64encode(value).decode('ascii')}
    if isinstance(value, tuple):
        return {
            _TYPE_TAG: 'tuple',
            'items': [_encode_value(item, f'{path}[{index}]') for index, item in enumerate(value)],
        }
    if isinstance(value, list):
        return [_encode_value(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f'{path}: dictionary key {key!r} is not a string')
        items = {key: _encode_value(value[key], f'{path}.{key}') for key in sorted(value)}
        if _TYPE_TAG in items:
            return {_TYPE_TAG: 'dict', 'items': [[key, item] for key, item in items.items()]}
        return items
    value_type = type(value)
    if value_type.__module__.startswith('numpy') and value_type.__name__ == 'ndarray':
        if getattr(value.dtype, 'hasobject', False):
            raise TypeError(f'{path}: object-dtype ndarray is not supported')
        return {
            _TYPE_TAG: 'ndarray',
            'data': _base64.b64encode(value.tobytes(order='C')).decode('ascii'),
            'dtype': value.dtype.str,
            'shape': list(value.shape),
        }
    raise TypeError(f'{path}: unsupported value type {value_type.__module__}.{value_type.__qualname__}')


# For Json deserialization safety.
def _decode_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not _math.isfinite(value):
            raise ValueError(f'{path}: non-finite floats are not supported')
        return value
    if isinstance(value, list):
        return [_decode_value(item, f'{path}[{index}]') for index, item in enumerate(value)]
    if not isinstance(value, dict):
        raise TypeError(f'{path}: invalid decoded value type {type(value).__name__}')
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f'{path}: decoded dictionary keys must be strings')
    tag = value.get(_TYPE_TAG)
    if tag is None:
        return {key: _decode_value(item, f'{path}.{key}') for key, item in value.items()}
    if tag == 'bytes' and set(value) == {_TYPE_TAG, 'data'}:
        try:
            return _base64.b64decode(value['data'], validate=True)
        except (TypeError, ValueError, _binascii.Error) as exc:
            raise ValueError(f'{path}: invalid bytes payload') from exc
    if tag == 'tuple' and set(value) == {_TYPE_TAG, 'items'}:
        if not isinstance(value['items'], list):
            raise ValueError(f'{path}: invalid tuple payload')
        return tuple(_decode_value(item, f'{path}[{index}]') for index, item in enumerate(value['items']))
    if tag == 'dict' and set(value) == {_TYPE_TAG, 'items'}:
        if not isinstance(value['items'], list):
            raise ValueError(f'{path}: invalid dictionary payload')
        decoded: dict[str, Any] = {}
        for index, pair in enumerate(value['items']):
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise ValueError(f'{path}.items[{index}]: invalid dictionary entry')
            key, item = pair
            if key in decoded:
                raise ValueError(f'{path}: duplicate dictionary key {key!r}')
            decoded[key] = _decode_value(item, f'{path}.{key}')
        return decoded
    if tag == 'ndarray' and set(value) == {_TYPE_TAG, 'data', 'dtype', 'shape'}:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on the consumer environment
            raise RuntimeError(f'{path}: decoding ndarray requires numpy') from exc
        try:
            dtype = np.dtype(value['dtype'])
            if dtype.hasobject:
                raise ValueError('object dtype')
            shape = value['shape']
            if not isinstance(shape, list) or any(not isinstance(size, int) or size < 0 for size in shape):
                raise ValueError('invalid shape')
            raw = _base64.b64decode(value['data'], validate=True)
            return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        except (TypeError, ValueError, _binascii.Error) as exc:
            raise ValueError(f'{path}: invalid ndarray payload') from exc
    raise ValueError(f'{path}: invalid or unknown tagged value {tag!r}')


# ---------------------------------------------------------------------------
# Recorded fact types
# ---------------------------------------------------------------------------


@dataclass
class ProviderFacts:
    """Per-call provider response facts (model routing / cost audit)."""

    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    inference_time: float | None = None
    abort_wait_time: float | None = None
    num_preempted: int | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = {
            'model': self.model,
            'response_id': self.response_id,
            'usage': self.usage,
            'finish_reason': self.finish_reason,
            'inference_time': self.inference_time,
            'abort_wait_time': self.abort_wait_time,
            'num_preempted': self.num_preempted,
        }
        return {k: v for k, v in fields.items() if v is not None}


@dataclass
class InvocationFacts:
    """Declared-only request ordinals (white-box scaffolds declare via
    request_meta; black-box lanes stay None — the engine never infers)."""

    iteration: int | None = None
    retry: int | None = None

    def to_dict(self) -> dict[str, Any]:
        fields = {'iteration': self.iteration, 'retry': self.retry}
        return {k: v for k, v in fields.items() if v is not None}


@dataclass
class SegmentBoundary:
    """Boundary conclusion on the llm node that opens a segment.

    Mapped from the assembler's boundary metadata: kind <- segment_type,
    causes <- boundary_causes, evidence <- the rest.
    """

    starts_new_segment: bool = False
    kind: str | None = None
    causes: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {'starts_new_segment': self.starts_new_segment}
        if self.kind is not None:
            data['kind'] = self.kind
        if self.causes:
            data['causes'] = list(self.causes)
        if self.evidence:
            data['evidence'] = self.evidence
        return data


@dataclass
class TokenAccount:
    """Tokens owned by one templated message.

    ``gen_start`` and ``gen_len`` locate the sampled span. Tokens outside that
    span come from the tokenizer template or observations and remain mask-0.
    """

    token_ids: list[int] | None = None
    gen_start: int = 0
    gen_len: int = 0
    logprobs: list[float] | None = None
    topk_indices: list[list[int]] | None = None
    topk_values: list[list[float]] | None = None
    routed_experts: Any | None = None  # only real hytuner backends fill this
    multi_modal: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        count = len(self.token_ids) if self.token_ids is not None else 0
        if not 0 <= self.gen_start <= count:
            raise ValueError(f'gen_start {self.gen_start} outside token_ids length {count}')
        if not 0 <= self.gen_len <= count - self.gen_start:
            raise ValueError(f'gen_start {self.gen_start} + gen_len {self.gen_len} outside token_ids length {count}')
        for name in ('logprobs', 'topk_indices', 'topk_values'):
            value = getattr(self, name)
            if value is not None and len(value) != self.gen_len:
                raise ValueError(f'{name} length {len(value)} != gen_len {self.gen_len}')
        if (self.topk_indices is None) != (self.topk_values is None):
            raise ValueError('topk_indices and topk_values must be present together')
        if self.topk_indices is not None and self.topk_values is not None:
            for row, (row_i, row_v) in enumerate(zip(self.topk_indices, self.topk_values, strict=True)):
                if len(row_i) != len(row_v):
                    raise ValueError(f'topk row {row} width mismatch')

    @property
    def generated_ids(self) -> list[int]:
        """The sampled span (mask=1 domain; numeric arrays align here)."""
        if not self.token_ids or self.gen_len <= 0:
            return []
        return self.token_ids[self.gen_start : self.gen_start + self.gen_len]

    @property
    def glue_ids(self) -> list[int]:
        """Prompt-side tokens before the sampled span."""
        if not self.token_ids:
            return []
        return self.token_ids[: self.gen_start]

    @property
    def trailing_ids(self) -> list[int]:
        """Mask-0 tokens after the sampled span."""
        if not self.token_ids:
            return []
        return self.token_ids[self.gen_start + self.gen_len :]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {'token_ids': self.token_ids}
        if self.gen_start:
            data['gen_start'] = self.gen_start
        if self.gen_len:
            data['gen_len'] = self.gen_len
        for key in ('logprobs', 'topk_indices', 'topk_values', 'routed_experts', 'multi_modal'):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


@dataclass
class SpawnLink:
    """Tool birthplace of a chain (None for client-side forks e.g. session_memory)."""

    node: NodeId | None = None
    tool_use_id: str | None = None
    candidates: list[str] = field(default_factory=list)


@dataclass
class JoinLink:
    """Re-integration of a chain's result into its parent (None = fire-and-forget)."""

    node: NodeId | None = None
    tool_use_id: str | None = None
    kind: str | None = None  # inline | poll | notify


@dataclass
class Chain:
    """One execution stream. 'main' is an ordinary entry (spawn/join/parent None).

    Identity rules: main='main'; declared chains pass their declared id
    through verbatim; inferred chains use the wire agent_id; identity-less side
    streams synthesize ``chain@<root NodeId>``. ``roots`` marks ownership
    entries whose chain cannot be inherited from a physical ancestor. Typed
    links contain resolved facts; evidence preserves ambiguous candidates.
    """

    chain_id: ChainId
    kind: str = 'main'
    parent_chain_id: ChainId | None = None
    roots: list[NodeId] = field(default_factory=list)
    spawn: SpawnLink | None = None
    join: JoinLink | None = None
    detector: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryNode:
    """One scaffold-agnostic wire message in the public trajectory contract.

    Three zones: wire facts / semantic annotations / reward.

    Chain membership and render continuity are derived — no chain_id, no
    render key. ``parent`` and ``children`` expose the public physical topology in both
    directions. Render variants share ``message_id`` but keep distinct
    ``node_id`` values; inference attempts never deduplicate on message
    identity.
    """

    node_id: NodeId
    message_id: MessageId
    parent: NodeId | None
    generated_by: Literal['llm', 'observation', 'client']
    children: list[NodeId] = field(kw_only=True)

    # -- zone 1: wire facts (recorded, append-only) --
    original_message: dict[str, Any] = field(default_factory=dict)
    templated_message: str | None = None  # Rendered by the tokenizer template.
    request_kwargs: dict[str, Any] | None = None  # llm only
    token_account: TokenAccount | None = None  # None on message lane, or when another node owns the rendered tokens.
    provider_facts: ProviderFacts | None = None  # llm only
    request_message_overrides: dict[NodeId, dict[str, Any]] = field(default_factory=dict)  # llm request variants
    timeline: list[dict[str, Any]] = field(default_factory=list)  # node-anchored events
    as_sent_variants: list[dict[str, Any]] = field(default_factory=list)  # audit-only wire occurrences

    # -- zone 2: trajectory semantics (plain data; reading needs no policy) --
    segment_boundary: SegmentBoundary | None = None
    selection: Literal['selected', 'rejected_sibling'] | None = None
    invocation: InvocationFacts | None = None

    # -- zone 3: writable slot (training side) --
    reward: float | None = None
    training_tensors: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _normalize_training_tensor(self, name: str, value: object) -> torch.Tensor:
        import torch

        _validate_node_training_tensor_name(self.node_id, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'node {self.node_id} training tensor {name!r} must be torch.Tensor')
        if self.token_account is None:
            raise ValueError(f'node {self.node_id} has no token_account')
        if value.ndim < 1:
            raise ValueError(f'node {self.node_id} training tensor {name!r} must have at least one dimension')
        if value.shape[0] != self.token_account.gen_len:
            raise ValueError(
                f'node {self.node_id} training tensor {name!r} length {value.shape[0]} '
                f'does not match gen_len {self.token_account.gen_len}'
            )
        if name == EFFECTIVE_TRAIN_MASK and value.dtype is not torch.bool:
            raise TypeError('effective_train_mask must have dtype torch.bool')
        if name != EFFECTIVE_TRAIN_MASK and not value.is_floating_point():
            raise TypeError(f'training tensor {name!r} must have a floating dtype')
        return value.detach().to(device='cpu').contiguous().clone()

    def set_training_tensor(self, name: str, value: torch.Tensor) -> None:
        if name in self.training_tensors:
            raise KeyError(f'node {self.node_id} already has training tensor {name!r}')
        self.training_tensors[name] = self._normalize_training_tensor(name, value)

    def replace_training_tensor(self, name: str, value: torch.Tensor) -> None:
        self.training_tensors[name] = self._normalize_training_tensor(name, value)

    def get_training_tensor(self, name: str) -> torch.Tensor:
        if name not in self.training_tensors:
            raise KeyError(f'node {self.node_id} has no training tensor {name!r}')
        return self.training_tensors[name]


# ---------------------------------------------------------------------------
# Segments (derived training view)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedTokenSpan:
    node_id: NodeId
    response_start: int
    response_stop: int
    node_start: int
    node_stop: int


@dataclass(frozen=True)
class ResponseTokenSpan:
    node_id: NodeId
    response_start: int
    response_stop: int
    node_token_start: int
    node_token_stop: int


@dataclass
class TrajectorySegment:
    """One training sample. Field names follow HyTunerSegment."""

    # sample identity
    trajectory_id: str
    segment_id: str
    segment_index: int
    chain_id: ChainId
    node_ids: list[NodeId]
    boundary_node_id: NodeId | None
    is_closed: bool
    # training payload
    prompt_ids: list[int] | None = None
    raw_prompt_ids: list[int] | None = None
    response_ids: list[int] | None = None
    response_mask: list[int] | None = None
    response_token_spans: list[ResponseTokenSpan] = field(default_factory=list)
    generated_token_spans: list[GeneratedTokenSpan] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    topk_indices: list[list[int]] | None = None
    topk_values: list[list[float]] | None = None
    routed_experts: Any | None = None
    multi_modal_data: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None
    num_turns: int = 0
    segment_type: str = 'conversation'
    sidechain_info: dict[str, Any] | None = None
    boundary: SegmentBoundary | None = None
    split_cause: str | None = None
    reward: float | None = None
    timeline: list[dict[str, Any]] = field(default_factory=list)


@dataclass(eq=False)
class SegmentTrainingData:
    """Canonical full-response tensors for one segment occurrence."""

    response_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    canonical_segment_fingerprint: tuple[Any, ...]
    loss_mask: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


class SegmentBatch(list):
    """List of TrajectorySegment plus an export manifest describing how it was built."""

    def __init__(self, segments: list[TrajectorySegment], manifest: dict[str, Any]):
        super().__init__(segments)
        self.manifest = manifest


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class Trajectory:
    """Unified trajectory: structure + training data in one picklable object.

    Forest semantics; self-contained; scaffold/policy agnostic. There is no
    synthetic root; an empty trajectory has roots == []. Gateway runtime mutates
    the live forest; freeze either copies it for an active snapshot or finalizes
    it in place when the session ends.
    """

    def __init__(
        self,
        *,
        session_id: str,
        state: str = 'completed',
        nodes: dict[NodeId, TrajectoryNode] | None = None,
        roots: list[NodeId] | None = None,
        chains: dict[ChainId, Chain] | None = None,
        latest_messages: list[dict[str, Any]] | None = None,
        timeline: list[dict[str, Any]] | None = None,
        final_reward: float | None = None,
        termination_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.schema_version = SCHEMA_VERSION
        self.session_id = session_id
        self.state = state
        self.nodes: dict[NodeId, TrajectoryNode] = {} if nodes is None else nodes
        self.roots: list[NodeId] = [] if roots is None else roots
        self.chains: dict[ChainId, Chain] = {} if chains is None else chains
        self.latest_messages = latest_messages
        self.timeline: list[dict[str, Any]] = [] if timeline is None else timeline  # non-node-anchored only
        self.final_reward = final_reward
        self.termination_reason = termination_reason
        self.metrics: dict[str, Any] = {} if metrics is None else metrics
        self.extra: dict[str, Any] = {} if extra is None else extra
        self.segment_training_data: dict[str, SegmentTrainingData] = {}
        self._invalidate_caches()

    # -- pickle: semantic state only, caches rebuilt on load --

    _STATE_FIELDS = (
        'schema_version',
        'session_id',
        'state',
        'nodes',
        'roots',
        'chains',
        'latest_messages',
        'timeline',
        'final_reward',
        'termination_reason',
        'metrics',
        'extra',
        'segment_training_data',
    )

    def __getstate__(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._STATE_FIELDS}

    def __setstate__(self, state: dict[str, Any]) -> None:
        version = state.get('schema_version')
        if version != SCHEMA_VERSION:
            raise ValueError(f'unsupported trajectory schema_version: {version!r} (expected {SCHEMA_VERSION})')
        for key in self._STATE_FIELDS:
            value = state.get(key, {}) if key == 'segment_training_data' else state[key]
            setattr(self, key, value)
        self._invalidate_caches()
        self._validate_segment_training_store()

    def _invalidate_caches(self) -> None:
        self._chain_index: dict[NodeId, ChainId] | None = None

    # ------------------------------------------------------------------
    # topology queries
    # ------------------------------------------------------------------

    def iter_nodes(self) -> Iterator[TrajectoryNode]:
        for node_id in sorted(self.nodes):  # node ids are assigned in assembly order
            yield self.nodes[node_id]

    def node_chain(self, node_id: NodeId) -> ChainId:
        """Derived membership: the chain whose root is the nearest ancestor
        chain-root; shared-prefix nodes belong to the parent chain."""
        index = self._chain_membership()
        return index.get(node_id, MAIN_CHAIN_ID)

    def _chain_membership(self) -> dict[NodeId, ChainId]:
        if self._chain_index is None:
            root_owner: dict[NodeId, ChainId] = {}
            for chain in self.chains.values():
                for root in chain.roots:
                    root_owner[root] = chain.chain_id
            # memoized upward walk: robust to any id order (restart-synthesized
            # branch parents carry LARGER ids than their children)
            index: dict[NodeId, ChainId] = {}
            for node_id in self.nodes:
                trail: list[NodeId] = []
                current: NodeId | None = node_id
                resolved = MAIN_CHAIN_ID
                while current is not None:
                    if current in index:
                        resolved = index[current]
                        break
                    if current in root_owner:
                        resolved = root_owner[current]
                        trail.append(current)
                        break
                    trail.append(current)
                    current = self.nodes[current].parent
                for nid in trail:
                    index[nid] = resolved
            self._chain_index = index
        return self._chain_index

    # ------------------------------------------------------------------
    # token reconstruction
    # ------------------------------------------------------------------

    def prompt_ids(self, node_id: NodeId) -> list[int]:
        """Input stream of an llm call: concat of ``token_ids`` along its own
        path (None skipped — those bytes live inside some llm's glue) ⊕ the
        node's own leading glue. Iterative: no recursion depth cap."""
        account = self.nodes[node_id].token_account
        if account is None or account.token_ids is None:
            raise ValueError(f'node {node_id} has no token account')
        result: list[int] = []
        for ancestor_id in self._path_to(node_id)[:-1]:
            ancestor = self.nodes[ancestor_id].token_account
            if ancestor is not None and ancestor.token_ids is not None:
                result += ancestor.token_ids
        result += account.glue_ids
        return result

    def _prompt_token_ownership(self, node_id: NodeId) -> list[tuple[NodeId, int, int, int, int]]:
        """Physical owners of the exact token stream returned by ``prompt_ids``."""
        request_account = self.nodes[node_id].token_account
        if request_account is None or request_account.token_ids is None:
            raise ValueError(f'node {node_id} has no token account')
        ownership: list[tuple[NodeId, int, int, int, int]] = []
        prompt_start = 0
        for ancestor_id in self._path_to(node_id)[:-1]:
            account = self.nodes[ancestor_id].token_account
            if account is None or account.token_ids is None:
                continue
            prompt_stop = prompt_start + len(account.token_ids)
            if prompt_stop > prompt_start:
                ownership.append((ancestor_id, prompt_start, prompt_stop, 0, len(account.token_ids)))
            prompt_start = prompt_stop
        prompt_stop = prompt_start + request_account.gen_start
        if prompt_stop > prompt_start:
            ownership.append((node_id, prompt_start, prompt_stop, 0, request_account.gen_start))
        return ownership

    def _path_to(self, node_id: NodeId) -> list[NodeId]:
        path: list[NodeId] = []
        current: NodeId | None = node_id
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # derived views
    # ------------------------------------------------------------------

    def _selected_paths(
        self,
        chain_ids: set[ChainId],
        roots: list[NodeId] | None = None,
    ) -> Iterator[tuple[ChainId, list[NodeId]]]:
        """Derive selected chain-local paths from frozen structure."""

        def chain_depth(chain_id: ChainId) -> int:
            depth = 0
            seen = {chain_id}
            chain = self.chains.get(chain_id)
            parent = chain.parent_chain_id if chain is not None else None
            while parent in self.chains and parent not in seen:
                seen.add(parent)
                depth += 1
                parent = self.chains[parent].parent_chain_id
            return depth

        entries = list(self.roots if roots is None else roots)
        if roots is None:
            entries.sort(
                key=lambda node_id: (
                    self.node_chain(node_id) == MAIN_CHAIN_ID,
                    -chain_depth(self.node_chain(node_id)),
                )
            )

        for root in entries:
            if self.nodes[root].selection == 'rejected_sibling':
                continue
            chain_id = self.node_chain(root)
            stack: list[tuple[str, NodeId, ChainId, NodeId, bool]] = [
                ('visit', root, chain_id, root, self.nodes[root].generated_by == 'llm')
            ]
            while stack:
                action, node_id, chain_id, branch_root, owns_llm = stack.pop()
                if action == 'emit':
                    if chain_id in chain_ids:
                        path = self._path_to(node_id)
                        yield chain_id, path[path.index(branch_root) :]
                    continue

                selected = [
                    child for child in self.nodes[node_id].children if self.nodes[child].selection != 'rejected_sibling'
                ]
                same = [child for child in selected if self.node_chain(child) == chain_id]
                foreign = [child for child in selected if self.node_chain(child) != chain_id]
                if not same and owns_llm:
                    stack.append(('emit', node_id, chain_id, branch_root, owns_llm))
                for child in reversed(same):
                    stack.append((
                        'visit',
                        child,
                        chain_id,
                        branch_root,
                        owns_llm or self.nodes[child].generated_by == 'llm',
                    ))
                for child in reversed(foreign):
                    child_chain = self.node_chain(child)
                    stack.append(('visit', child, child_chain, child, self.nodes[child].generated_by == 'llm'))

    def _belongs_to_previous_turn(self, node: TrajectoryNode) -> bool:
        return node.generated_by == 'observation' or node.original_message.get('role') in {'tool', 'function'}

    def to_messages(self, chain_id: ChainId = MAIN_CHAIN_ID) -> list[dict[str, Any]]:
        """Wire transcript of the chain's latest selected path."""
        paths = [path for _, path in self._selected_paths({chain_id})]
        if not paths:
            return []
        last_path = max(paths, key=lambda path: (self.nodes[path[-1]].message_id, path[-1]))
        request_node_id = next(
            (node_id for node_id in reversed(last_path) if self.nodes[node_id].generated_by == 'llm'),
            None,
        )
        if request_node_id is None:
            return [_copy.deepcopy(self.nodes[node_id].original_message) for node_id in last_path]
        return self._messages_for_nodes(last_path, request_node_id)

    # ------------------------------------------------------------------
    # Canonical training exit
    # ------------------------------------------------------------------

    def to_segments(
        self,
        chains: str | list[ChainId] = 'all',
        include_rejected: bool = False,
        reward_policy: Any | None = None,
    ) -> SegmentBatch:
        """Build read-only training segments from the frozen trajectory."""
        if include_rejected:
            raise NotImplementedError('rejected-sibling export is reserved but not implemented yet')
        if reward_policy is not None:
            raise NotImplementedError('custom reward_policy lands with the training-side integration')
        selected: list[ChainId]
        if isinstance(chains, str):
            if chains == 'all':
                selected = list(self.chains)
            elif chains == 'main':
                selected = [MAIN_CHAIN_ID]
            else:
                raise ValueError(f"chains must be 'all', 'main', or a collection of chain ids; got {chains!r}")
        else:
            selected = list(chains)
            missing = [c for c in selected if c not in self.chains]
            if missing:
                raise ValueError(f'unknown chain ids: {missing}')

        return self._build_segments(
            selected,
            chains=chains,
            legacy=False,
        )

    def _build_segments(
        self,
        selected: list[ChainId],
        *,
        chains: str | list[ChainId],
        legacy: bool,
    ) -> SegmentBatch:
        segments: list[TrajectorySegment] = []
        exported: set[NodeId] = set()
        paths = list(self._selected_paths(set(selected)))
        if legacy:
            ordered: list[tuple[ChainId, list[NodeId]]] = []
            index = 0
            while index < len(paths):
                chain_id, path = paths[index]
                if chain_id == MAIN_CHAIN_ID:
                    ordered.append((chain_id, path))
                    index += 1
                    continue
                end = index + 1
                while end < len(paths) and paths[end][0] == chain_id and paths[end][1][0] == path[0]:
                    end += 1
                ordered.extend(reversed(paths[index:end]))
                index = end
            paths = ordered

        for chain_id, path in paths:
            sidechain_info = self._project_sidechain_info(chain_id)
            for group, boundary_node in self._semantic_groups(path, legacy=legacy):
                fresh = [node_id for node_id in group if node_id not in exported]
                exported.update(fresh)
                if not fresh:
                    continue
                entry_diagnostic = None
                if legacy:
                    first_boundary = self.nodes[fresh[0]].segment_boundary
                    if first_boundary is not None and self._physical_only(first_boundary):
                        entry_diagnostic = self._split_cause(fresh[0])
                # A declared boundary (e.g. compaction_resume) may coincide with physical
                # divergence evidence; the split_cause channel reports PHYSICAL-ONLY splits,
                # so the fallback must honor the same physical_only guard as above —
                # otherwise declared compaction segments get mislabeled as restarts downstream.
                boundary_split_cause = None
                if boundary_node is not None:
                    node_boundary = self.nodes[boundary_node].segment_boundary
                    if node_boundary is not None and self._physical_only(node_boundary):
                        boundary_split_cause = self._split_cause(boundary_node)
                for physical_group, diagnostic in self._frozen_physical_groups(fresh):
                    diagnostic = diagnostic or (entry_diagnostic if physical_group[0] == fresh[0] else None)
                    segment = self._build_segment(
                        chain_id,
                        path,
                        physical_group,
                        boundary_node,
                        diagnostic or boundary_split_cause,
                        sidechain_info,
                        legacy=legacy,
                    )
                    if segment is not None:
                        if legacy and diagnostic:
                            segment.timeline.append({
                                'event_type': 'gateway_diagnostic',
                                'status': 'WARN',
                                'attributes': {'code': diagnostic},
                            })
                        segments.append(segment)
        chain_order: dict[ChainId, int] = {}
        for segment in segments:
            chain_order.setdefault(segment.chain_id, len(chain_order))
        segments.sort(
            key=lambda segment: (
                chain_order[segment.chain_id],
                self.nodes[segment.node_ids[0]].message_id,
            )
        )
        for index, segment in enumerate(segments):
            segment.segment_index = index
        manifest = {
            'chains': chains if isinstance(chains, str) else list(chains),
            'dedup': True,
            'include_rejected': False,
            'ordering': (
                'legacy chain order; assembly-message order within each chain'
                if legacy
                else 'canonical chain order; assembly-message order within each chain'
            ),
            'dedup_ownership': 'first-visited selected path',
            'reward_policy': 'default: node rewards only; final_reward is not broadcast',
        }
        return SegmentBatch(segments, manifest)

    def to_legacy_segments(self) -> SegmentBatch:
        """Return the training projection consumed by the current HyTuner path."""
        return self._build_segments(
            list(self.chains),
            chains='all',
            legacy=True,
        )

    # ------------------------------------------------------------------
    # Segment-occurrence training tensors
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_segment_fingerprint(segment: TrajectorySegment) -> tuple[Any, ...]:
        """Immutable canonical identity and axes; projection order is intentionally excluded."""
        return (
            segment.trajectory_id,
            segment.segment_id,
            segment.chain_id,
            tuple(segment.node_ids),
            None if segment.prompt_ids is None else tuple(segment.prompt_ids),
            None if segment.response_ids is None else tuple(segment.response_ids),
            None if segment.response_mask is None else tuple(bool(item) for item in segment.response_mask),
            tuple(
                (
                    span.node_id,
                    span.response_start,
                    span.response_stop,
                    span.node_token_start,
                    span.node_token_stop,
                )
                for span in segment.response_token_spans
            ),
            tuple(
                (span.node_id, span.response_start, span.response_stop, span.node_start, span.node_stop)
                for span in segment.generated_token_spans
            ),
        )

    def _canonical_segments_by_id(self) -> dict[str, TrajectorySegment]:
        segments: dict[str, TrajectorySegment] = {}
        for segment in self.to_segments():
            if segment.segment_id in segments:
                raise ValueError(
                    f'trajectory {self.session_id!r} has ambiguous canonical segment {segment.segment_id!r}'
                )
            segments[segment.segment_id] = segment
        return segments

    @staticmethod
    def _normalize_loss_mask(segment_id: str, response_count: int, loss_mask: object) -> torch.Tensor:
        import torch

        if not isinstance(loss_mask, torch.Tensor) or loss_mask.dtype is not torch.bool:
            raise TypeError(f'segment {segment_id!r} loss_mask must be a torch.bool tensor')
        if loss_mask.ndim != 1:
            raise ValueError(f'segment {segment_id!r} loss_mask must be one-dimensional')
        if loss_mask.shape[0] != response_count:
            raise ValueError(
                f'segment {segment_id!r} loss_mask length {loss_mask.shape[0]} '
                f'does not match response length {response_count}'
            )
        return loss_mask.detach().to(device='cpu').contiguous().clone()

    @staticmethod
    def _normalize_segment_training_tensor(
        segment_id: str,
        name: str,
        response_count: int,
        value: object,
    ) -> torch.Tensor:
        import torch

        if not isinstance(name, str):
            raise TypeError('segment training tensor name must be a string')
        _validate_segment_training_tensor_name(name, reserved_error=KeyError)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'segment {segment_id!r} training tensor {name!r} must be torch.Tensor')
        if value.ndim < 1:
            raise ValueError(f'segment {segment_id!r} training tensor {name!r} must have at least one dimension')
        if value.shape[0] != response_count:
            raise ValueError(
                f'segment {segment_id!r} training tensor {name!r} length {value.shape[0]} '
                f'does not match response length {response_count}'
            )
        return value.detach().to(device='cpu').contiguous().clone()

    @staticmethod
    def _clone_segment_training_record(record: SegmentTrainingData) -> SegmentTrainingData:
        return SegmentTrainingData(
            response_ids=record.response_ids,
            response_mask=record.response_mask,
            canonical_segment_fingerprint=record.canonical_segment_fingerprint,
            loss_mask=record.loss_mask.clone(),
            tensors={name: value.clone() for name, value in record.tensors.items()},
        )

    def _validate_segment_training_store(self) -> None:
        if not isinstance(self.segment_training_data, dict):
            raise TypeError('segment_training_data must be a dictionary')
        if self.segment_training_data and self.state != 'completed':
            raise ValueError('segment training data requires a completed trajectory')
        canonical_segments = self._canonical_segments_by_id() if self.segment_training_data else {}
        for segment_id, record in self.segment_training_data.items():
            if not isinstance(segment_id, str):
                raise TypeError('segment training data keys must be strings')
            canonical = canonical_segments.get(segment_id)
            if canonical is None:
                raise KeyError(f'trajectory {self.session_id!r} has no canonical segment {segment_id!r}')
            if not isinstance(record, SegmentTrainingData):
                raise TypeError(f'segment training data {segment_id!r} must be SegmentTrainingData')
            if record.canonical_segment_fingerprint != self._canonical_segment_fingerprint(canonical):
                raise ValueError(f'canonical segment fingerprint changed for {segment_id!r}')
            if record.response_ids != tuple(canonical.response_ids or ()) or record.response_mask != tuple(
                bool(item) for item in (canonical.response_mask or ())
            ):
                raise ValueError(f'canonical response fingerprint changed for segment {segment_id!r}')

    def prepare_segment_training_data(
        self,
        initializations: Any | None = None,
        tensor_updates: Any | None = None,
        replace: bool = False,
    ) -> dict[str, SegmentTrainingData]:
        """Validate a batch and return a complete replacement without publishing it."""
        if self.state != 'completed':
            raise ValueError('segment training data requires a completed trajectory')
        initializations = tuple(initializations or ())
        canonical_segments = self._canonical_segments_by_id() if initializations else {}
        replacement = dict(self.segment_training_data)
        cloned: set[str] = set()

        def mutable_record(segment_id: str) -> SegmentTrainingData:
            if segment_id not in replacement:
                raise KeyError(f'segment {segment_id!r} has no initialized training data')
            record = replacement[segment_id]
            if segment_id not in cloned:
                record = self._clone_segment_training_record(record)
                replacement[segment_id] = record
                cloned.add(segment_id)
            return record

        for segment, loss_mask in initializations or ():
            if not isinstance(segment, TrajectorySegment):
                raise TypeError('segment training initialization requires a TrajectorySegment')
            canonical = canonical_segments.get(segment.segment_id)
            if canonical is None:
                raise KeyError(f'trajectory {self.session_id!r} has no canonical segment {segment.segment_id!r}')
            normalized_loss_mask = self._normalize_loss_mask(
                segment.segment_id,
                len(canonical.response_ids or ()),
                loss_mask,
            )
            existing = replacement.get(segment.segment_id)
            if existing is None:
                replacement[segment.segment_id] = SegmentTrainingData(
                    response_ids=tuple(canonical.response_ids or ()),
                    response_mask=tuple(bool(item) for item in (canonical.response_mask or ())),
                    canonical_segment_fingerprint=self._canonical_segment_fingerprint(canonical),
                    loss_mask=normalized_loss_mask,
                )
                cloned.add(segment.segment_id)
                continue
            record = mutable_record(segment.segment_id)
            if not record.loss_mask.equal(normalized_loss_mask):
                raise ValueError(f'canonical loss_mask fingerprint changed for segment {segment.segment_id!r}')

        for segment_id, name, value in tensor_updates or ():
            record = mutable_record(segment_id)
            normalized = self._normalize_segment_training_tensor(
                segment_id,
                name,
                len(record.response_ids),
                value,
            )
            if not replace and name in record.tensors:
                raise KeyError(f'segment {segment_id!r} already has training tensor {name!r}')
            record.tensors[name] = normalized

        return replacement

    def to_legacy_chat_completion(self) -> dict[str, Any]:
        """Project one row for the existing chat_completions JSONL artifact."""
        metric_names = ('abort_wait_time', 'llm_inference_time', 'observation_time', 'reward_time')
        row = {
            'messages': self._latest_messages(),
            'num_turns': self._legacy_num_turns(),
            'trajectory_id': self.session_id,
            'termination_reason': self.termination_reason,
            'runtime_id': self.extra.get('runtime_id') or 'unknown',
            'runtime_ids': _copy.deepcopy(self.extra.get('runtime_ids') or []),
            'eval_runtime_id': self.extra.get('eval_runtime_id') or 'unknown',
            'metrics': {name: self.metrics.get(name, 0.0) for name in metric_names},
        }
        for name in ('tool_call_stats', 'tool_call_summary'):
            value = self.extra.get(name)
            if value:
                row[name] = _copy.deepcopy(value)
        # The chat_completions JSONL is a debug/observability artifact that must be JSON-safe.
        # Structural training tensors (R3 routed_experts, per-token logprobs/topk numpy arrays)
        # can ride inside messages/extra here but belong only in the token/segment lane
        # (to_legacy_hytuner). Drop any non-JSON-serializable values so the artifact dump can't
        # crash the trajectory — matching the legacy client, whose chat dump never carried them.
        return _json_safe(row)

    def to_legacy_hytuner(self) -> dict[str, Any]:
        """Project the current HyTuner runner payload without importing Nexus."""
        chunks = self.to_legacy_segments()
        has_llm = any(node.generated_by == 'llm' for node in self.nodes.values())
        if has_llm and (
            not chunks
            or any(
                chunk.prompt_ids is None or chunk.response_ids is None or chunk.response_mask is None
                for chunk in chunks
            )
        ):
            raise RuntimeError('Gateway HyTuner path requires a token-level trajectory')

        segments = [
            {
                'prompt_ids': list(chunk.prompt_ids or []),
                'raw_input_ids': list(chunk.raw_prompt_ids or []),
                'response_ids': list(chunk.response_ids or []),
                'response_mask': list(chunk.response_mask or []),
                'response_logprobs': _copy.deepcopy(chunk.response_logprobs),
                'num_turns': chunk.num_turns,
                'segment_type': chunk.segment_type,
                'split_cause': chunk.split_cause,
                'multi_modal_data': _copy.deepcopy(chunk.multi_modal_data or {}),
                'sidechain_info': _copy.deepcopy(chunk.sidechain_info),
                'routed_experts': _copy.deepcopy(chunk.routed_experts),
            }
            for chunk in chunks
        ]
        if not segments:
            segments = [{'prompt_ids': [], 'response_ids': [], 'response_mask': []}]

        primary = [chunk for chunk in chunks if chunk.chain_id == MAIN_CHAIN_ID] or list(chunks)
        extra_fields = _copy.deepcopy(self.extra)
        extra_fields.pop('tool_call_stats', None)
        extra_fields.pop('tool_call_summary', None)
        if primary:
            extra_fields.update(self._legacy_topk_fields(primary[-1]))
        routed_experts = next(
            (chunk.routed_experts for chunk in reversed(chunks) if chunk.routed_experts is not None),
            None,
        )
        return {
            'segments': segments,
            'routed_experts': routed_experts,
            'reward_score': self.final_reward,
            'termination_reason': self.termination_reason,
            'metrics': _copy.deepcopy(self.metrics),
            'extra_fields': extra_fields,
            'request_id': self.session_id,
        }

    @staticmethod
    def _legacy_topk_fields(segment: TrajectorySegment) -> dict[str, Any]:
        if segment.topk_indices is None and segment.topk_values is None:
            return {}
        if segment.topk_indices is None or segment.topk_values is None:
            raise ValueError('top-k indices and values must be present together')
        response_ids = segment.response_ids or []
        response_mask = segment.response_mask or []
        if len(segment.topk_indices) != len(response_ids) or len(segment.topk_values) != len(response_ids):
            raise ValueError('top-k rows must align with response_ids')
        if len(response_mask) != len(response_ids):
            raise ValueError('response_mask must align with response_ids')
        widths = {len(row) for row in segment.topk_indices if row}
        if len(widths) != 1:
            raise ValueError('top-k width must remain constant within a segment')
        width = next(iter(widths), 0)
        indices: list[list[int]] = []
        values: list[list[float]] = []
        for index_row, value_row, generated in zip(
            segment.topk_indices,
            segment.topk_values,
            response_mask,
            strict=True,
        ):
            if len(index_row) != len(value_row) or (index_row and len(index_row) != width):
                raise ValueError('top-k index/value rows must have equal, constant width')
            if generated and not index_row:
                raise ValueError('generated tokens must have top-k rows when top-k is present')
            indices.append(list(index_row) if index_row else [0] * width)
            values.append(list(value_row) if value_row else [0.0] * width)
        return {
            'resp_topk_logprob_indice': indices,
            'resp_topk_logprob_value': values,
        }

    def _latest_messages(self) -> list[dict[str, Any]]:
        if self.latest_messages is not None:
            return _copy.deepcopy(self.latest_messages)
        llm_nodes = [node for node in self.nodes.values() if node.generated_by == 'llm']
        if not llm_nodes:
            return []
        latest = max(llm_nodes, key=lambda node: node.message_id)
        tail = latest
        for node in self.nodes.values():
            if node.message_id <= tail.message_id:
                continue
            if latest.node_id in self._path_to(node.node_id):
                tail = node
        return self._messages_for_nodes(self._path_to(tail.node_id), latest.node_id)

    def _legacy_num_turns(self) -> int:
        llm_nodes = sorted(
            (node for node in self.nodes.values() if node.generated_by == 'llm'),
            key=lambda node: node.message_id,
        )
        reset_at = -1
        for node in llm_nodes:
            boundary = node.segment_boundary
            if boundary is not None and boundary.kind == 'compaction_resume':
                reset_at = node.message_id
        return sum(node.message_id >= reset_at for node in llm_nodes)

    def _semantic_groups(
        self,
        path: list[NodeId],
        *,
        legacy: bool = False,
    ) -> list[tuple[list[NodeId], NodeId | None]]:
        """Split the path's llm nodes at segment_boundary.starts_new_segment."""
        groups: list[tuple[list[NodeId], NodeId | None]] = []
        current: list[NodeId] = []
        current_boundary: NodeId | None = None
        for node_id in path:
            node = self.nodes[node_id]
            if node.generated_by != 'llm':
                continue
            boundary = node.segment_boundary
            starts = boundary is not None and boundary.starts_new_segment
            if boundary is not None and starts and legacy and self._physical_only(boundary):
                starts = False
            if current and starts:
                groups.append((current, current_boundary))
                current, current_boundary = [], node_id
            elif not current and starts:
                current_boundary = node_id
            current.append(node_id)
        if current:
            groups.append((current, current_boundary))
        return groups

    def _split_cause(self, boundary_node: NodeId | None) -> str | None:
        if boundary_node is None:
            return None
        boundary = self.nodes[boundary_node].segment_boundary
        if boundary is None:
            return None
        physical = {
            'render_divergence',
            'render_context_discontinuity',
            'token_prefix_discontinuity',
            'training_representation_discontinuity',
        }
        return next((cause for cause in boundary.causes if cause in physical), None)

    def _frozen_physical_groups(self, nodes: list[NodeId]) -> list[tuple[list[NodeId], str | None]]:
        groups: list[tuple[list[NodeId], str | None]] = []
        current: list[NodeId] = []
        current_cause: str | None = None
        for node_id in nodes:
            boundary = self.nodes[node_id].segment_boundary
            cause = self._split_cause(node_id) if boundary is not None and self._physical_only(boundary) else None
            if current and cause:
                groups.append((current, current_cause))
                current = []
                current_cause = cause
            current.append(node_id)
        if current:
            groups.append((current, current_cause))
        return groups

    @staticmethod
    def _physical_only(boundary: SegmentBoundary) -> bool:
        return boundary.evidence.get('physical_only') is True

    def _project_sidechain_info(self, chain_id: ChainId) -> dict[str, Any] | None:
        if chain_id == MAIN_CHAIN_ID:
            return None
        chain = self.chains[chain_id]
        evidence = chain.evidence
        descriptor: dict[str, Any] = {'kind': chain.kind}
        if chain.detector is not None:
            descriptor['detector'] = chain.detector
        agent_id = evidence.get('collides_with')
        if agent_id is None and not chain_id.startswith('chain@'):
            agent_id = chain_id
        if agent_id is not None:
            descriptor['agent_id'] = agent_id
        if evidence.get('attributes'):
            descriptor['attributes'] = _copy.deepcopy(evidence['attributes'])

        correlation = evidence.get('correlation')
        if isinstance(correlation, dict):
            anchor = _copy.deepcopy(correlation)
        else:
            candidates = evidence.get('correlation_candidates') or []
            anchor = {}
            if candidates:
                first = candidates[0]
                anchor = {
                    key: _copy.deepcopy(value)
                    for key, value in first.items()
                    if all(candidate.get(key) == value for candidate in candidates[1:])
                }

        for key in ('spawn_tool_use_id', 'spawn_candidates', 'joined', 'join_tool_use_id', 'join_kind'):
            anchor.pop(key, None)
        if chain.spawn is not None:
            if chain.spawn.tool_use_id is not None:
                anchor['spawn_tool_use_id'] = chain.spawn.tool_use_id
            elif chain.spawn.candidates:
                anchor['spawn_candidates'] = list(chain.spawn.candidates)
            anchor['joined'] = chain.join is not None
        elif chain.join is not None:
            anchor['joined'] = True
        if chain.join is not None:
            if chain.join.tool_use_id is not None:
                anchor['join_tool_use_id'] = chain.join.tool_use_id
            if chain.join.kind is not None:
                anchor['join_kind'] = chain.join.kind

        return {
            'kind': chain.kind,
            'descriptor': descriptor,
            'fork_anchor': anchor or None,
        }

    def _build_segment(
        self,
        chain_id: ChainId,
        path: list[NodeId],
        group: list[NodeId],
        boundary_node: NodeId | None,
        split_cause: str | None,
        sidechain_info: dict[str, Any] | None,
        *,
        legacy: bool,
    ) -> TrajectorySegment | None:
        if not group:
            return None
        first = self.nodes[group[0]]
        token_level = first.token_account is not None
        if any((self.nodes[node_id].token_account is not None) != token_level for node_id in group):
            raise ValueError('training representations must not be mixed within a segment')
        boundary = self.nodes[boundary_node].segment_boundary if boundary_node is not None else None
        segment_type = boundary.kind if boundary is not None and boundary.kind else 'conversation'
        node_interval = self._group_message_nodes(path, group)
        messages = self._messages_for_nodes(node_interval, group[0])
        timeline: list[dict[str, Any]] = []
        start = node_interval.index(group[0])
        for node_index, node_id in enumerate(node_interval[start:], start=start):
            node = self.nodes[node_id]
            for event in node.timeline:
                item = _copy.deepcopy(event)
                item.setdefault('node_index', node_index)
                item.setdefault('node_role', node.original_message.get('role'))
                item.setdefault('generated_by', node.generated_by)
                timeline.append(item)
        multi_modal_data: dict[str, Any] = {}
        for node_id in reversed(group):
            account = self.nodes[node_id].token_account
            if account is not None and account.multi_modal is not None:
                multi_modal_data = _copy.deepcopy(account.multi_modal)
                break
        common: dict[str, Any] = {
            'trajectory_id': self.session_id,
            'segment_id': f'{chain_id}@{group[0]}',
            'segment_index': -1,
            'chain_id': chain_id,
            'node_ids': list(group),
            'boundary_node_id': boundary_node,
            'is_closed': self._group_is_closed(path, group),
            'messages': messages,
            'num_turns': len(group),
            'multi_modal_data': multi_modal_data,
            'segment_type': segment_type,
            'sidechain_info': _copy.deepcopy(sidechain_info),
            'boundary': boundary,
            'split_cause': split_cause,
            'timeline': timeline,
        }
        if first.token_account is None:
            # Representation boundaries keep message- and token-level segments homogeneous.
            rewards = [self.nodes[n].reward for n in group if self.nodes[n].reward is not None]
            return TrajectorySegment(**common, reward=sum(rewards) if rewards else None)
        prompt_ids = self.prompt_ids(group[0])
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_token_spans: list[ResponseTokenSpan] = []
        generated_token_spans: list[GeneratedTokenSpan] = []
        response_logprobs: list[float] = []
        topk_indices: list[list[int]] = []
        topk_values: list[list[float]] = []
        generated_accounts: list[TokenAccount] = []
        for node_id in group:
            account = self.nodes[node_id].token_account
            if account is not None and account.gen_len > 0:
                generated_accounts.append(account)
        logprob_presence = {account.logprobs is not None for account in generated_accounts}
        topk_presence = {account.topk_indices is not None for account in generated_accounts}
        if len(logprob_presence) > 1 and not legacy:
            raise ValueError('logprobs must be present for every generated token in a segment or absent for all')
        if len(topk_presence) > 1:
            raise ValueError('top-k must be present for every generated token in a segment or absent for all')
        has_logprobs = True in logprob_presence
        has_topk = True in topk_presence
        routed_experts: Any = None
        rewards: list[float] = []

        expected_stream = prompt_ids
        for node_id in group:
            node = self.nodes[node_id]
            acct = node.token_account
            assert acct is not None
            turn_prompt = self.prompt_ids(node_id)
            if turn_prompt[: len(expected_stream)] != expected_stream:
                raise ValueError('non-contiguous nodes share a segment')
            # inter-turn input tokens (mask=0): templated messages between
            # the previous llm and this one ⊕ this llm's leading glue
            observation = turn_prompt[len(expected_stream) :]
            if node_id != group[0] and observation:
                observation_response_start = len(response_ids)
                response_ids.extend(observation)
                response_mask.extend([0] * len(observation))
                response_logprobs.extend([0.0] * len(observation))
                topk_indices.extend([[] for _ in observation])
                topk_values.extend([[] for _ in observation])
                for owner_node_id, prompt_start, prompt_stop, node_start, _ in self._prompt_token_ownership(node_id):
                    intersection_start = max(prompt_start, len(expected_stream))
                    intersection_stop = min(prompt_stop, len(turn_prompt))
                    if intersection_start >= intersection_stop:
                        continue
                    owner_start = node_start + intersection_start - prompt_start
                    owner_stop = owner_start + intersection_stop - intersection_start
                    response_start = observation_response_start + intersection_start - len(expected_stream)
                    response_stop = response_start + intersection_stop - intersection_start
                    response_token_spans.append(
                        ResponseTokenSpan(
                            node_id=owner_node_id,
                            response_start=response_start,
                            response_stop=response_stop,
                            node_token_start=owner_start,
                            node_token_stop=owner_stop,
                        )
                    )
            # model tokens (mask=1): the sampled span
            generated = acct.generated_ids
            response_start = len(response_ids)
            response_ids.extend(generated)
            response_stop = len(response_ids)
            if generated:
                response_token_spans.append(
                    ResponseTokenSpan(
                        node_id=node_id,
                        response_start=response_start,
                        response_stop=response_stop,
                        node_token_start=acct.gen_start,
                        node_token_stop=acct.gen_start + acct.gen_len,
                    )
                )
                generated_token_spans.append(
                    GeneratedTokenSpan(
                        node_id=node_id,
                        response_start=response_start,
                        response_stop=response_stop,
                        node_start=0,
                        node_stop=acct.gen_len,
                    )
                )
            response_mask.extend([1] * len(generated))
            if acct.logprobs is not None:
                response_logprobs.extend(acct.logprobs)  # exact gen_len (validated)
            else:
                response_logprobs.extend([0.0] * len(generated))
            if acct.topk_indices is not None:
                topk_indices.extend(acct.topk_indices)
                topk_values.extend(acct.topk_values or [])
            else:
                topk_indices.extend([[] for _ in generated])
                topk_values.extend([[] for _ in generated])
            trailing = acct.trailing_ids
            if trailing:
                trailing_response_start = len(response_ids)
                response_ids.extend(trailing)
                response_token_spans.append(
                    ResponseTokenSpan(
                        node_id=node_id,
                        response_start=trailing_response_start,
                        response_stop=len(response_ids),
                        node_token_start=acct.gen_start + acct.gen_len,
                        node_token_stop=len(acct.token_ids or []),
                    )
                )
                response_mask.extend([0] * len(trailing))
                response_logprobs.extend([0.0] * len(trailing))
                topk_indices.extend([[] for _ in trailing])
                topk_values.extend([[] for _ in trailing])
            if acct.routed_experts is not None:
                routed_experts = acct.routed_experts  # last-wins (known compromise)
            if node.reward is not None:
                rewards.append(node.reward)
            expected_stream = turn_prompt + generated + trailing

        if legacy:
            last_account = self.nodes[group[-1]].token_account
            assert last_account is not None
            raw_prompt_ids = self.prompt_ids(group[-1]) + list(last_account.generated_ids)
        else:
            raw_prompt_ids = self.prompt_ids(group[0])
        return TrajectorySegment(
            **common,
            prompt_ids=prompt_ids,
            raw_prompt_ids=raw_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_token_spans=response_token_spans,
            generated_token_spans=generated_token_spans,
            response_logprobs=response_logprobs if has_logprobs else None,
            topk_indices=topk_indices if has_topk else None,
            topk_values=topk_values if has_topk else None,
            routed_experts=routed_experts,
            reward=sum(rewards) if rewards else None,
        )

    def _group_message_nodes(self, path: list[NodeId], group: list[NodeId]) -> list[NodeId]:
        """Message span of a segment: full structural context (walk parents to
        the forest root — sidechain leaf paths start at the chain root, but the
        model's real input includes the shared parent-chain prefix; legacy
        recovers it from the first llm's recorded input) through the group's
        trailing observations."""
        end = path.index(group[-1]) + 1
        while end < len(path) and self._belongs_to_previous_turn(self.nodes[path[end]]):
            end += 1
        # ancestors above the path head (for chain-rooted paths)
        above: list[NodeId] = []
        current = self.nodes[path[0]].parent
        while current is not None:
            above.append(current)
            current = self.nodes[current].parent
        above.reverse()
        return above + path[:end]

    def _messages_for_nodes(self, node_ids: list[NodeId], request_node_id: NodeId) -> list[dict[str, Any]]:
        overrides = self.nodes[request_node_id].request_message_overrides
        return [_copy.deepcopy(overrides.get(node_id, self.nodes[node_id].original_message)) for node_id in node_ids]

    def _group_is_closed(self, path: list[NodeId], group: list[NodeId]) -> bool:
        """Turn closure is structural: a trailing llm with unanswered
        tool_calls is a dangling turn even in a completed trajectory."""
        if self.state == 'active':
            return False
        last = self.nodes[group[-1]]
        calls = last.original_message.get('tool_calls') or []
        if not calls:
            return True
        idx = path.index(group[-1])
        answered = 0
        for node_id in path[idx + 1 :]:
            if self._belongs_to_previous_turn(self.nodes[node_id]):
                answered += 1
            else:
                break
        return answered >= len(calls)

    # ------------------------------------------------------------------
    # dump / from_dump (versioned plain-data form; preserves provider-native facts)
    # ------------------------------------------------------------------

    def dump(self) -> dict[str, Any]:
        if self.segment_training_data:
            raise TypeError(
                'trajectory contains segment training tensors; '
                'use pickle or Ray serialization instead of Trajectory.dump()'
            )
        for node_id in sorted(self.nodes):
            if self.nodes[node_id].training_tensors:
                raise TypeError(
                    f'node {node_id} contains inline training tensors; '
                    'use pickle or Ray serialization instead of Trajectory.dump()'
                )
        data = {
            'schema_version': self.schema_version,
            'session_id': self.session_id,
            'state': self.state,
            'roots': list(self.roots),
            'nodes': [self._dump_node(self.nodes[node_id]) for node_id in sorted(self.nodes)],
            'chains': [self._dump_chain(chain) for chain in self.chains.values()],
            'latest_messages': self.latest_messages,
            'timeline': self.timeline,
            'final_reward': self.final_reward,
            'termination_reason': self.termination_reason,
            'metrics': self.metrics,
            'extra': self.extra,
        }
        return _encode_value(data, '$')

    @staticmethod
    def _dump_node(node: TrajectoryNode) -> dict[str, Any]:
        data: dict[str, Any] = {
            'node_id': node.node_id,
            'message_id': node.message_id,
            'parent': node.parent,
            'children': list(node.children),
            'generated_by': node.generated_by,
            'original_message': node.original_message,
        }
        if node.templated_message is not None:
            data['templated_message'] = node.templated_message
        if node.request_kwargs is not None:
            data['request_kwargs'] = node.request_kwargs
        if node.token_account is not None:
            data['token_account'] = node.token_account.to_dict()
        if node.provider_facts is not None:
            data['provider_facts'] = node.provider_facts.to_dict()
        if node.request_message_overrides:
            data['request_message_overrides'] = [
                {'node_id': node_id, 'message': message} for node_id, message in node.request_message_overrides.items()
            ]
        if node.timeline:
            data['timeline'] = node.timeline
        if node.as_sent_variants:
            data['as_sent_variants'] = node.as_sent_variants
        if node.segment_boundary is not None:
            data['segment_boundary'] = node.segment_boundary.to_dict()
        if node.selection is not None:
            data['selection'] = node.selection
        if node.invocation is not None:
            data['invocation'] = node.invocation.to_dict()
        if node.reward is not None:
            data['reward'] = node.reward
        return data

    @staticmethod
    def _dump_chain(chain: Chain) -> dict[str, Any]:
        data: dict[str, Any] = {
            'chain_id': chain.chain_id,
            'kind': chain.kind,
            'parent_chain_id': chain.parent_chain_id,
            'roots': list(chain.roots),
        }
        if chain.spawn is not None:
            data['spawn'] = {
                'node': chain.spawn.node,
                'tool_use_id': chain.spawn.tool_use_id,
                'candidates': list(chain.spawn.candidates),
            }
        if chain.join is not None:
            data['join'] = {'node': chain.join.node, 'tool_use_id': chain.join.tool_use_id, 'kind': chain.join.kind}
        if chain.detector is not None:
            data['detector'] = chain.detector
        if chain.evidence:
            data['evidence'] = chain.evidence
        return data

    @classmethod
    def from_dump(cls, data: dict[str, Any]) -> Trajectory:
        data = _decode_value(data, '$')
        if not isinstance(data, dict):
            raise TypeError('trajectory dump must decode to a dictionary')
        version = data.get('schema_version')
        if version != SCHEMA_VERSION:
            raise ValueError(f'unsupported trajectory schema_version: {version!r} (expected {SCHEMA_VERSION})')
        kwargs_pool = data.get('kwargs_pool')
        if kwargs_pool is not None and not isinstance(kwargs_pool, list):
            raise TypeError('trajectory kwargs_pool must be a list')
        nodes: dict[NodeId, TrajectoryNode] = {}
        for item in data['nodes']:
            account_data = item.get('token_account')
            account = None
            if account_data is not None:
                account = TokenAccount(
                    token_ids=account_data.get('token_ids'),
                    gen_start=account_data.get('gen_start', 0),
                    gen_len=account_data.get('gen_len', 0),
                    logprobs=account_data.get('logprobs'),
                    topk_indices=account_data.get('topk_indices'),
                    topk_values=account_data.get('topk_values'),
                    routed_experts=account_data.get('routed_experts'),
                    multi_modal=account_data.get('multi_modal'),
                )
            boundary_data = item.get('segment_boundary')
            boundary = None
            if boundary_data is not None:
                boundary = SegmentBoundary(
                    starts_new_segment=boundary_data.get('starts_new_segment', False),
                    kind=boundary_data.get('kind'),
                    causes=tuple(boundary_data.get('causes', ())),
                    evidence=boundary_data.get('evidence', {}),
                )
            provider_data = item.get('provider_facts')
            invocation_data = item.get('invocation')
            kwargs_ref = item.get('request_kwargs_ref')
            if kwargs_ref is not None:
                if kwargs_pool is None or not isinstance(kwargs_ref, int) or not 0 <= kwargs_ref < len(kwargs_pool):
                    raise ValueError(f'invalid request_kwargs_ref: {kwargs_ref!r}')
                kwargs = _copy.deepcopy(kwargs_pool[kwargs_ref])
            else:
                kwargs = item.get('request_kwargs')
            if kwargs is not None and not isinstance(kwargs, dict):
                raise TypeError('trajectory node request_kwargs must be a dictionary')
            node = TrajectoryNode(
                node_id=item['node_id'],
                message_id=item['message_id'],
                parent=item.get('parent'),
                generated_by=item['generated_by'],
                children=list(item['children']),
                original_message=item['original_message'],
                templated_message=item.get('templated_message'),
                request_kwargs=kwargs,
                token_account=account,
                provider_facts=ProviderFacts(**provider_data) if provider_data else None,
                request_message_overrides={
                    override['node_id']: override['message'] for override in item.get('request_message_overrides', [])
                },
                timeline=item.get('timeline', []),
                as_sent_variants=item.get('as_sent_variants', []),
                segment_boundary=boundary,
                selection=item.get('selection'),
                invocation=InvocationFacts(**invocation_data) if invocation_data else None,
                reward=item.get('reward'),
            )
            if node.node_id in nodes:
                raise ValueError(f'duplicate node id {node.node_id}')
            nodes[node.node_id] = node
        chains: dict[ChainId, Chain] = {}
        for item in data['chains']:
            chain_id = item['chain_id']
            if chain_id in chains:
                raise ValueError(f'duplicate chain id {chain_id!r}')
            spawn_data, join_data = item.get('spawn'), item.get('join')
            chains[chain_id] = Chain(
                chain_id=chain_id,
                kind=item.get('kind', 'main'),
                parent_chain_id=item.get('parent_chain_id'),
                roots=list(item.get('roots', [])),
                spawn=SpawnLink(**spawn_data) if spawn_data else None,
                join=JoinLink(**join_data) if join_data else None,
                detector=item.get('detector'),
                evidence=item.get('evidence', {}),
            )
        for node in nodes.values():
            if node.parent is not None and node.parent not in nodes:
                raise ValueError(f'invalid trajectory node {node.node_id}: parent {node.parent} does not exist')
        acyclic: set[NodeId] = set()
        for node_id in nodes:
            trail: list[NodeId] = []
            current: NodeId | None = node_id
            while current is not None and current not in acyclic:
                if current in trail:
                    raise ValueError(f'parent cycle through node {current}')
                trail.append(current)
                current = nodes[current].parent
            acyclic.update(trail)
        declared_children: set[NodeId] = set()
        for node in nodes.values():
            for child_id in node.children:
                if child_id not in nodes:
                    raise ValueError(f'invalid trajectory node {node.node_id}: child {child_id} does not exist')
                if child_id in declared_children or nodes[child_id].parent != node.node_id:
                    raise ValueError(f'invalid trajectory node {node.node_id}: children do not match parent links')
                declared_children.add(child_id)
        for node in nodes.values():
            if node.parent is not None and node.node_id not in declared_children:
                raise ValueError(f'invalid trajectory node {node.node_id}: children do not match parent links')
        for node in nodes.values():
            ancestors = set()
            current = node.parent
            while current is not None:
                ancestors.add(current)
                current = nodes[current].parent
            invalid = set(node.request_message_overrides) - ancestors
            if invalid:
                raise ValueError(f'node {node.node_id} has request overrides outside its ancestors: {sorted(invalid)}')
        roots = list(data.get('roots', []))
        if len(roots) != len(set(roots)):
            raise ValueError('duplicate trajectory root')
        actual_roots = {node.node_id for node in nodes.values() if node.parent is None}
        if set(roots) != actual_roots:
            raise ValueError('trajectory roots do not match parentless nodes')
        for root in roots:
            if root not in nodes:
                raise ValueError(f'root {root} does not exist')
        chain_owners: dict[NodeId, ChainId] = {}
        for chain in chains.values():
            for root in chain.roots:
                if root not in nodes:
                    raise ValueError(f'chain {chain.chain_id!r} root {root} does not exist')
                owner = chain_owners.setdefault(root, chain.chain_id)
                if owner != chain.chain_id or chain.roots.count(root) > 1:
                    raise ValueError(f'chain root {root} belongs to more than one chain entry')
            for relation, link in (('spawn', chain.spawn), ('join', chain.join)):
                if link is not None and link.node is not None and link.node not in nodes:
                    raise ValueError(f'chain {chain.chain_id!r} {relation} node {link.node} does not exist')
        return cls(
            session_id=data['session_id'],
            state=data.get('state', 'completed'),
            nodes=nodes,
            roots=roots,
            chains=chains,
            latest_messages=data.get('latest_messages'),
            timeline=data.get('timeline', []),
            final_reward=data.get('final_reward'),
            termination_reason=data.get('termination_reason'),
            metrics=data.get('metrics', {}),
            extra=data.get('extra', {}),
        )
