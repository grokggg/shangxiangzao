"""Extended types for multi-provider LLM compatibility.

These types extend OpenAI's standard types with additional fields
needed for compatibility with various LLM providers (Anthropic, DeepSeek, etc.)
through litellm or direct integrations.
"""

from nexus.core.types.base import RobustBaseModel as RobustBaseModel
from nexus.core.types.completion import ExtendedChatCompletion as ExtendedChatCompletion
from nexus.core.types.completion import ExtendedChoice as ExtendedChoice
from nexus.core.types.message import ExtendedMessage as ExtendedMessage
from nexus.core.types.message import RedactedThinkingBlock as RedactedThinkingBlock
from nexus.core.types.message import ThinkingBlock as ThinkingBlock
from nexus.core.types.tokenization import TokenCountResult as TokenCountResult
from nexus.core.types.usage import ExtendedCompletionTokensDetails as ExtendedCompletionTokensDetails
from nexus.core.types.usage import ExtendedPromptTokensDetails as ExtendedPromptTokensDetails
from nexus.core.types.usage import ExtendedUsage as ExtendedUsage
