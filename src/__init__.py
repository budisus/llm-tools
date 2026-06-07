"""
llm-tools - Utilities for working with LLMs and AI APIs.
"""

__version__ = "0.1.0"

from .llm_client import (
    OpenAIClient,
    AnthropicClient,
    get_client,
    LLMResponse,
    Message,
)

from .prompt_utils import (
    system_prompt,
    few_shot_prompt,
    chain_of_thought,
    extract_json,
    parse_xml_tags,
    extract_between,
    count_tokens_estimate,
    truncate_to_token_limit,
    PromptTemplate,
    CODE_REVIEW_TEMPLATE,
    SUMMARIZE_TEMPLATE,
    create_react_prompt,
)

__all__ = [
    # Client
    "OpenAIClient",
    "AnthropicClient",
    "get_client",
    "LLMResponse",
    "Message",
    # Prompt utils
    "system_prompt",
    "few_shot_prompt",
    "chain_of_thought",
    "extract_json",
    "parse_xml_tags",
    "extract_between",
    "count_tokens_estimate",
    "truncate_to_token_limit",
    "PromptTemplate",
    "CODE_REVIEW_TEMPLATE",
    "SUMMARIZE_TEMPLATE",
    "create_react_prompt",
]