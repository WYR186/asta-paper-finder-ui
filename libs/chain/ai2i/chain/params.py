from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from ai2i.chain.builders import AnyTypedDict

# Default LLM parameters
DEFAULT_BATCH_MAX_CONCURRENCY = 5
DEFAULT_TEMPERATURE = 1


class LLMModelParams(TypedDict, total=False):
    temperature: float
    top_p: int
    max_tokens: int


# OpenAI Reasoning Models (GPT-5.1/2)
# Key restriction: NO temperature parameter when reasoning is active
class GPT51ReasoningParams(AnyTypedDict, total=False):
    """Parameters for GPT-5.1 reasoning models with active reasoning.

    Note: temperature is NOT used at all - the parameter doesn't exist for reasoning models.
    Control model behavior via reasoning_effort instead.
    Use gpt51_no_reasoning() if you need temperature control.
    """

    reasoning_effort: Literal["low", "medium", "high"]
    verbosity: Literal["low", "medium", "high"]
    max_tokens: int


class GPT52ReasoningParams(AnyTypedDict, total=False):
    """Parameters for GPT-5.2 reasoning models with active reasoning.

    Note: temperature is NOT used at all - the parameter doesn't exist for reasoning models.
    Control model behavior via reasoning_effort instead.
    Use gpt51_no_reasoning() if you need temperature control.
    """

    reasoning_effort: Literal["low", "medium", "high", "xhigh"]
    verbosity: Literal["low", "medium", "high"]
    max_tokens: int


class GPT5MiniReasoningParams(AnyTypedDict, total=False):
    reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"]
    verbosity: Literal["low", "medium", "high"]
    max_tokens: int


# OpenAI Reasoning Models with reasoning disabled
class OpenAINoReasoningParams(AnyTypedDict, total=False):
    """Parameters for OpenAI reasoning models with reasoning_effort='none'.

    When reasoning is disabled, temperature becomes configurable.
    """

    reasoning_effort: Literal["none"]
    temperature: float
    verbosity: Literal["low", "medium", "high"]
    max_tokens: int


# Gemini 2.5 Reasoning Models (Flash & Pro)
class Gemini25ThinkingParams(TypedDict, total=False):
    """Parameters for Gemini 2.5 reasoning models (Flash and Pro).

    thinking_budget controls internal reasoning:
    - -1: dynamic (model decides based on complexity)
    - 0: disabled (Flash only - Pro doesn't support this)
    - 1-24576: explicit token budget for Flash
    - 1-32768: explicit token budget for Pro
    """

    thinking_budget: int
    temperature: float
    top_p: int
    max_tokens: int


# Gemini 3 Reasoning Models
class Gemini3ThinkingParams(AnyTypedDict, total=False):
    """Parameters for Gemini 3 reasoning models.

    Note:
    - Uses thinking_level instead of thinking_budget (breaking change from 2.5)
    - Cannot use both thinking_level and thinking_budget (API error)
    - Temperature is always 1.0 (not configurable - Google requirement)
    """

    thinking_level: Literal["low", "medium", "high"]
    top_p: int
    max_tokens: int


class APIParamsConverter[P: AnyTypedDict](Protocol):
    def __call__(self, params: P) -> dict[str, Any]: ...


def convert_openai_model_params(params: LLMModelParams) -> dict[str, Any]:
    return {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "max_tokens": params.get("max_tokens"),
        "max_retries": 0,
    }


def convert_google_model_params(params: LLMModelParams) -> dict[str, Any]:
    return {
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "candidate_count": params.get("n"),
        "max_output_tokens": params.get("max_tokens"),
    }


def convert_openai_reasoning_params(
    params: GPT51ReasoningParams | GPT52ReasoningParams | GPT5MiniReasoningParams | OpenAINoReasoningParams,
) -> dict[str, Any]:
    """Convert OpenAI reasoning parameters to ChatOpenAI kwargs.

    Key behavior:
    - If reasoning_effort != "none": temperature is NOT set (parameter doesn't exist)
    - If reasoning_effort == "none": temperature is configurable
    - reasoning_effort controls model behavior
    - verbosity controls output detail level
    """
    reasoning_effort = params.get("reasoning_effort")

    result: dict[str, Any] = {
        "max_retries": 0,
    }

    # Handle temperature based on reasoning_effort
    if reasoning_effort == "none":
        # Temperature is configurable when reasoning is disabled
        result["temperature"] = params.get("temperature", DEFAULT_TEMPERATURE)
    # NOTE: When reasoning is active, temperature is NOT set at all

    if reasoning_effort is not None:
        result["reasoning_effort"] = reasoning_effort
    if "verbosity" in params:
        result["verbosity"] = params["verbosity"]
    if "max_tokens" in params:
        result["max_tokens"] = params["max_tokens"]

    return result


def convert_gemini25_pro_thinking_params(params: Gemini25ThinkingParams) -> dict[str, Any]:
    """Convert Gemini 2.5 thinking parameters to Google GenAI format.

    Key behavior:
    - thinking_budget controls internal reasoning token budget
    - All standard parameters (temperature, top_p) are allowed
    """

    # Gemini 2.5 Pro doesn't support thinking_budget=0
    thinking_budget = params.get("thinking_budget")
    if thinking_budget == 0:
        raise ValueError(
            "thinking_budget=0 (disabled) is not supported for Gemini 2.5 Pro. "
            "Use thinking_budget=-1 for dynamic or 1-32768 for explicit budget."
        )
    return convert_gemini25_flash_thinking_params(params)


def convert_gemini25_flash_thinking_params(params: Gemini25ThinkingParams) -> dict[str, Any]:
    """Convert Gemini 2.5 thinking parameters to Google GenAI format.

    Key behavior:
    - thinking_budget controls internal reasoning token budget
    - All standard parameters (temperature, top_p) are allowed
    """
    result: dict[str, Any] = {}

    # Thinking budget (special for 2.5 models)
    if "thinking_budget" in params:
        result["thinking_budget"] = params["thinking_budget"]

    # Standard parameters
    if "temperature" in params:
        result["temperature"] = params["temperature"]
    if "top_p" in params:
        result["top_p"] = params["top_p"]
    if "max_tokens" in params:
        result["max_output_tokens"] = params["max_tokens"]

    return result


def convert_gemini3_thinking_params(params: Gemini3ThinkingParams) -> dict[str, Any]:
    """Convert Gemini 3 thinking parameters to Google GenAI format.

    Key behavior:
    - thinking_level controls reasoning depth ("low" or "high")
    - CANNOT use thinking_budget (API will error)
    - Temperature is always 1.0 (not configurable - Google requirement)
    """
    result: dict[str, Any] = {
        "temperature": 1.0,  # Always 1.0 for Gemini 3 (Google requirement)
    }

    # Thinking level (replaces thinking_budget in Gemini 3)
    if "thinking_level" in params:
        result["thinking_level"] = params["thinking_level"]

    if "top_p" in params:
        result["top_p"] = params["top_p"]
    if "max_tokens" in params:
        result["max_output_tokens"] = params["max_tokens"]

    return result
