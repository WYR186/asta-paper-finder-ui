from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Unpack, overload

from ai2i.chain.builders import AnyTypedDict
from ai2i.chain.params import (
    DEFAULT_TEMPERATURE,
    APIParamsConverter,
    Gemini3ThinkingParams,
    Gemini25ThinkingParams,
    GPT5MiniReasoningParams,
    GPT51ReasoningParams,
    GPT52ReasoningParams,
    LLMModelParams,
    OpenAINoReasoningParams,
    convert_gemini3_thinking_params,
    convert_gemini25_flash_thinking_params,
    convert_gemini25_pro_thinking_params,
    convert_google_model_params,
    convert_openai_model_params,
    convert_openai_reasoning_params,
)

# Default model names
GPT4_DEFAULT_MODEL = "gpt-4-0613"
GPT4O_DEFAULT_MODEL = "gpt-4o-2024-08-06"
GPT41_DEFAULT_MODEL = "gpt-4.1-2025-04-14"
GPT41MINI_DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
GPT4TURBO_DEFAULT_MODEL = "gpt-4-turbo-2024-04-09"
GEMINI2FLASH_DEFAULT_MODEL = "gemini-2.0-flash-001"
# Reasoning models
GPT52_DEFAULT_MODEL = "gpt-5.2-2025-12-11"
GPT51_DEFAULT_MODEL = "gpt-5.1-2025-11-13"
GPT5MINI_DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
GEMINI25FLASH_DEFAULT_MODEL = "gemini-2.5-flash"
GEMINI3FLASH_DEFAULT_MODEL = "gemini-3-flash-preview"
GEMINI25PRO_DEFAULT_MODEL = "gemini-2.5-pro"
GEMINI3PRO_DEFAULT_MODEL = "gemini-3-pro-preview"


type ModelName = Literal[
    "openai:gpt4-default",
    "openai:gpt41-default",
    "openai:gpt41-mini-default",
    "openai:gpt4o-default",
    "openai:gpt4turbo-default",
    "google:gemini2flash-default",
]

type ReasoningModelName = Literal[
    "openai:gpt52-reasoning-default",
    "openai:gpt52-no-reasoning-default",
    "openai:gpt51-reasoning-default",
    "openai:gpt51-no-reasoning-default",
    "openai:gpt5mini-high-reasoning-default",
    "openai:gpt5mini-medium-reasoning-default",
    "openai:gpt5mini-minimal-reasoning-default",
    "google:gemini25flash-default",
    "google:gemini25pro-default",
    "google:gemini3pro-default",
    "google:gemini3flash-high-reasoning-default",
    "google:gemini3flash-medium-reasoning-default",
]

type ModelFamily = Literal["openai", "google"]


@dataclass(frozen=True)
class LLMModel[P: AnyTypedDict]:
    name: str
    family: ModelFamily
    params: P
    params_converter: APIParamsConverter[P]
    is_reasoning: bool

    @overload
    @staticmethod
    def from_name(name: ModelName, **params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt51-reasoning-default"], **params: Unpack[GPT51ReasoningParams]
    ) -> LLMModel[GPT51ReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt51-no-reasoning-default"], **params: Unpack[OpenAINoReasoningParams]
    ) -> LLMModel[OpenAINoReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt52-no-reasoning-default"], **params: Unpack[OpenAINoReasoningParams]
    ) -> LLMModel[OpenAINoReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt5mini-high-reasoning-default"], **params: Unpack[GPT5MiniReasoningParams]
    ) -> LLMModel[GPT5MiniReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt5mini-default"], **params: Unpack[GPT5MiniReasoningParams]
    ) -> LLMModel[GPT5MiniReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["openai:gpt5mini-minimal-reasoning-default"], **params: Unpack[GPT5MiniReasoningParams]
    ) -> LLMModel[GPT5MiniReasoningParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["google:gemini25flash-default"], **params: Unpack[Gemini25ThinkingParams]
    ) -> LLMModel[Gemini25ThinkingParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["google:gemini25pro-default"], **params: Unpack[Gemini25ThinkingParams]
    ) -> LLMModel[Gemini25ThinkingParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["google:gemini3pro-default"], **params: Unpack[Gemini3ThinkingParams]
    ) -> LLMModel[Gemini3ThinkingParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["google:gemini3flash-high-reasoning-default"], **params: Unpack[Gemini3ThinkingParams]
    ) -> LLMModel[Gemini3ThinkingParams]: ...

    @overload
    @staticmethod
    def from_name(
        name: Literal["google:gemini3flash-medium-reasoning-default"], **params: Unpack[Gemini3ThinkingParams]
    ) -> LLMModel[Gemini3ThinkingParams]: ...

    @staticmethod
    def from_name(name: str, **params: Any) -> LLMModel[Any]:
        match name:
            case "openai:gpt4-default":
                return LLMModel.gpt4(**params)
            case "openai:gpt4o-default":
                return LLMModel.gpt4o(**params)
            case "openai:gpt41-default":
                return LLMModel.gpt41(**params)
            case "openai:gpt41-mini-default":
                return LLMModel.gpt41mini(**params)
            case "openai:gpt4turbo-default":
                return LLMModel.gpt4turbo(**params)
            case "openai:gpt51-reasoning-default":
                return LLMModel.gpt51(**params)
            case "openai:gpt51-no-reasoning-default":
                return LLMModel.gpt51_no_reasoning(**params)
            case "openai:gpt52-reasoning-default":
                return LLMModel.gpt52(**params)
            case "openai:gpt52-no-reasoning-default":
                return LLMModel.gpt52_no_reasoning(**params)
            case "openai:gpt5mini-high-reasoning-default":
                return LLMModel.gpt5mini_high_reasoning(**params)
            case "openai:gpt5mini-medium-reasoning-default":
                return LLMModel.gpt5mini_medium_reasoning(**params)
            case "openai:gpt5mini-minimal-reasoning-default":
                return LLMModel.gpt5mini_minimal_reasoning(**params)
            case "google:gemini2flash-default":
                return LLMModel.gemini2flash(**params)
            case "google:gemini25flash-default":
                return LLMModel.gemini25flash(**params)
            case "google:gemini25pro-default":
                return LLMModel.gemini25pro(**params)
            case "google:gemini3pro-default":
                return LLMModel.gemini3pro(**params)
            case "google:gemini3flash-high-reasoning-default":
                return LLMModel.gemini3flash_high_reasoning(**params)
            case "google:gemini3flash-medium-reasoning-default":
                return LLMModel.gemini3flash_medium_reasoning(**params)
            case _:
                raise ValueError(f"Unrecognized model name: {name}")

    @staticmethod
    def _get_default_llm_params() -> LLMModelParams:
        return {"temperature": DEFAULT_TEMPERATURE}

    @staticmethod
    def gpt4o(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GPT4O_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_openai_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt41(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GPT41_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_openai_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt41mini(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GPT41MINI_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_openai_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt4turbo(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GPT4TURBO_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_openai_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt4(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GPT4_DEFAULT_MODEL,
            family="openai",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_openai_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gemini2flash(**params: Unpack[LLMModelParams]) -> LLMModel[LLMModelParams]:
        return LLMModel(
            name=GEMINI2FLASH_DEFAULT_MODEL,
            family="google",
            params={**LLMModel._get_default_llm_params(), **params},
            params_converter=convert_google_model_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt51(
        **params: Unpack[GPT51ReasoningParams],
    ) -> LLMModel[GPT51ReasoningParams]:
        """Create GPT-5.1 reasoning model with active reasoning.

        Args:
            reasoning_effort: Controls reasoning depth (default: "medium")
                - "low": Light reasoning
                - "medium": Balanced reasoning/speed
                - "high": Maximum reasoning depth
            verbosity: Verbosity level for reasoning output
            max_tokens: Maximum tokens to generate

        Note: Temperature does not exist as a parameter (not configurable when reasoning is active).
        Use gpt51_no_reasoning() if you need temperature control.
        """
        merged_params: GPT51ReasoningParams = {"reasoning_effort": "medium", **params}
        return LLMModel(
            name=GPT51_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=True,
        )

    @staticmethod
    def gpt52(
        **params: Unpack[GPT52ReasoningParams],
    ) -> LLMModel[GPT52ReasoningParams]:
        """Create GPT-5.2 reasoning model with active reasoning.

        Args:
            reasoning_effort: Controls reasoning depth (default: "medium")
                - "low": Light reasoning
                - "medium": Balanced reasoning/speed
                - "high": More reasoning depth
                - "xhigh": Maximum reasoning depth
            verbosity: Verbosity level for reasoning output
            max_tokens: Maximum tokens to generate

        Note: Temperature does not exist as a parameter (not configurable when reasoning is active).
        Use gpt51_no_reasoning() if you need temperature control.
        """
        merged_params: GPT52ReasoningParams = {"reasoning_effort": "medium", **params}
        return LLMModel(
            name=GPT52_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=True,
        )

    @staticmethod
    def gpt51_no_reasoning(
        **params: Unpack[OpenAINoReasoningParams],
    ) -> LLMModel[OpenAINoReasoningParams]:
        """Create GPT-5.1 with reasoning disabled (reasoning_effort='none').

        Args:
            temperature: Sampling temperature (configurable when reasoning is disabled, default: 1.0)
            verbosity: Verbosity level for output
            max_tokens: Maximum tokens to generate

        Note: This disables the reasoning capability and behaves like a standard model.
        Use gpt51() for actual reasoning tasks.
        """
        merged_params: OpenAINoReasoningParams = {"reasoning_effort": "none", **params}
        return LLMModel(
            name=GPT51_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt52_no_reasoning(
        **params: Unpack[OpenAINoReasoningParams],
    ) -> LLMModel[OpenAINoReasoningParams]:
        merged_params: OpenAINoReasoningParams = {"reasoning_effort": "none", **params}
        return LLMModel(
            name=GPT52_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=False,
        )

    @staticmethod
    def gpt5mini_high_reasoning(
        **params: Unpack[GPT5MiniReasoningParams],
    ) -> LLMModel[GPT5MiniReasoningParams]:
        merged_params: GPT5MiniReasoningParams = {"reasoning_effort": "high", **params}
        return LLMModel(
            name=GPT5MINI_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=True,
        )

    @staticmethod
    def gpt5mini_medium_reasoning(
        **params: Unpack[GPT5MiniReasoningParams],
    ) -> LLMModel[GPT5MiniReasoningParams]:
        merged_params: GPT5MiniReasoningParams = {"reasoning_effort": "medium", **params}
        return LLMModel(
            name=GPT5MINI_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=True,
        )

    @staticmethod
    def gpt5mini_minimal_reasoning(
        **params: Unpack[GPT5MiniReasoningParams],
    ) -> LLMModel[GPT5MiniReasoningParams]:
        merged_params: GPT5MiniReasoningParams = {"reasoning_effort": "minimal", **params}
        return LLMModel(
            name=GPT5MINI_DEFAULT_MODEL,
            family="openai",
            params=merged_params,
            params_converter=convert_openai_reasoning_params,
            is_reasoning=True,
        )

    @staticmethod
    def gemini25flash(
        **params: Unpack[Gemini25ThinkingParams],
    ) -> LLMModel[Gemini25ThinkingParams]:
        """Create Gemini 2.5 Flash reasoning model.

        Args:
            thinking_budget: Internal reasoning token budget (default: -1)
                - -1: Dynamic (model decides based on complexity)
                - 0: Disabled (no thinking)
                - 1-24576: Explicit token budget
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
        """
        merged_params: Gemini25ThinkingParams = {"thinking_budget": -1, **params}
        return LLMModel(
            name=GEMINI25FLASH_DEFAULT_MODEL,
            family="google",
            params=merged_params,
            params_converter=convert_gemini25_flash_thinking_params,
            is_reasoning=True,
        )

    @staticmethod
    def gemini25pro(
        **params: Unpack[Gemini25ThinkingParams],
    ) -> LLMModel[Gemini25ThinkingParams]:
        """Create Gemini 2.5 Pro reasoning model.

        Args:
            thinking_budget: Internal reasoning token budget (default: -1)
                - -1: Dynamic (model decides based on complexity)
                - 1-32768: Explicit token budget (higher than Flash's 24576)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate

        Note:
        - Gemini 2.5 Pro has a higher thinking budget limit than Flash (32K vs 24K)
        - Unlike Flash, thinking_budget=0 (disabled) is NOT supported for Pro
        """
        thinking_budget = params.get("thinking_budget", -1)
        if thinking_budget == 0:
            raise ValueError(
                "thinking_budget=0 (disabled) is not supported for Gemini 2.5 Pro. "
                "Thinking cannot be turned off for this model. "
                "Use thinking_budget=-1 for dynamic or 1-32768 for explicit budget."
            )

        merged_params: Gemini25ThinkingParams = {"thinking_budget": -1, **params}
        return LLMModel(
            name=GEMINI25PRO_DEFAULT_MODEL,
            family="google",
            params=merged_params,
            params_converter=convert_gemini25_pro_thinking_params,
            is_reasoning=True,
        )

    @staticmethod
    def gemini3pro(
        **params: Unpack[Gemini3ThinkingParams],
    ) -> LLMModel[Gemini3ThinkingParams]:
        """Create Gemini 3 Pro reasoning model.

        Args:
            thinking_level: Reasoning depth control (default: "high")
                - "low": Minimizes latency/cost, best for simple tasks
                - "high": Maximizes reasoning depth
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate

        IMPORTANT:
        - Cannot use thinking_budget (breaking change from Gemini 2.5)
        - Temperature is always 1.0 (not configurable - Google requirement)
        """
        merged_params: Gemini3ThinkingParams = {"thinking_level": "high", **params}
        return LLMModel(
            name=GEMINI3PRO_DEFAULT_MODEL,
            family="google",
            params=merged_params,
            params_converter=convert_gemini3_thinking_params,
            is_reasoning=True,
        )

    @staticmethod
    def gemini3flash_medium_reasoning(
        **params: Unpack[Gemini3ThinkingParams],
    ) -> LLMModel[Gemini3ThinkingParams]:
        merged_params: Gemini3ThinkingParams = {"thinking_level": "medium", **params}
        return LLMModel(
            name=GEMINI3FLASH_DEFAULT_MODEL,
            family="google",
            params=merged_params,
            params_converter=convert_gemini3_thinking_params,
            is_reasoning=True,
        )

    @staticmethod
    def gemini3flash_high_reasoning(
        **params: Unpack[Gemini3ThinkingParams],
    ) -> LLMModel[Gemini3ThinkingParams]:
        merged_params: Gemini3ThinkingParams = {"thinking_level": "high", **params}
        return LLMModel(
            name=GEMINI3FLASH_DEFAULT_MODEL,
            family="google",
            params=merged_params,
            params_converter=convert_gemini3_thinking_params,
            is_reasoning=True,
        )

    def override(self, updated_params: P) -> LLMModel[P]:
        return LLMModel(self.name, self.family, self.params | updated_params, self.params_converter, self.is_reasoning)

    def to_api_params(self) -> dict[str, Any]:
        return self.params_converter(self.params)
