import random

import pytest

from ai2i.chain.models import LLMModel


@pytest.fixture
def random_float() -> float:
    return random.random()


def test_model_for_name() -> None:
    assert LLMModel.from_name("openai:gpt4-default") == LLMModel.gpt4()
    assert LLMModel.from_name("openai:gpt4turbo-default") == LLMModel.gpt4turbo()
    assert LLMModel.from_name("openai:gpt4o-default") == LLMModel.gpt4o()


def test_overriding_of_params(random_float: float) -> None:
    model = LLMModel.gpt4().override({"temperature": random_float})

    assert model.params.get("temperature") == random_float
    assert model.name == LLMModel.gpt4().name


def test_reasoning_model_gpt51() -> None:
    model = LLMModel.gpt51(reasoning_effort="high")
    assert model.params.get("reasoning_effort") == "high"


def test_reasoning_model_gpt51_no_reasoning() -> None:
    model = LLMModel.gpt51_no_reasoning(temperature=0.5)
    assert model.params.get("reasoning_effort") == "none"
    assert model.params.get("temperature") == 0.5


def test_reasoning_model_gemini25flash() -> None:
    model = LLMModel.gemini25flash(thinking_budget=5000, temperature=0.7)
    assert model.params.get("thinking_budget") == 5000
    assert model.params.get("temperature") == 0.7


def test_reasoning_model_gemini3pro() -> None:
    model = LLMModel.gemini3pro(thinking_level="low")
    assert model.params.get("thinking_level") == "low"


def test_reasoning_model_override_gpt51() -> None:
    """Test that override preserves the generic type for GPT-5.1."""
    model = LLMModel.gpt51(reasoning_effort="medium")
    overridden = model.override({"reasoning_effort": "high", "max_tokens": 1000})

    assert overridden.params.get("reasoning_effort") == "high"
    assert overridden.params.get("max_tokens") == 1000


def test_reasoning_model_override_gemini25flash() -> None:
    """Test that override preserves the generic type for Gemini 2.5 Flash."""
    model = LLMModel.gemini25flash(thinking_budget=1000)
    overridden = model.override({"thinking_budget": 5000, "temperature": 0.8})

    assert overridden.params.get("thinking_budget") == 5000
    assert overridden.params.get("temperature") == 0.8


def test_reasoning_model_gemini25pro_thinking_budget_zero_error() -> None:
    """Test that Gemini 2.5 Pro raises an error when thinking_budget=0."""
    with pytest.raises(ValueError, match="thinking_budget=0.*not supported"):
        LLMModel.gemini25pro(thinking_budget=0)


def test_reasoning_model_gemini25pro_override_thinking_budget_zero_error() -> None:
    """Test that overriding Gemini 2.5 Pro with thinking_budget=0 raises an error."""
    model = LLMModel.gemini25pro(thinking_budget=1000)
    with pytest.raises(ValueError, match="thinking_budget=0.*not supported"):
        model.override({"thinking_budget": 0}).to_api_params()


def test_reasoning_model_from_name() -> None:
    """Test that from_name returns models with correct generic types."""
    gpt51 = LLMModel.from_name("openai:gpt51-reasoning-default", reasoning_effort="high")
    assert gpt51.params.get("reasoning_effort") == "high"

    gemini25 = LLMModel.from_name("google:gemini25flash-default", thinking_budget=5000)
    assert gemini25.params.get("thinking_budget") == 5000

    gemini3 = LLMModel.from_name("google:gemini3pro-default", thinking_level="low")
    assert gemini3.params.get("thinking_level") == "low"
