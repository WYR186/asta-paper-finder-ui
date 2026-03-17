import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypedDict, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import BaseModel, SecretStr
from tenacity import RetryError, stop_after_attempt

from ai2i.chain.builders import define_prompt_llm_call
from ai2i.chain.computation import ChainComputation
from ai2i.chain.endpoints import (
    LLMEndpoint,
    ModelFactory,
    Timeouts,
    default_retry_settings,
)
from ai2i.chain.models import (
    GEMINI2FLASH_DEFAULT_MODEL,
    GPT4_DEFAULT_MODEL,
    GPT41_DEFAULT_MODEL,
    GPT41MINI_DEFAULT_MODEL,
    LLMModel,
    ModelFamily,
)
from ai2i.chain.retry import MaboolTimeout, RacingRetrySettings, TenacityRetrySettings
from ai2i.chain.tests.mocks import MockModelRunnable

# Module paths for mocking factory functions
OPENAI_FACTORY_PATH = "ai2i.chain.endpoints._openai_chat_factory"
GOOGLE_FACTORY_PATH = "ai2i.chain.endpoints._google_chat_factory_with_async"


@dataclass
class CallBehavior:
    delay_seconds: float = 0.0
    return_value: Optional[str] = None
    throw_error: bool = False
    error_message: str = "Mock Error"


@dataclass
class DelayedMockModelRunnable(MockModelRunnable):
    delay_seconds: float = 0.0
    name: str | None = "default"
    call_spy: MagicMock = field(default_factory=MagicMock)

    # For multi-call scenarios - different behavior for each call
    call_behaviors: list[CallBehavior] = field(default_factory=list)
    _call_count: int = field(default=0, init=False)

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    async def ainvoke(self, input: PromptValue, config: RunnableConfig | None = None, **kwargs: Any) -> BaseMessage:
        self.call_spy()  # Track the call
        self._call_count += 1
        self.last_input = input

        # Determine behavior for this call
        if self.call_behaviors and self._call_count <= len(self.call_behaviors):
            # Use call-specific behavior
            behavior = self.call_behaviors[self._call_count - 1]
            delay = behavior.delay_seconds
            return_value = behavior.return_value or getattr(self, "return_value", None)
            should_throw = behavior.throw_error
            error_msg = behavior.error_message
        else:
            # Use default behavior
            delay = self.delay_seconds
            return_value = getattr(self, "return_value", None)
            should_throw = getattr(self, "throw_error_n_times", 0) > 0
            error_msg = f"Mock Error, {getattr(self, 'throw_error_n_times', 0) - 1} throws left"
            if should_throw:
                self.throw_error_n_times -= 1

        if should_throw:
            raise ValueError(error_msg)

        if delay > 0:
            await asyncio.sleep(delay)

        if return_value is not None:
            return BaseMessage(
                content=return_value, type="ChatGeneration", response_metadata=getattr(self, "return_metadata", {})
            )
        else:
            raise ValueError("No return_value defined for this mock")

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def was_called(self) -> bool:
        return self.call_spy.called


class MockModelRegistry:
    def __init__(self):
        self.models: dict[str, DelayedMockModelRunnable] = {}

    def register(
        self,
        model_name: str,
        delay_seconds: float = 0.0,
        return_value: str | None = None,
        call_behaviors: list[CallBehavior] | None = None,
    ) -> DelayedMockModelRunnable:
        mock = DelayedMockModelRunnable(
            name=model_name, delay_seconds=delay_seconds, call_behaviors=call_behaviors or []
        )
        if return_value is not None:
            mock.return_value = return_value
        self.models[model_name] = mock
        return mock

    def get_model(self, model_name: str) -> DelayedMockModelRunnable:
        return self.models[model_name]

    def create_factory(self) -> ModelFactory:
        def model_factory(
            structured_response: bool,
            model: LLMModel[Any],
            timeout: MaboolTimeout,
            api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
        ) -> BaseChatModel:
            if model.name in self.models:
                return cast(BaseChatModel, self.models[model.name])
            else:
                raise ValueError(f"No mock registered for model: {model.name}")

        return model_factory

    def assert_all_called(self, model_names: list[str]) -> None:
        for name in model_names:
            assert self.models[name].was_called, f"Model {name} was not called"

    def get_call_counts(self) -> dict[str, int]:
        return {name: mock.call_count for name, mock in self.models.items()}


class SomeInput(TypedDict):
    input: str


class SomeResponse(BaseModel):
    value: str


@pytest.fixture
def input_value() -> str:
    return f"in1_{uuid.uuid4()}"


@pytest.fixture
def input_value2() -> str:
    return f"in2_{uuid.uuid4()}"


@pytest.fixture
def output_value() -> str:
    return f"out1_{uuid.uuid4()}"


@pytest.fixture
def output_value2() -> str:
    return f"out2_{uuid.uuid4()}"


@pytest.fixture
def api_key_mapper() -> Callable[[ModelFamily], SecretStr | None]:
    return lambda _: SecretStr("API_KEY")


@pytest.fixture
def registry() -> MockModelRegistry:
    return MockModelRegistry()


async def test_run(
    input_value: str, input_value2: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None]
) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    def _pass_through_model_factory(
        structured_response: bool,
        model: LLMModel[Any],
        timeout: MaboolTimeout,
        api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
    ) -> BaseChatModel:
        def _internal(input: StringPromptValue) -> BaseMessage:
            return BaseMessage(content=SomeResponse(value=input.text).model_dump_json(), type="test")

        return cast(BaseChatModel, RunnableLambda(_internal))

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=_pass_through_model_factory,
    )

    r = await endpoint.execute(call).once({"input": input_value})

    assert r.startswith(input_value)  # followed by injected format instructions

    rb = await endpoint.execute(call).many([{"input": input_value}, {"input": input_value2}])

    assert rb[0].startswith(input_value)  # followed by injected format instructions
    assert rb[1].startswith(input_value2)  # followed by injected format instructions


async def test_retries_fail(
    input_value: str, output_value: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None]
) -> None:
    error_throws = 2

    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.throw_error_n_times = error_throws

    endpoint = LLMEndpoint(
        TenacityRetrySettings({**default_retry_settings(), "stop": stop_after_attempt(error_throws)}),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=lambda structured_response, model, timeout, api_key_mapper=lambda _: None: cast(
            BaseChatModel, model_mock
        ),
    )

    with pytest.raises(RetryError):
        await endpoint.execute(call).once({"input": input_value})


async def test_retries_success(
    input_value: str, output_value: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None]
) -> None:
    error_throws = 2

    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    model_mock = MockModelRunnable()
    model_mock.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock.throw_error_n_times = error_throws

    endpoint = LLMEndpoint(
        TenacityRetrySettings({**default_retry_settings(), "stop": stop_after_attempt(error_throws + 1)}),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=lambda structured_response, model, timeout, api_key_mapper=None: cast(BaseChatModel, model_mock),
    )

    r = await endpoint.execute(call).once({"input": input_value})
    assert r == output_value


async def test_retries_multi_call_one_fully_fail(
    input_value: str, output_value: str, output_value2: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None]
) -> None:
    error_throws = 2

    call1 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call2 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call = ChainComputation.map_n(lambda x, y: (x, y), call1, call2)

    model_mock1 = MockModelRunnable()
    model_mock1.return_value = SomeResponse(value=output_value).model_dump_json()

    model_mock2 = MockModelRunnable()
    model_mock2.return_value = SomeResponse(value=output_value2).model_dump_json()
    model_mock2.throw_error_n_times = error_throws

    # NOTE: Each mock appears twice: once for the base model, once for the output-fixing model
    model_mocks = [model_mock1, model_mock1, model_mock2, model_mock2]

    def model_factory(
        structured_response: bool,
        model: LLMModel[Any],
        timeout: MaboolTimeout,
        api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
    ) -> BaseChatModel:
        chosen_model = model_mocks.pop(0)
        return cast(BaseChatModel, chosen_model)

    endpoint = LLMEndpoint(
        TenacityRetrySettings({**default_retry_settings(), "stop": stop_after_attempt(error_throws)}),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=model_factory,
    )

    with pytest.raises(RetryError):
        await endpoint.execute(call).once({"input": input_value})


async def test_retries_multi_call_both_fail_recover(
    input_value: str, output_value: str, output_value2: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None]
) -> None:
    error_throws = 2

    call1 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call2 = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    call = ChainComputation.map_n(lambda x, y: (x, y), call1, call2)

    model_mock1 = MockModelRunnable()
    model_mock1.return_value = SomeResponse(value=output_value).model_dump_json()
    model_mock1.throw_error_n_times = error_throws

    model_mock2 = MockModelRunnable()
    model_mock2.return_value = SomeResponse(value=output_value2).model_dump_json()
    model_mock2.throw_error_n_times = error_throws

    # NOTE: Each mock appears twice: once for the base model, once for the output-fixing model
    model_mocks = [model_mock1, model_mock1, model_mock2, model_mock2]

    def model_factory(
        structured_response: bool,
        model: LLMModel[Any],
        timeout: MaboolTimeout,
        api_key_mapper: Callable[[ModelFamily], SecretStr | None] | None = None,
    ) -> BaseChatModel:
        chosen_model = model_mocks.pop(0)
        return cast(BaseChatModel, chosen_model)

    endpoint = LLMEndpoint(
        TenacityRetrySettings({**default_retry_settings(), "stop": stop_after_attempt(error_throws + 1)}),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=model_factory,
    )

    r = await endpoint.execute(call).once({"input": input_value})
    assert r == (output_value, output_value2)


@pytest.mark.parametrize("cancel_pending", [True, False], ids=["cancel_pending=True", "cancel_pending=False"])
async def test_racing_retry(
    input_value: str,
    output_value: str,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None],
    registry: MockModelRegistry,
    cancel_pending: bool,
) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    # Slow primary model
    registry.register(
        model_name=GPT4_DEFAULT_MODEL,
        delay_seconds=2.5,
        return_value=SomeResponse(value="slow_result").model_dump_json(),
    )

    # Fast fallback model
    registry.register(
        model_name=GPT41MINI_DEFAULT_MODEL,
        delay_seconds=0.1,
        return_value=SomeResponse(value=output_value).model_dump_json(),
    )

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        api_key_mapper=api_key_mapper,
        model_factory=registry.create_factory(),
        default_racing_retry_settings=RacingRetrySettings(
            soft_timeout_ratio=0.1,  # Creates 1.1s timeout (11.0 * 0.1)
            fallback_models=[LLMModel.gpt41mini()],
            cancel_pending=cancel_pending,
        ),
    )

    start_time = time.perf_counter()
    result, elapsed = await endpoint.execute(call).once({"input": input_value}), time.perf_counter() - start_time

    assert registry.get_model(GPT4_DEFAULT_MODEL).was_called, "Primary model should have been invoked"
    assert registry.get_model(GPT41MINI_DEFAULT_MODEL).was_called, "Fallback model should have been invoked"
    assert result == output_value, f"Expected fast model result '{output_value}', got '{result}'"
    assert 1.0 < elapsed < 2.0, f"Expected timing 1.0-2.0s (soft timeout + fast model), got {elapsed:.2f}s"


async def test_racing_retry_no_fallback(
    input_value: str,
    output_value: str,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None],
    registry: MockModelRegistry,
) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    # Set up mock with different behavior for each call
    registry.register(
        model_name=GPT41_DEFAULT_MODEL,
        call_behaviors=[
            CallBehavior(delay_seconds=5.0, return_value=SomeResponse(value="slow_result").model_dump_json()),
            CallBehavior(delay_seconds=0.1, return_value=SomeResponse(value=output_value).model_dump_json()),
        ],
    )

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt41(),
        model_factory=registry.create_factory(),
        api_key_mapper=api_key_mapper,
        default_racing_retry_settings=RacingRetrySettings(
            soft_timeout_ratio=0.1,  # Creates 1.1s timeout (11.0 * 0.1)
            cancel_pending=True,
        ),
    )

    start_time = time.perf_counter()
    result, elapsed = await endpoint.execute(call).once({"input": input_value}), time.perf_counter() - start_time

    model_mock = registry.get_model(GPT41_DEFAULT_MODEL)
    assert model_mock.call_count == 2, "Model should have been invoked twice"
    assert result == output_value, f"Expected result '{output_value}', got '{result}'"
    assert 1.0 < elapsed < 2.0, f"Expected timing 1.0-2.0s, got {elapsed:.2f}s"


async def test_racing_retry_waits_for_pending_when_no_more_fallbacks(
    input_value: str,
    output_value: str,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None],
    registry: MockModelRegistry,
) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    # Primary model - will time out and eventually complete with one result
    registry.register(
        model_name=GPT4_DEFAULT_MODEL,
        delay_seconds=3.0,  # Will soft timeout after 1.2s, and then time out after 2.0s.
        return_value=SomeResponse(value="primary_slow_result").model_dump_json(),
    )

    # Single fallback model - will complete successfully but takes some time
    registry.register(
        model_name=GPT41MINI_DEFAULT_MODEL,
        delay_seconds=1.5,
        return_value=SomeResponse(value=output_value).model_dump_json(),
    )

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.micro,
        default_model=LLMModel.gpt4(),
        model_factory=registry.create_factory(),
        api_key_mapper=api_key_mapper,
        default_racing_retry_settings=RacingRetrySettings(
            soft_timeout_ratio=0.2,  # Creates 1.2s timeout (6.0 * 0.2)
            fallback_models=[LLMModel.gpt41mini()],  # Only one fallback model
            cancel_pending=False,
        ),
    )

    start_time = time.perf_counter()
    result = await endpoint.execute(call).once({"input": input_value})
    elapsed = time.perf_counter() - start_time

    # Should get the faster fallback result
    assert result == output_value, f"Expected fallback result '{output_value}', got '{result}'"

    # Timing should be ~2.7s (1.2s timeout + 1.5s additional for fallback to complete)
    assert 2.5 < elapsed < 3.0, f"Expected timing 2.5-3.0s, got {elapsed:.2f}s"


async def test_racing_retry_returns_last_task_when_all_canceled(
    input_value: str, api_key_mapper: Callable[[ModelFamily], SecretStr | None], registry: MockModelRegistry
) -> None:
    """Test that when all tasks are canceled, we return the last completed task."""
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    # Both models will be relatively fast but we'll set cancel_pending=True
    # so when the first one completes, it will cancel the other
    registry.register(
        model_name=GPT4_DEFAULT_MODEL,
        delay_seconds=1.5,
        return_value=SomeResponse(value="primary_result").model_dump_json(),
    )

    registry.register(
        model_name=GPT41MINI_DEFAULT_MODEL,
        delay_seconds=2.0,
        return_value=SomeResponse(value="fallback_result").model_dump_json(),
    )

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt4(),
        model_factory=registry.create_factory(),
        api_key_mapper=api_key_mapper,
        default_racing_retry_settings=RacingRetrySettings(
            soft_timeout_ratio=0.1,  # Creates 1.1s timeout (11.0 * 0.1)
            fallback_models=[LLMModel.gpt41mini()],
            cancel_pending=True,  # This will cancel pending tasks when one completes
        ),
    )

    start_time = time.perf_counter()
    result = await endpoint.execute(call).once({"input": input_value})
    elapsed = time.perf_counter() - start_time

    # Should get the first successful result (primary completes first)
    assert result == "primary_result", f"Expected primary result, got '{result}'"

    # Timing should be ~1.5s (1s timeout + 0.5s for primary to complete)
    assert 1.0 < elapsed < 2.0, f"Expected timing 1.0-2.0s, got {elapsed:.2f}s"


async def test_racing_retry_multiple_models(
    input_value: str,
    output_value: str,
    api_key_mapper: Callable[[ModelFamily], SecretStr | None],
    registry: MockModelRegistry,
) -> None:
    call = define_prompt_llm_call(
        "{input}",
        input_type=SomeInput,
        output_type=SomeResponse,
    ).map(lambda s: s.value)

    # Primary model - very slow
    registry.register(
        model_name=GPT41_DEFAULT_MODEL,
        delay_seconds=5.0,
        return_value=SomeResponse(value="very_slow_result").model_dump_json(),
    )

    # First fallback - medium speed, also times out
    registry.register(
        model_name=GEMINI2FLASH_DEFAULT_MODEL,
        delay_seconds=2.5,
        return_value=SomeResponse(value="medium_result").model_dump_json(),
    )

    # Second fallback - fast
    registry.register(
        model_name=GPT41MINI_DEFAULT_MODEL,
        delay_seconds=0.1,
        return_value=SomeResponse(value=output_value).model_dump_json(),
    )

    fallback_models: list[LLMModel] = [
        LLMModel.gemini2flash(),
        LLMModel.gpt41mini(),
    ]

    endpoint = LLMEndpoint(
        default_retry_settings(),
        default_timeout=Timeouts.tiny,
        default_model=LLMModel.gpt41(),
        model_factory=registry.create_factory(),
        api_key_mapper=api_key_mapper,
        default_racing_retry_settings=RacingRetrySettings(
            soft_timeout_ratio=0.1,  # Creates 1.1s timeout (11.0 * 0.1)
            fallback_models=fallback_models,
            cancel_pending=True,
        ),
    )

    start_time = time.perf_counter()
    result, elapsed = await endpoint.execute(call).once({"input": input_value}), time.perf_counter() - start_time

    registry.assert_all_called([GPT41_DEFAULT_MODEL, GEMINI2FLASH_DEFAULT_MODEL, GPT41MINI_DEFAULT_MODEL])
    assert result == output_value, f"Expected fast model result '{output_value}', got '{result}'"
    assert 2.0 < elapsed < 3.0, f"Expected timing 2.0-3.0s, got {elapsed:.2f}s"

    call_counts = registry.get_call_counts()
    assert call_counts[GPT41_DEFAULT_MODEL] == 1, "Primary model should be called once"
    assert call_counts[GEMINI2FLASH_DEFAULT_MODEL] == 1, "First fallback should be called once"
    assert call_counts[GPT41MINI_DEFAULT_MODEL] == 1, "Second fallback should be called once"
