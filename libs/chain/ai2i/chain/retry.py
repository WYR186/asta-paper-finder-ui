from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import get_config_list
from langchain_core.runnables.utils import gather_with_concurrency
from tenacity import RetryCallState, RetryError, retry
from tenacity.asyncio.retry import RetryBaseT as AsyncRetryBaseT
from tenacity.retry import RetryBaseT, retry_if_exception
from tenacity.stop import StopBaseT
from tenacity.wait import WaitBaseT

from ai2i.chain.models import LLMModel

logger = logging.getLogger(__name__)


class TenacityRetrySettings(dict):
    sleep: Callable[[int | float], Awaitable[None] | None]
    stop: StopBaseT
    wait: WaitBaseT
    retry: RetryBaseT | AsyncRetryBaseT
    before: Callable[[RetryCallState], Awaitable[None] | None]
    after: Callable[[RetryCallState], Awaitable[None] | None]
    before_sleep: Callable[[RetryCallState], Awaitable[None] | None] | None
    reraise: bool
    retry_error_cls: type[RetryError]
    retry_error_callback: Callable[[RetryCallState], Awaitable[Any] | Any] | None


class RetryWithTenacity[IN, OUT](Runnable[IN, OUT]):
    _decorated: Runnable[IN, OUT]
    _retry_settings: TenacityRetrySettings
    _should_retry_on_timeout: bool

    def __init__(
        self, decorated: Runnable[IN, OUT], retry_settings: TenacityRetrySettings, should_retry_on_timeout: bool = True
    ) -> None:
        self._decorated = decorated
        self._should_retry_on_timeout = should_retry_on_timeout

        original_retry = retry_settings.get("retry")

        if should_retry_on_timeout:
            self._retry_settings = TenacityRetrySettings(retry_settings)
        else:
            no_timeout_retry = retry_if_exception(lambda e: not is_timeout_error(e))
            retry_condition = no_timeout_retry & original_retry if original_retry else no_timeout_retry
            self._retry_settings = TenacityRetrySettings(**retry_settings, retry=retry_condition)

        super().__init__()

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        return f"{self._decorated.get_name(suffix='(retry)', name=name)}"

    async def ainvoke(self, input: IN, config: RunnableConfig | None = None, **kwargs: Any) -> OUT:
        @retry(**self._retry_settings)
        async def invoke_with_retry() -> OUT:
            return await self._decorated.ainvoke(input, config, **kwargs)

        return await invoke_with_retry()

    def invoke(self, input: IN, config: RunnableConfig | None = None, **kwargs: Any) -> OUT:
        raise NotImplementedError(f"No support for blocking calls in {self.__class__.__name__}")

    async def abatch(
        self,
        inputs: list[IN],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[OUT]:
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        async def ainvoke(input: IN, config: RunnableConfig) -> OUT | Exception:
            if return_exceptions:
                try:
                    return await self.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await self.ainvoke(input, config, **kwargs)

        coros = map(ainvoke, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    def batch(
        self,
        inputs: list[IN],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[OUT]:
        raise NotImplementedError(f"No support for blocking calls in {self.__class__.__name__}")


@dataclass
class RacingRetrySettings:
    soft_timeout_ratio: float = 0.25
    fallback_models: list[LLMModel] = field(default_factory=list)
    model_factory: Callable[[LLMModel], Runnable] | None = None
    cancel_pending: bool = True


class RacingRetryWithTenacity[IN, OUT](RetryWithTenacity[IN, OUT]):
    def __init__(
        self,
        decorated: Runnable[IN, OUT],
        decorated_timeout: float,
        retry_settings: TenacityRetrySettings,
        racing_settings: RacingRetrySettings,
        fallback_decorated: list[Runnable[IN, OUT]] | None = None,
    ):
        super().__init__(decorated, retry_settings, should_retry_on_timeout=False)
        self._decorated_timeout = decorated_timeout
        self._racing_settings = racing_settings

        self._fallback_decorated = [
            RetryWithTenacity(fallback, retry_settings, should_retry_on_timeout=False)
            for fallback in (fallback_decorated or [])
        ]

    async def ainvoke(self, input: IN, config: RunnableConfig | None = None, **kwargs: Any) -> OUT:
        soft_timeout = self._decorated_timeout * self._racing_settings.soft_timeout_ratio
        return await self._race_calls(input, config, soft_timeout, **kwargs)

    async def _race_calls(self, input: IN, config: RunnableConfig | None, soft_timeout: float, **kwargs: Any) -> OUT:
        result, original_task = await self._try_original_with_timeout(input, config, soft_timeout, **kwargs)
        if result is not None:
            return result

        result, remaining_tasks = await self._race_with_fallbacks(original_task, input, config, soft_timeout, **kwargs)
        if result is not None:
            return result

        return await self._wait_for_remaining_tasks(remaining_tasks, original_task)

    async def _try_original_with_timeout(
        self, input: IN, config: RunnableConfig | None, soft_timeout: float, **kwargs: Any
    ) -> tuple[OUT | None, asyncio.Task]:
        original_task = asyncio.create_task(super().ainvoke(input, config, **kwargs))
        try:
            result = await asyncio.wait_for(asyncio.shield(original_task), timeout=soft_timeout)
            return result, original_task
        except asyncio.TimeoutError:
            logger.info(f"Call exceeded soft timeout of {soft_timeout}sec, starting racing calls with fallback models")
            return None, original_task

    async def _race_with_fallbacks(
        self, original_task: asyncio.Task, input: IN, config: RunnableConfig | None, soft_timeout: float, **kwargs: Any
    ) -> tuple[OUT | None, list[asyncio.Task]]:
        models = self._fallback_decorated or [
            self._decorated  # Fallback to the original model if no fallbacks are provided.
        ]
        active_tasks = [original_task]

        for i, model in enumerate(models):
            active_tasks.append(asyncio.create_task(model.ainvoke(input, config, **kwargs)))

            timeout = soft_timeout if i < len(models) - 1 else None
            done, pending = await self._wait_for_completion(active_tasks, timeout)

            completed_task = self._get_first_completed_result(done, pending)
            if completed_task:
                return await completed_task, []

            active_tasks = list(pending)
            if i < len(models) - 1 and not done:
                logger.info(f"Fallback model {model.get_name()}#{i + 1} timed out, trying next fallback model")

        return None, active_tasks

    async def _wait_for_remaining_tasks(self, active_tasks: list[asyncio.Task], fallback_task: asyncio.Task) -> OUT:
        logger.info(f"No more fallback models, waiting for remaining {len(active_tasks)} pending tasks")
        while active_tasks:
            done, pending = await self._wait_for_completion(active_tasks, None)
            successful_task = self._get_first_completed_result(done, pending)
            if successful_task:
                return await successful_task
            active_tasks = list(pending)

        return await fallback_task

    async def _wait_for_completion(
        self, active_tasks: list[asyncio.Task], timeout: float | None
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        return await asyncio.wait(active_tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    def _get_first_completed_result(
        self, done_tasks: set[asyncio.Task], pending_tasks: set[asyncio.Task]
    ) -> asyncio.Task | None:
        for task in done_tasks:
            if self._is_completed_task(task):
                if self._racing_settings.cancel_pending:
                    for t in pending_tasks:
                        t.cancel()
                return task
        return None

    def _is_completed_task(self, task: asyncio.Task) -> bool:
        return not task.cancelled() and not (task.exception() and is_timeout_error(task.exception()))


def is_timeout_error(error: BaseException | None) -> bool:
    if error is None:
        return False

    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
        return True

    httpx_timeouts = ["ReadTimeout", "WriteTimeout", "ConnectTimeout", "PoolTimeout"]
    if any(hasattr(httpx, timeout) and isinstance(error, getattr(httpx, timeout)) for timeout in httpx_timeouts):
        return True

    error_module = getattr(error, "__module__", "") or ""
    error_type_name = type(error).__name__.lower()
    if any(provider in error_module and "timeout" in error_type_name for provider in ["openai", "google"]):
        return True

    timeout_indicators = [
        "timeout",
        "timed out",
        "deadline exceeded",
        "request timeout",
        "read timeout",
        "connection timeout",
        "gateway timeout",
        "server timeout",
    ]
    return any(indicator in str(error).lower() for indicator in timeout_indicators)


@dataclass(frozen=True)
class MaboolTimeout:
    httpx_timeout: httpx.Timeout

    def total_seconds(self) -> float | None:
        total = 0.0
        if self.httpx_timeout.connect is not None:
            total += self.httpx_timeout.connect
        if self.httpx_timeout.read is not None:
            total += self.httpx_timeout.read
        if self.httpx_timeout.write is not None:
            total += self.httpx_timeout.write

        if total == 0.0:
            return None
        return total
