"""
SSE streaming endpoint for /api/2/rounds/stream.
Emits real progress events based on LLM call count, then the final result.
"""
import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.outputs import LLMResult, ChatGeneration
from langchain_core.messages import AIMessage

from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.agents.paper_finder.paper_finder_agent import run_agent
from mabool.data_model.agent import AgentOperationMode
from mabool.data_model.ids import generate_conversation_thread_id
from mabool.data_model.rounds import RoundRequest
from mabool.infra.operatives import CompleteResponse, PartialResponse, VoidResponse
from mabool.services.prioritized_task import DEFAULT_PRIORITY
from mabool.utils.dc import DC
from mabool.api.round_v2_routes import (
    MaboolCallbackHandler, mabool_callback_var, round_semaphore, TokenUsage
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rounds"])

# Maps LLM call count thresholds to progress stages
_PROGRESS_STAGES = [
    (1, "analyzing",  "Analyzing your query…"),
    (2, "searching",  "Searching for relevant papers…"),
    (5, "evaluating", "Evaluating paper relevance…"),
    (12, "ranking",   "Ranking and sorting results…"),
]


class ProgressMaboolCallbackHandler(MaboolCallbackHandler):
    """Extends MaboolCallbackHandler with on_llm_start for progress events."""

    def __init__(self, queue: asyncio.Queue) -> None:
        super().__init__()
        self._queue = queue
        self._llm_count = 0

    async def on_llm_start(self, serialized: dict, prompts: list, **kwargs: Any) -> None:
        self._llm_count += 1
        for threshold, stage, message in _PROGRESS_STAGES:
            if self._llm_count == threshold:
                await self._queue.put({"stage": stage, "message": message})
                break


@router.post("/api/2/rounds/stream")
async def stream_round(round_request: RoundRequest) -> StreamingResponse:
    """POST endpoint that returns Server-Sent Events with progress + final result."""
    queue: asyncio.Queue = asyncio.Queue()

    async def run_search() -> None:
        try:
            inp = PaperFinderInput(
                doc_collection=DC.empty(),
                query=round_request.paper_description,
                anchor_corpus_ids=round_request.anchor_corpus_ids or [],
                operation_mode=round_request.operation_mode or AgentOperationMode("infer"),
            )
            thread_id = generate_conversation_thread_id()
            cb = ProgressMaboolCallbackHandler(queue)
            mabool_callback_var.set(cb)
            try:
                async with round_semaphore.priority_context(DEFAULT_PRIORITY):
                    response = await run_agent(inp, thread_id)
            finally:
                mabool_callback_var.set(None)

            await queue.put({"stage": "ranking", "message": "Ranking results…"})

            match response:
                case VoidResponse():
                    await queue.put({"type": "error", "message": response.error.message})
                case CompleteResponse() | PartialResponse():
                    result = response.data.model_dump(mode="json")
                    result["token_breakdown_by_model"] = cb.tokens_by_model
                    result["session_id"] = thread_id
                    await queue.put({"type": "result", "data": result})
                case _:
                    await queue.put({"type": "error", "message": "Unexpected response type"})
        except Exception as e:
            logger.exception("stream_round error")
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)  # sentinel

    asyncio.create_task(run_search())

    async def generate() -> AsyncGenerator[str, None]:
        # Emit immediately so client knows request was received
        yield f"event: progress\ndata: {json.dumps({'stage': 'queued', 'message': 'Request received…'})}\n\n"
        while True:
            item = await queue.get()
            if item is None:
                break
            t = item.get("type")
            if t == "result":
                yield f"event: result\ndata: {json.dumps(item['data'])}\n\n"
            elif t == "error":
                yield f"event: error\ndata: {json.dumps({'message': item['message']})}\n\n"
            else:
                yield f"event: progress\ndata: {json.dumps(item)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
