import logging
from asyncio import TimeoutError, wait_for

from langchain_classic.chains.moderation import OpenAIModerationChain
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Over 99% of requests complete within 5 seconds,
# for the others, it's better to retry than to wait too long.
moderation_timeout = 5.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def is_dangerous_query(text: str, model_name: str = "omni-moderation-latest") -> bool:
    try:
        moderation_chain = OpenAIModerationChain(model_name=model_name, error=True)

        await wait_for(moderation_chain.ainvoke({"input": text}), timeout=moderation_timeout)
        return False

    except TimeoutError:
        logger.warning(f"Moderation request timed out after {moderation_timeout} seconds for text: {text[:50]}...")
        raise
    except ValueError as e:
        # LangChain raises ValueError when content is flagged
        if "Text was found that violates OpenAI's content policy" in str(e):
            logger.warning(f"Text flagged by moderation: {text[:50]}...")
            return True
        else:
            # Re-raise if it's a different ValueError
            raise
    except Exception as e:
        logger.error(f"Moderation API error: {e}")
        raise
