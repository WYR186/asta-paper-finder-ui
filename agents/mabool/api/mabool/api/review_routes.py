"""
Literature review draft generation.

POST /api/review/generate
  - Accepts a list of paper dicts (title, authors, year, abstract, venue, url)
  - Optional: focus_prompt, language, length
  - Streams the LLM response as SSE

Uses the OPENAI_API_KEY already loaded from .env.secret.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])


# ── Request model ─────────────────────────────────────────────────────────────

class PaperForReview(BaseModel):
    corpus_id: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    url: str | None = None


class ReviewRequest(BaseModel):
    papers: list[PaperForReview]
    focus_prompt: str = ""          # e.g. "focus on methodology comparison"
    language: str = "English"       # "English" | "Chinese" | "Bilingual"
    length: str = "standard"        # "brief" | "standard" | "detailed"


# ── System prompt builder ─────────────────────────────────────────────────────

_LENGTH_GUIDANCE = {
    "brief":    "Write a concise review of ~300 words with a brief intro, key themes, and conclusion.",
    "standard": "Write a thorough review of ~600 words with background, main themes, methodology overview, and conclusion.",
    "detailed": "Write a comprehensive review of ~1200 words covering background, detailed thematic analysis, methodology comparison, gaps, and future directions.",
}

_LANG_INSTRUCTION = {
    "English":   "Write entirely in English.",
    "Chinese":   "Write entirely in Chinese (简体中文).",
    "Bilingual": "Write section headings in both English and Chinese; body text in English.",
}


def _build_system_prompt(req: ReviewRequest) -> str:
    length_guide = _LENGTH_GUIDANCE.get(req.length, _LENGTH_GUIDANCE["standard"])
    lang_guide = _LANG_INSTRUCTION.get(req.language, _LANG_INSTRUCTION["English"])

    return f"""You are an expert academic writing assistant. Your task is to write a literature review draft based on the provided papers.

Structure the review with these sections:
1. **Introduction** – Background and motivation of the research area
2. **Main Themes** – Categorize and synthesize the papers by theme or approach
3. **Methodology Overview** – Compare key methods and techniques used
4. **Key Findings** – Highlight important results and contributions
5. **Research Gaps & Future Directions** – Identify what remains unsolved
6. **Conclusion** – Summarize the state of the field

Rules:
- {length_guide}
- {lang_guide}
- Cite papers inline as (Author et al., Year) or (First Author, Year)
- Be analytical and critical, not just descriptive
- Identify connections and contradictions between papers
- Use academic writing style
"""


def _build_user_prompt(req: ReviewRequest) -> str:
    if req.focus_prompt:
        focus = f"\n\nSpecial focus instruction: {req.focus_prompt}"
    else:
        focus = ""

    papers_text = "\n\n".join(
        f"[{i+1}] {p.title or 'Untitled'} "
        f"({', '.join(p.authors[:3]) + (' et al.' if len(p.authors) > 3 else '') if p.authors else 'Unknown authors'}, {p.year or 'n.d.'})\n"
        f"Venue: {p.venue or 'Unknown'}\n"
        f"Abstract: {(p.abstract or 'No abstract available.')[:600]}"
        for i, p in enumerate(req.papers)
    )

    return f"Please write a literature review based on these {len(req.papers)} papers:{focus}\n\n{papers_text}"


# ── SSE streaming endpoint ────────────────────────────────────────────────────

@router.post("/api/review/generate")
async def generate_review(req: ReviewRequest) -> StreamingResponse:
    """Stream a literature review draft as SSE (event: chunk, data: text)."""

    if not req.papers:
        async def empty() -> AsyncGenerator[str, None]:
            yield "event: error\ndata: " + json.dumps({"message": "No papers provided"}) + "\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    # Cap at 30 papers to avoid token overflows
    papers = req.papers[:30]
    req = ReviewRequest(papers=papers, focus_prompt=req.focus_prompt,
                        language=req.language, length=req.length)

    async def generate() -> AsyncGenerator[str, None]:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            yield "event: error\ndata: " + json.dumps({"message": "OPENAI_API_KEY not configured"}) + "\n\n"
            return

        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
            client = AsyncOpenAI(api_key=api_key)

            system_prompt = _build_system_prompt(req)
            user_prompt = _build_user_prompt(req)

            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                stream=True,
                temperature=0.7,
                max_tokens=3000,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield "event: chunk\ndata: " + json.dumps({"text": delta.content}) + "\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as exc:
            logger.exception("Review generation failed")
            yield "event: error\ndata: " + json.dumps({"message": str(exc)}) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
