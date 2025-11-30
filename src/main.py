from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    STATIC_DIR,
)
from .llm_service import LLMService

app = FastAPI(title="LLM Demo", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLMService()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt to feed into the model.", min_length=1)
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        description="Maximum tokens to generate.",
        ge=8,
        le=512,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="Sampling temperature for randomness.",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        description="Nucleus sampling probability mass.",
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        description="Top-k sampling cutoff (0 disables).",
        ge=0,
        le=200,
    )


class GenerateResponse(BaseModel):
    completion: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest) -> GenerateResponse:
    try:
        completion = llm.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail="Generation failed") from exc

    return GenerateResponse(completion=completion)


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)

    return HTMLResponse("<h1>LLM Demo</h1><p>Static UI not found.</p>")
