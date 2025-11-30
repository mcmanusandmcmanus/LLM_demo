from __future__ import annotations

import json
from typing import Any, Dict

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from config import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P
from llm_service import LLMService


llm = LLMService()


def home(request: HttpRequest) -> HttpResponse:
    return render(request, "playground/index.html")


@csrf_exempt
def generate(request: HttpRequest) -> JsonResponse:
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)
    try:
        payload: Dict[str, Any] = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return JsonResponse({"error": "Prompt is required"}, status=400)

    max_new_tokens = int(payload.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(payload.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(payload.get("top_p", DEFAULT_TOP_P))
    top_k = int(payload.get("top_k", DEFAULT_TOP_K))

    try:
        completion = llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:  # pylint: disable=broad-except
        return JsonResponse({"error": "Generation failed", "details": str(exc)}, status=500)

    return JsonResponse({"completion": completion})
