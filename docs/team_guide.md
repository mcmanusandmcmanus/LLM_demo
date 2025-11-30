# LLM Web App: Mission and Framework Guide

## Mission
- Give teammates a clear, fast path to experiment with LLMs (prompts, decoding, models) via a web UI and API.
- Support end-to-end iteration: data prep → fine-tuning → evaluation → serving (FastAPI or Django).
- Keep quality measurable with standard metrics (perplexity, ROUGE) and behavioral checks.

## Architecture at a Glance
- **LLM core** (`src/llm_service.py`): Hugging Face text-generation pipeline; configurable via env vars (`MODEL_NAME`, decoding params). Single shared instance per process.
- **FastAPI app** (`src/main.py`): `/generate`, `/health`, serves UI at `src/static/index.html`.
- **Django app** (`llm_site/`): UI at `/`, API at `/api/generate`; reuses the same LLM service.
- **Data + training**: `src/data_utils.py` (clean/dedupe/split), `src/train.py` (fine-tune), sample splits in `data/sample/`.
- **Evaluation**: `src/eval.py` (perplexity, ROUGE-L), `src/generate.py` (CLI prompt helper).
- **Docs**: `docs/llm_ops.md` (ops playbook), `docs/historical_nlp_methods.md` (method history).

## How to Use (happy path)
1) Prep data: `python -m src.data_utils --input_file data/raw.txt --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --lower --dedupe`
2) Train: `python -m src.train --train_file data/processed/train.txt --validation_file data/processed/validation.txt --output_dir artifacts/run-001 --num_train_epochs 1 --batch_size 2 --block_size 128`
3) Evaluate: `python -m src.eval --model_name artifacts/run-001 --test_file data/processed/test.txt`
4) Serve:
   - FastAPI: `MODEL_NAME=artifacts/run-001 uvicorn src.main:app --port 8000`
   - Django: `cd llm_site && MODEL_NAME=artifacts/run-001 python manage.py migrate && python manage.py runserver 8000`
5) UI: hit `/` (FastAPI or Django) to tweak prompts/decoding; API: POST to `/generate` or `/api/generate`.

## Quality & Metrics
- **Training metrics**: loss/val loss; monitor for divergence.
- **Eval metrics**: perplexity on held-out test set; ROUGE-L for generation vs references.
- **Behavioral checks**: keep a fixed prompt set for safety/domain sanity; track pass/fail.
- **Performance**: latency p50/p95, tokens/sec, memory/VRAM; test batch vs single.
- **Regression guard**: re-run `src.eval.py` on the same test set per change; log results.

## Responsibilities & Roles
- **Data**: ensure clean splits, provenance, and license checks; avoid/strip PII.
- **Model**: choose appropriate base checkpoint; set decoding defaults; verify metrics before promotion.
- **App/Infra**: secure endpoints (auth/CSRF/rate limits), configure env vars, set timeouts, and monitor.
- **Ops**: track latency/errors/quality probes in prod; rotate models with model cards and changelogs.

## Security & Safety (prod checklist)
- Turn off `DEBUG`, set a real `SECRET_KEY`, restrict `ALLOWED_HOSTS`.
- Add auth/CSRF (Django) and rate limits; cap `max_new_tokens` and input length.
- Log latency/errors but avoid storing raw user prompts if privacy-sensitive.
- Consider background jobs for long generations or use a dedicated serving stack (vLLM/TGI) for larger models.

## Extending
- Add quantization or smaller checkpoints for CPU deployments.
- Add CI to run `src.eval.py` + smoke prompts on PRs.
- Layer retrieval (RAG) or tools as needed; document new endpoints and metrics alongside changes.
