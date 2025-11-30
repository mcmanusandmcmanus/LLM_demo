# LLM ops playbook

## Data engineering
- Source text → clean → dedupe → split train/val/test → tokenize/batch → train → eval → serve.
- Use `python -m src.data_utils --input_file data/raw.txt --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --lower --dedupe` to produce clean splits.
- Store raw data immutable; version processed splits in `data/processed/` with metadata (source, timestamp, sha256) when scaling up.
- Keep a small smoke set (10–50 lines) to spot regressions quickly and a larger representative validation/test set for true quality checks.
- Track dataset cards (schema, provenance, licenses, known biases) alongside the data; ban PII where possible and add automated scrubbing.

## Training workflow
1) Prepare splits: use `src/data_utils.py` or Hugging Face datasets with `train/validation/test`.  
2) Fine-tune: `python -m src.train --train_file data/processed/train.txt --validation_file data/processed/validation.txt --output_dir artifacts/run-001 --num_train_epochs 1 --batch_size 2 --block_size 128`.  
3) Log metrics: loss, eval loss, learning rate, tokens/sec; store checkpoints every N steps.  
4) Promote: pick best checkpoint by validation loss/perplexity and mark with a model card (hyperparams, data slices, metrics).  
5) Serve: set `MODEL_NAME=artifacts/run-001` in the API/Django app.

## Evaluation and performance metrics
- Perplexity (lower is better): `python -m src.eval --model_name artifacts/run-001 --test_file data/processed/test.txt`.  
- Generation quality: ROUGE-L against references: `python -m src.eval --model_name artifacts/run-001 --prompts_file data/prompts.txt --references_file data/references.txt`.  
- Behavioral checks: keep a curated prompt set (safety, refusal, style, domain questions) and store expected patterns; measure pass/fail.  
- Latency/throughput: record p50/p95 latency, tokens/sec, and GPU/CPU utilization; test both with and without batching.  
- Cost/footprint: VRAM/RAM use, model size on disk, and peak memory during batch=1 and batch>1.  
- Drift/regression: re-run eval on the same test set for every checkpoint and diff metrics to catch regressions early.

## Serving patterns (FastAPI or Django)
- Instantiate the model once per worker (already done) and share it across requests.  
- Use environment variables for model path, decoding defaults, and device selection; keep them in `.env` for deployments.  
- For production: add request rate limiting, logging of prompts/latency (not raw user text if privacy-sensitive), and circuit breakers around long generations.  
- Batch inference where possible; for small models on CPU, consider `torch.compile` or quantized weights (GGUF/`bitsandbytes`).  
- Add health and readiness probes (model loaded flag, test generation under timeout).

## Django-specific best practices
- Start dev server: `cd llm_site && python manage.py migrate && python manage.py runserver 8000`. The LLM UI lives at `/`, API at `/api/generate`.  
- Keep `src/` on `PYTHONPATH` (already configured in `llm_site/settings.py`). Use `MODEL_NAME` env var to swap checkpoints.  
- Move long generations off the request path for heavy models: wrap calls in Celery/RQ and poll; set request timeouts.  
- Add authentication/CSRF for non-demo use; separate public UI from internal admin; log metrics (latency, errors) via middleware.  
- Configure static files and HTTPS for production (Whitenoise/ASGI server like Uvicorn+Daphne or Gunicorn+Uvicorn workers).

## Validation gates to ship a model
- Data QA: schema checks, dedupe %, vocabulary coverage, PII scan.  
- Training QA: loss curves stable, no divergence, checkpoint reproducibility (seeded runs).  
- Eval QA: perplexity <= target, ROUGE/behavioral passes, latency budget met.  
- Release: create a model card (data slices, metrics, constraints) and version the checkpoint + config.  
- Monitoring: track latency, error rate, and response length; add periodic quality probes with fixed prompts.
