# LLM Demo

FastAPI + Django playground for running and fine-tuning Hugging Face causal language models. Includes a web UI, API endpoints, data prep utilities, evaluation, and a sample Django site that consumes the same LLM service.

## Project layout
- `src/main.py` – FastAPI server with `/generate`, `/health`, and the bundled UI.
- `src/llm_service.py` – Loads the model and performs text generation; shared by FastAPI and Django.
- `src/train.py` – Fine-tuning loop using Hugging Face `Trainer`.
- `src/data_utils.py` – Clean/dedupe/split raw text into train/val/test.
- `src/eval.py` – Perplexity + ROUGE evaluation helpers.
- `src/generate.py` – CLI helper for a single prompt.
- `llm_site/` – Minimal Django project with a playground UI (`/`) and API (`/api/generate`).
- `data/sample/` – Tiny train/validation/test set for smoke tests.
- `docs/` – LLM ops playbook and historical NLP comparisons.
- `requirements.txt` – Runtime + training dependencies.

## Quickstart (FastAPI + UI)
1) Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate            # PowerShell: .\.venv\Scripts\Activate
```
2) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3) Run the API + UI (defaults to `distilgpt2`):
```bash
uvicorn src.main:app --reload --port 8000
```
4) Open http://localhost:8000 or call the API:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Summarize why transformers matter.","max_new_tokens":64}'
```
5) Switch models by setting an env var before starting:
```bash
export MODEL_NAME="gpt2"             # PowerShell: $env:MODEL_NAME="gpt2"
```
You can point `MODEL_NAME` at a fine-tuned checkpoint directory.

## Django playground
1) Start the Django app (shares the same LLM service):
```bash
cd llm_site
python manage.py migrate
python manage.py runserver 8000
```
2) Visit http://localhost:8000 for the UI; POST to `/api/generate` for the API.  
3) Swap models by exporting `MODEL_NAME` before running `manage.py`.
See `docs/llm_ops.md` for production hardening tips (timeouts, background jobs, auth).

## Data prep (train/validation/test)
- Clean/dedupe/split a raw text file:
```bash
python -m src.data_utils --input_file data/raw.txt --output_dir data/processed --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --lower --dedupe
```
- Sample data lives in `data/sample/` (train/validation/test.txt) for quick smoke tests.

## Fine-tune on your data
```bash
python -m src.train \
  --train_file data/processed/train.txt \
  --validation_file data/processed/validation.txt \
  --output_dir artifacts/run-001 \
  --num_train_epochs 1 \
  --batch_size 2 \
  --block_size 128
```
Serve the fine-tuned model:
```bash
export MODEL_NAME="artifacts/run-001"
uvicorn src.main:app --reload --port 8000
```

## Evaluate quality and performance
- Perplexity on held-out test data:
```bash
python -m src.eval --model_name artifacts/run-001 --test_file data/processed/test.txt
```
- Generation quality (ROUGE-L) given prompts + references:
```bash
python -m src.eval --model_name artifacts/run-001 --prompts_file data/prompts.txt --references_file data/references.txt
```
- Track latency/tokens/sec via your server logs; add a small fixed prompt set to watch regressions.

## Notes and next steps
- Defaults favor small models (`distilgpt2`) for quick iteration; heavier checkpoints need more RAM/GPU or quantization.
- FastAPI serves static UI from `src/static/index.html`; Django ships its own UI at `llm_site/playground/templates/playground/index.html`.
- See `docs/llm_ops.md` for ops guidance and `docs/historical_nlp_methods.md` for era-by-era method comparisons (1990s–today).
