# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunks of the **GDPR** (EU Regulation 2016/679, 88-page PDF):

- **Pipeline 1 (baseline):** a Bi-Encoder (`all-MiniLM-L6-v2`) with FAISS retrieves the Top-10 chunks; the Top-3 go to the LLM → Answer 1.
- **Pipeline 2 (re-ranking):** a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) re-sorts the same Top-10; the new Top-3 go to the LLM → Answer 2.
- **LLM:** `Qwen2.5-3B-Instruct`, running locally. Both pipelines use the same prompt.

| File | What it is |
|---|---|
| `RAG_Baseline_vs_Reranking.ipynb` | The full RAG code and analysis, with outputs |
| `2025201096_report.pdf` | The report |
| `gdpr.pdf` | The source document (GDPR) |
| `requirements.txt` | Python packages |

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

The first run saves the chunks and the FAISS index to `untracked/cache/`, and later runs load them. To try your own question, add a cell with `try_question("your question")` at the end of the notebook.

The models download from Hugging Face on the first run (about 6.5 GB). The 3B LLM needs about 12 GB RAM, and a full run takes about 15 minutes on a laptop CPU (mostly the LLM writing answers).
