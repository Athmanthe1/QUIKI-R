# QUIKI-R v1.4 — category suggestions for Wikimedia Commons

Given a new image (and optional title/description), QUIKI-R suggests Commons categories for a **human editor to review**.
It is a retrieval-plus-classifier pipeline over a **frozen** `google/siglip-base-patch16-224` encoder (768-d):

| Scorer | What it is |
|---|---|
| kNN | softmax-weighted vote (τ=0.05) of the labels of the K nearest *train* images; the neighbours are shown as the explanation |
| text | SigLIP pairwise sigmoid between the image and category-name prototypes (3 prompt templates, cached) |
| head | linear multi-label head trained with the problem statement's asymmetric loss (three-state targets) |
| meta | SigLIP text embedding of title + description (optional) |

The scorers are fused by logistic-regression stacking over the union of each scorer's top-200 candidates (missing-scorer flags,
no min-max scaling), then a category-DAG post-process suppresses an ancestor only if an above-threshold descendant is selected.
Problem statement: the supplied PDF is authoritative; where this repo and the PDF disagreed, v1.4 follows the PDF.

> **Status (read this).** The logic is unit-tested (106 tests, run without torch/faiss/network). **No real evaluation numbers exist yet**:
> the build environment had no access to Hugging Face, PyPI or Commons, so SigLIP, FAISS and the MediaWiki API were never executed there.
> Numbers appear in this repo only when `evaluate.py` produces them on real artifacts (see *Metrics*). The live demo shows a
> banner telling you whether it runs on **real artifacts**, the small built-in **demo knowledge base**, or **mock mode**.

## Pipeline

```
build_seed_dataset.py  -> seed/raw_records.jsonl (+ hidden_categories.txt)       MediaWiki API, open licences only
cleaning_and_split.py  -> data/{train,val,test}.parquet                           clean, pHash dedup, temporal + uploader-disjoint split
extract_embeddings.py  -> artifacts/ (manifest.json, index, thumbs, parquet)      streaming SigLIP, embedding dedup, TRAIN-only index
trainer.py             -> artifacts/supervised_head.npz                           ASL head on train embeddings
fit_stacker.py         -> artifacts/stacker.json                                  logistic fusion fitted on the val split
evaluate.py            -> artifacts/eval/eval_report.{json,md}                    temporal hold-out metrics
app.py (Streamlit)     -> Suggest / Review / Train                                reads artifacts/, per-visitor feedback sessions
```

`manifest.json` is the single contract between stages (ordered `classes`, list index = class id, model name, embedding dim, file names,
logit scale/bias). Loading fails loudly on a dimension or shape mismatch. `common.py`, `scoring.py` and `fusion.py` are shared, so
training, evaluation and serving use one code path (`scoring.rank_categories`).

## Run it

**Demo (Codespaces / Streamlit Cloud, CPU).** `pip install -r requirements.txt && streamlit run app.py`.
Without artifacts it uses a small built-in demo taxonomy; this is *not* Commons data and the banner says so.

**Real artifacts.**
```bash
pip install -r requirements.txt -r requirements-dev.txt
python build_seed_dataset.py --out_dir seed --per_category 40          # needs network; --synthetic for plumbing tests only
python cleaning_and_split.py --source seed/raw_records.jsonl --output_dir data --hidden_categories seed/hidden_categories.txt
python extract_embeddings.py --splits_dir data --output_dir artifacts --min_freq 20
python trainer.py --artifacts_dir artifacts --epochs 15
python fit_stacker.py --artifacts_dir artifacts
python evaluate.py --artifacts_dir artifacts --split test --review_csv_out artifacts/eval/review_template.csv
streamlit run app.py                                                    # picks up ./artifacts or $QUIKI_ARTIFACTS_DIR
```
The GitHub Action **build-artifacts** (manual `workflow_dispatch`) runs the same chain on a runner with network access and uploads
`artifacts/`. For the hosted app, keep artifacts in Git LFS or a Hugging Face dataset repo and set `QUIKI_HF_REPO`.
Do not deploy artifacts built with `--encoder fake` (test encoder, not a vision-language model).

**Tests.** `pytest -q` (CI does this). Torch-dependent tests skip when torch is absent.

## Metrics

None are published here. `evaluate.py` reports, on the temporal test split with a **train-only index**:
P@k / R@k / F1@k (k=1,3,5,10) sample-wise and macro; mAP; hierarchy-aware precision (ancestor/descendant = 0.5);
head/torso/tail by train frequency; resolution bands; category type (needs a CSV `category,type`; otherwise reported as *not computed*);
p50/p95 scoring latency, throughput, peak RSS and index size; and a strict **plus** human-reviewed score
(`--review_csv_out` writes a random sample of "wrong" suggestions to label `correct/incorrect/unsure`; `--review_csv_in` computes
`reviewed = strict + (1 − strict)·f`, with a Wilson interval).
Latency excludes image decode and the encoder forward pass unless stated in the report.

## Limitations (honest list)

- Nothing here has been run with the real SigLIP model, FAISS ANN indexes, pyarrow I/O or a live Streamlit server by the author of v1.4; see `V1.4_REPORT.md`.
- A seed dataset of a few thousand images and ~100 categories is a prototype, not Commons scale. Tail and "event/abstract" categories will be weak.
- Stored labels are incomplete; strict scores understate quality. Use the reviewed score.
- The head, text prototypes and metadata scorer inherit SigLIP's biases; no regional/language audit has been done (no language or region labels are collected).
- No blur/JPEG/downscale training augmentation (the head trains on frozen embeddings), no quantisation/ONNX, no multilingual aliases.
- Feedback in the hosted demo is per-visitor and fine-tunes only a session copy of the head; it is logged to JSONL and is not merged into shared knowledge automatically. Retrain offline from reviewed events.
- Suggestions are for human approval; the app does not identify people and should not be used for automatic edits.
