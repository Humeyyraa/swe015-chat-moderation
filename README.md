# SWE015 – Real-Time Chat Moderation (Use Case 4)

A gaming platform must flag offensive Turkish chat messages within a strict
time limit on a **CPU-only** server. This repository contains the full model
ladder and a Gradio chat application that labels every message live
(clean / offensive) with a confidence score, masks flagged messages, logs the
response time of every message and lets the user switch between models.

**Latency budget:** 50 ms per message on CPU.
**Model in the application:** fine-tuned BERTurk, dynamic INT8 quantization, ONNX Runtime.

| # | Model | Area |
|---|---|---|
| 1a | Lexicon + minimum edit distance (rule baseline) | B |
| 1b | tf-idf (word + char n-grams) + logistic regression | B |
| 2 | Own word2vec (skip-gram, negative sampling) + feedforward network trained with SGD | A |
| 3 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned for classification | C |
| 4 | BERTurk, ONNX Runtime, dynamic INT8 quantization | D |

All models are evaluated on the **same held-out test set** (official
OffensEval-TR 2020 test set, 3,528 tweets). Results: [`results/results.md`](results/results.md).
Architecture: [`docs/architecture.md`](docs/architecture.md).

## Run the application

Requires Python 3.10–3.12.

```bash
git clone https://github.com/Humeyyraa/swe015-chat-moderation.git
cd swe015-chat-moderation
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On the first start the trained weights (a few hundred MB)
are downloaded from the Hugging Face Hub:
https://huggingface.co/humeyrakoctemiz/swe015-chat-moderation

## Reproduce training and the results table

Open `notebooks/colab_pipeline.ipynb` in Google Colab (Runtime → T4 GPU) and
run all cells, or run the scripts in order:

```bash
python scripts/prepare_data.py     # download OffensEval-TR, train/val/test splits
python scripts/train_baselines.py  # 1a lexicon, 1b tf-idf + LogReg
python scripts/train_w2v_ffn.py    # 2  word2vec + feedforward network
python scripts/train_bert.py       # 3  BERTurk fine-tuning (GPU)
python scripts/quantize_bert.py    # 4  ONNX export + dynamic INT8
python scripts/benchmark.py        # results/results.md (CPU only)
python scripts/upload_models.py    # push weights to the HF Hub
```

## Data

* **Train / validation:** official OffensEval-TR 2020 training file (31,756
  tweets), split 90 / 10 stratified (seed 42).
* **Test:** official OffensEval-TR 2020 test set (3,528 tweets) with gold labels.
* The decision threshold of every model is tuned on the validation set only.

Source: Çağrı Çöltekin (2020). *A Corpus of Turkish Offensive Language on
Social Media.* Proceedings of LREC 2020, pp. 6174–6184.
https://coltekin.github.io/offensive-turkish/ – licence CC-BY 2.0.

**Warning:** the dataset and the demo contain offensive language.

## Repository layout

```
app.py                  Gradio application
src/normalize.py        regex normalization (leetspeak, repeats, Turkish chars)
src/lexicon.py          lexicon matcher + Levenshtein distance
src/w2v_ffn.py          own word2vec + feedforward network
src/predictors.py       one inference interface for all models (app + benchmark)
src/metrics.py          macro-F1, threshold tuning
scripts/                data preparation, training, quantization, benchmark
data/lexicon_tr.txt     hand-written offensive-word lexicon
results/                results table, chart, failure examples
docs/architecture.md    architecture diagram
```
