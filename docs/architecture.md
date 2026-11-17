# Architecture

```mermaid
flowchart LR
    U["Player (Kerem or Elif) types a chat message<br/>(Gradio web UI, app.py + src/ui.py)"] --> N

    subgraph PRE["Pre-processing - src/normalize.py (Area B)"]
        N["Regex normalization<br/>leetspeak · separated letters · repeated letters<br/>Turkish lowercase · @USER / URL removal"]
        N --> F["ASCII folding ş→s, ı→i<br/>(classical models only)"]
    end

    F --> M1["1a. Lexicon + min. edit distance<br/>src/lexicon.py · rules"]
    F --> M2["1b. tf-idf (word 1-2 + char 2-5) + LogReg<br/>FP64 · scikit-learn"]
    F --> M3["2. Own word2vec (skip-gram NS) → [mean;max]<br/>→ feedforward NN (SGD) · FP32 · PyTorch"]
    N --> T["BERTurk WordPiece tokenizer<br/>max 128 tokens"]
    T --> M4["3. BERTurk fine-tuned<br/>FP32 · PyTorch"]
    T --> M5["4. BERTurk dynamic INT8<br/>ONNX Runtime · DEFAULT"]

    M1 & M2 & M3 & M4 & M5 --> P["Post-processing<br/>P(offensive) ≥ threshold (tuned on validation)<br/>→ label + confidence"]
    P --> K["Masking: lexicon words → s****<br/>else whole message hidden"]
    K --> UI["Game-lobby chat with label + confidence<br/>moderator panel: latency log, model switch,<br/>all-model comparison"]

    subgraph RUN["Runs on: CPU only (local machine or Colab CPU), weights downloaded from Hugging Face Hub"]
    end
```

**Training (offline, Google Colab T4 GPU):**
`prepare_data.py` → `train_baselines.py` → `train_w2v_ffn.py` → `train_bert.py` → `quantize_bert.py` → `benchmark.py` (CPU) → `upload_models.py`.

| Component | File | Technique area |
|---|---|---|
| Regex normalization, ASCII folding | `src/normalize.py` | B |
| Lexicon matcher, Levenshtein distance | `src/lexicon.py`, `data/lexicon_tr.txt` | B |
| tf-idf + logistic regression | `scripts/train_baselines.py` | B |
| Own word2vec (skip-gram, negative sampling) | `src/w2v_ffn.py` | B (embeddings), A |
| Feedforward network, SGD + backprop, dropout | `src/w2v_ffn.py`, `scripts/train_w2v_ffn.py` | A |
| BERTurk fine-tuning, WordPiece tokenizer | `scripts/train_bert.py` | C |
| ONNX export + dynamic INT8 quantization | `scripts/quantize_bert.py` | D |
| Benchmark: macro-F1, size, latency | `scripts/benchmark.py` | D |
| Common inference interface | `src/predictors.py` | - |
| User interface | `app.py`, `src/ui.py` | - |
