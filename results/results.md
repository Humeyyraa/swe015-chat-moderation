# Results on the OffensEval-TR 2020 test set (n = 3528)

Hardware: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical cores, CPU only. Latency = mean over 3528 single-message runs (batch size 1, after 10 warm-up runs), end to end. Budget: 50 ms / message.

| Model | Precision | Macro-F1 | Accuracy | F1 (OFF) | Size (MB) | Latency mean (ms) | p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|---|
| Lexicon + edit distance | rules | 0.6618 | 0.8353 | 0.4196 | 0.00 | 3.52 | 8.49 | yes |
| tf-idf + LogReg | FP64 | 0.7695 | 0.8600 | 0.6252 | 2.70 | 1.94 | 2.85 | yes |
| word2vec + FFN | FP32 | 0.6894 | 0.8160 | 0.4910 | 11.51 | 0.40 | 0.63 | yes |
| BERTurk (PyTorch) | FP32 | 0.8089 | 0.8778 | 0.6941 | 422.00 | 126.87 | 184.20 | no |
| BERTurk FP32 (ONNX) | FP32 | 0.8089 | 0.8778 | 0.6941 | 422.17 | 66.92 | 128.40 | no |
| BERTurk INT8 (ONNX) | INT8 (dynamic) | 0.8106 | 0.8784 | 0.6972 | 106.20 | 39.27 | 84.43 | yes |

| Comparison | Same label (%) | Mean abs. P diff | Max abs. P diff |
|---|---|---|---|
| BERTurk (PyTorch) vs BERTurk FP32 (ONNX) | 100.0 | 0.0 | 0.0001 |
| BERTurk FP32 (ONNX) vs BERTurk INT8 (ONNX) | 98.98 | 0.0107 | 0.1891 |
