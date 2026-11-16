"""Real-time chat moderation demo (Gradio).

Every message is labelled live (clean / offensive) with a confidence score,
flagged messages are masked, and a side panel logs the response time of every
message and lets the user switch between models. The default model is the
quantized Transformer (BERTurk, dynamic INT8, ONNX Runtime, CPU).

Run:  python app.py            (add --share for a public link, e.g. in Colab)
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only, like the target server

import argparse
import html
import time

import gradio as gr
import pandas as pd

from src.config import LATENCY_BUDGET_MS
from src.hub import ensure_models
from src.lexicon import Lexicon
from src.predictors import load_all

DEFAULT_MODEL = "BERTurk INT8 (ONNX)"
LOG_COLS = ["#", "time", "model", "label", "confidence", "latency (ms)", f"≤ {LATENCY_BUDGET_MS:.0f} ms"]
CMP_COLS = ["model", "precision", "label", "confidence", "latency (ms)"]
EXAMPLES = [
    "iyi oyundu kanka, yarın tekrar atalım mı?",
    "sen ne salak bir oyuncusun ya",
    "s4l4k mısın nesin, takımı batırdın",
    "a.p.t.a.l gibi oynama şunu",
    "bu boss öldürücü derecede zor ama çok güzel",
    "takımımız rezil oldu bugün",
]

ensure_models()
MODELS = load_all(include_onnx_fp32=False, include_torch_fp32=True)
if DEFAULT_MODEL not in MODELS:
    raise SystemExit(f"{DEFAULT_MODEL} not found - run the training scripts or set HF_REPO_ID.")
LEX = Lexicon()


def classify(name, text):
    m = MODELS[name]
    t0 = time.perf_counter()
    p = m.predict_proba(text)
    ms = (time.perf_counter() - t0) * 1000
    off = p >= m.threshold
    return off, (p if off else 1 - p), ms


def masked(text):
    stars = LEX.mask(text)
    return stars if stars != text else "█████ (message hidden)"


def bubble(msg):
    color = "#c0392b" if msg["off"] else "#1e8449"
    label = "OFFENSIVE" if msg["off"] else "CLEAN"
    body = html.escape(msg["shown"])
    if msg["off"]:
        body += ("<details style='margin-top:4px;font-size:12px;opacity:.75'><summary>show original</summary>"
                 f"{html.escape(msg['original'])}</details>")
    return (
        "<div style='margin:8px 0;padding:10px 12px;border-radius:12px;"
        f"border-left:5px solid {color};background:rgba(127,127,127,.08)'>"
        f"<div style='font-size:15px'>{body}</div>"
        f"<div style='margin-top:6px;font-size:12px'>"
        f"<span style='background:{color};color:white;padding:2px 8px;border-radius:8px;font-weight:600'>"
        f"{label} {msg['conf']:.0%}</span>"
        f"<span style='opacity:.7'> &nbsp;{msg['ms']:.1f} ms · {html.escape(msg['model'])}</span></div></div>"
    )


def render(chat):
    if not chat:
        return "<div style='opacity:.6;padding:20px'>No messages yet - type something below.</div>"
    return ("<div style='max-height:520px;overflow-y:auto'>"
            + "".join(bubble(m) for m in reversed(chat)) + "</div>")


def stats(log, model):
    ms = [r[5] for r in log if r[2] == model]
    if not ms:
        return f"**Latency budget:** {LATENCY_BUDGET_MS:.0f} ms / message (CPU)"
    avg = sum(ms) / len(ms)
    ok = "within budget ✅" if avg <= LATENCY_BUDGET_MS else "over budget ❌"
    return (f"**Latency budget:** {LATENCY_BUDGET_MS:.0f} ms / message (CPU)  \n"
            f"**{model}:** mean {avg:.1f} ms over {len(ms)} message(s) - {ok}")


def send(text, model, compare, chat, log):
    text = (text or "").strip()
    if not text:
        return "", render(chat), chat, log, pd.DataFrame(log, columns=LOG_COLS), gr.update(), stats(log, model)
    off, conf, ms = classify(model, text)
    chat = chat + [{"original": text, "shown": masked(text) if off else text,
                    "off": off, "conf": conf, "ms": ms, "model": model}]
    log = log + [[len(log) + 1, time.strftime("%H:%M:%S"), model, "OFF" if off else "NOT",
                  round(conf, 3), round(ms, 2), "yes" if ms <= LATENCY_BUDGET_MS else "no"]]
    cmp_rows = []
    if compare:
        for name, m in MODELS.items():
            o, c, t = classify(name, text)
            cmp_rows.append([name, m.precision, "OFF" if o else "NOT", round(c, 3), round(t, 2)])
    return ("", render(chat), chat, log, pd.DataFrame(log[::-1], columns=LOG_COLS),
            pd.DataFrame(cmp_rows, columns=CMP_COLS), stats(log, model))


def clear():
    return render([]), [], [], pd.DataFrame(columns=LOG_COLS), pd.DataFrame(columns=CMP_COLS), stats([], DEFAULT_MODEL)


with gr.Blocks(title="SWE015 - Real-time chat moderation") as demo:
    gr.Markdown("## 🎮 Real-time chat moderation (Turkish)\n"
                "Each message is classified live as **clean** or **offensive**. "
                "Offensive messages are masked. Default model: **BERTurk, dynamic INT8, ONNX Runtime on CPU**.")
    chat_state, log_state = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat_html = gr.HTML(render([]))
            with gr.Row():
                box = gr.Textbox(placeholder="Type a chat message and press Enter…", show_label=False, scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(EXAMPLES, inputs=[box], label="Try these")
        with gr.Column(scale=2):
            model_sel = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Model")
            compare = gr.Checkbox(value=True, label="Also run all models on each message (comparison table)")
            stat_md = gr.Markdown(stats([], DEFAULT_MODEL))
            log_df = gr.Dataframe(pd.DataFrame(columns=LOG_COLS), label="Response-time log (every message)",
                                  interactive=False, wrap=True)
            cmp_df = gr.Dataframe(pd.DataFrame(columns=CMP_COLS), label="All models on the last message",
                                  interactive=False, wrap=True)
            clear_btn = gr.Button("Clear chat")

    io = dict(fn=send, inputs=[box, model_sel, compare, chat_state, log_state],
              outputs=[box, chat_html, chat_state, log_state, log_df, cmp_df, stat_md])
    box.submit(**io)
    send_btn.click(**io)
    model_sel.change(stats, [log_state, model_sel], stat_md)
    clear_btn.click(clear, None, [chat_html, chat_state, log_state, log_df, cmp_df, stat_md])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port)
