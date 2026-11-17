"""Real-time chat moderation demo (Gradio): a game-lobby chat with a moderator panel.

Two players (Kerem, blue team, and Elif, red team) chat on one screen. Every
message is labelled live (clean / offensive) with its confidence P(offensive);
flagged messages are masked. The moderator panel logs the response time of
every message, compares all models on the last message and lets the user
switch the moderation model. The default model is the quantized Transformer
(BERTurk, dynamic INT8, ONNX Runtime, CPU).

Run:  python app.py            (add --share for a public link, e.g. in Colab)
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only, like the target server

import argparse
import inspect
import time
from datetime import datetime

import gradio as gr

from src.config import LATENCY_BUDGET_MS
from src.hub import ensure_models
from src.lexicon import Lexicon
from src.predictors import load_all
from src.ui import CSS, PLAYERS, chat_html, header, moderator_html

DEFAULT_MODEL = "BERTurk INT8 (ONNX)"
TZ = os.environ.get("APP_TZ", "Europe/Istanbul")
EXAMPLES = {
    "Kerem": ["gg wp, iyi oyundu kanka", "s4l4k mısın nesin, takımı batırdın",
              "a.p.t.a.l gibi oynama şunu", "bu boss öldürücü derecede zor ama çok güzel"],
    "Elif": ["mid'e geliyorum, beni bekle", "sen ne salak bir oyuncusun ya",
             "takımımız rezil oldu bugün", "tamam sakin ol, bir sonraki maçı alırız"],
}

ensure_models()
MODELS = load_all(include_onnx_fp32=False, include_torch_fp32=True)
if DEFAULT_MODEL not in MODELS:
    raise SystemExit(f"{DEFAULT_MODEL} not found - run the training scripts or set HF_REPO_ID.")
LEX = Lexicon()


def now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(TZ)).strftime("%H:%M:%S")
    except Exception:
        return time.strftime("%H:%M:%S")


def classify(name, text):
    """Returns (is_offensive, P(offensive), latency in ms) for one message."""
    m = MODELS[name]
    t0 = time.perf_counter()
    p = m.predict_proba(text)
    ms = (time.perf_counter() - t0) * 1000
    return p >= m.threshold, p, ms


def masked(text):
    """Star-masked text if the lexicon finds the offensive words, else '' (whole message hidden)."""
    stars = LEX.mask(text)
    return stars if stars != text else ""


def send(player, text, model, compare, chat, log):
    text = (text or "").strip()
    if not text:
        return "", chat_html(chat), chat, log, gr.update()
    off, p, ms = classify(model, text)
    t = now()
    chat = chat + [{"player": player, "time": t, "original": text, "shown": masked(text) if off else "",
                    "off": off, "p": p, "th": MODELS[model].threshold, "ms": ms, "model": model}]
    log = log + [{"time": t, "player": player, "off": off, "p": p, "ms": ms, "model": model}]
    cmp_rows = []
    if compare:
        for name in MODELS:
            o, q, s = classify(name, text)
            cmp_rows.append({"model": name, "off": o, "p": q, "ms": s})
    return "", chat_html(chat), chat, log, moderator_html(log, cmp_rows, model, LATENCY_BUDGET_MS)


def send_kerem(text, model, compare, chat, log):
    return send("Kerem", text, model, compare, chat, log)


def send_elif(text, model, compare, chat, log):
    return send("Elif", text, model, compare, chat, log)


def change_model(model, log):
    return header(model, LATENCY_BUDGET_MS), moderator_html(log, [], model, LATENCY_BUDGET_MS)


def clear(model):
    return chat_html([]), [], [], moderator_html([], [], model, LATENCY_BUDGET_MS)


# CSS / dark mode: newer Gradio versions take these in launch(), older ones in Blocks()
DARK_JS = "() => { document.body.classList.add('dark'); }"
blocks_kw, launch_kw = {}, {}
for key, val in (("css", CSS), ("js", DARK_JS)):
    if key in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw[key] = val
    else:
        launch_kw[key] = val

with gr.Blocks(title="SWE015 - Real-time chat moderation", **blocks_kw) as demo:
    chat_state, log_state = gr.State([]), gr.State([])
    head = gr.HTML(header(DEFAULT_MODEL, LATENCY_BUDGET_MS))
    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            chat_view = gr.HTML(chat_html([]))
            with gr.Row():
                with gr.Column():
                    k_box = gr.Textbox(label=f"Kerem · {PLAYERS['Kerem']['team']}", placeholder="Message as Kerem…")
                    k_btn = gr.Button("Send as Kerem", variant="primary")
                    gr.Examples(EXAMPLES["Kerem"], inputs=[k_box], label="Kerem's examples")
                with gr.Column():
                    e_box = gr.Textbox(label=f"Elif · {PLAYERS['Elif']['team']}", placeholder="Message as Elif…")
                    e_btn = gr.Button("Send as Elif", variant="primary")
                    gr.Examples(EXAMPLES["Elif"], inputs=[e_box], label="Elif's examples")
        with gr.Column(scale=2):
            model_sel = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Moderation model")
            compare = gr.Checkbox(value=True, label="Also run all models on each message")
            mod_view = gr.HTML(moderator_html([], [], DEFAULT_MODEL, LATENCY_BUDGET_MS))
            clear_btn = gr.Button("Clear chat")

    shared_in = [model_sel, compare, chat_state, log_state]
    outs = [None, chat_view, chat_state, log_state, mod_view]
    for box, btn, fn in ((k_box, k_btn, send_kerem), (e_box, e_btn, send_elif)):
        o = [box] + outs[1:]
        box.submit(fn, [box] + shared_in, o)
        btn.click(fn, [box] + shared_in, o)
    model_sel.change(change_model, [model_sel, log_state], [head, mod_view])
    clear_btn.click(clear, [model_sel], [chat_view, chat_state, log_state, mod_view])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.launch(share=a.share, server_port=a.port, **launch_kw)
