"""HTML rendering for the retro game chat and the moderator console (used by app.py).

The demo is framed as PIXEL KEEP, a fictional 8-bit co-op dungeon game: two
party members (Kerem the knight, Elif the mage) chat while every message is
checked by the moderation model. Pure functions: they take plain Python data
and return HTML strings, so they can be tested without Gradio or any model.
"""
import html

PLAYERS = {
    "Kerem": {"role": "Knight", "color": "#4CC9F0"},
    "Elif": {"role": "Mage", "color": "#C77DFF"},
}
SHORT = {"Lexicon + edit distance": "LEXICON", "tf-idf + LogReg": "TF-IDF+LR",
         "word2vec + FFN": "W2V+FFN", "BERTurk (PyTorch)": "BERT FP32",
         "BERTurk INT8 (ONNX)": "BERT INT8"}
GAME = {"name": "PIXEL KEEP", "level": "LEVEL 3-2", "stage": "CRYSTAL CAVES",
        "hiscore": "048200", "party": ["Kerem", "Elif"]}

# 8x8 pixel sprites: one character per pixel, "." is transparent
SPRITES = {
    "Kerem": ["...YY...", "..GGGG..", ".GGGGGG.", ".GDDDDG.", ".GGGGGG.", ".GGDDGG.", "..GGGG..", ".CCCCCC."],
    "Elif": ["...PP...", "..PPPP..", ".PPPPPP.", "PPPPPPPP", ".SSSSSS.", ".SDSSDS.", ".SSSSSS.", "..WWWW.."],
}
SPRITE_COLORS = {"Y": "#FFD23F", "G": "#C0C7D6", "D": "#1D1640", "C": "#4CC9F0",
                 "P": "#9D4EDD", "S": "#F4C7A1", "W": "#F8F8F8"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Pixelify+Sans:wght@400..700&family=IBM+Plex+Mono:wght@400;600&display=swap');
.gradio-container { background:#0F0B1E !important; max-width:1400px !important;
  font-family:'IBM Plex Mono', 'Courier New', monospace !important; }
.gradio-container label, .gradio-container .label-wrap span {
  font-family:'Pixelify Sans', 'IBM Plex Mono', monospace !important; font-size:17px !important; }
.gradio-container button { font-family:'IBM Plex Mono', 'Courier New', monospace !important; font-size:15px !important; }
.gradio-container textarea, .gradio-container input[type=text] {
  font-family:'IBM Plex Mono', 'Courier New', monospace !important; font-size:16px !important;
  background:#120D26 !important; color:#F8F8F8 !important; border:3px solid #F8F8F8 !important; border-radius:0 !important; }
.gradio-container .retro-btn { font-family:'Pixelify Sans', 'IBM Plex Mono', monospace !important; font-size:18px !important;
  font-weight:700 !important; background:#FFD23F !important; color:#1D1640 !important; border-radius:0 !important;
  border:3px solid #F8F8F8 !important; box-shadow:4px 4px 0 #000 !important; }
.retro-btn:active { transform:translate(2px,2px); box-shadow:2px 2px 0 #000 !important; }

.px { font-family:'Pixelify Sans', 'IBM Plex Mono', monospace; }
.vt { font-family:'IBM Plex Mono', 'Courier New', monospace; }
.stage { position:relative; overflow:hidden; border:4px solid #F8F8F8; box-shadow:6px 6px 0 #000; height:190px; background:#0F0B1E; }
.stage svg.scene { position:absolute; left:0; top:0; width:100%; height:100%; }
.stage .crt { position:absolute; inset:0; pointer-events:none;
  background:repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px, rgba(0,0,0,.22) 3px); }
.stage .over { position:absolute; inset:0; display:flex; justify-content:space-between; align-items:flex-start; padding:18px 22px; }
.title { font-family:'Press Start 2P', 'Pixelify Sans', monospace; font-size:34px; color:#FFD23F;
  text-shadow:4px 4px 0 #B4410F, 8px 8px 0 #000; line-height:1.1; }
.subtitle { font-size:16px; color:#F8F8F8; margin-top:14px; letter-spacing:1px; text-shadow:2px 2px 0 #000; }
.hud { display:flex; flex-direction:column; align-items:flex-end; gap:8px; font-size:16px; color:#F8F8F8; text-shadow:2px 2px 0 #000; }
.hearts { color:#FF4F5E; font-size:20px; letter-spacing:4px; }
.modon { background:#3BD16F; color:#0F0B1E; text-shadow:none; padding:4px 10px; border:3px solid #F8F8F8; font-weight:700;
  animation:blink 1.2s steps(2,start) infinite; }
@keyframes blink { 50% { opacity:.55; } }
.party { display:flex; gap:12px; margin-top:12px; }
.member { display:flex; align-items:center; gap:10px; background:#1D1640; border:3px solid #F8F8F8; padding:6px 12px; }
.member .nm { font-size:17px; font-weight:700; }
.member .rl { font-size:13px; color:#B8B0D8; }

.tabs { display:flex; gap:8px; margin:14px 0 8px; }
.tab { font-size:15px; padding:6px 12px; border:3px solid #F8F8F8; color:#B8B0D8; background:#1D1640; }
.tab.on { background:#FFD23F; color:#1D1640; font-weight:700; }
.chatbox { height:470px; overflow-y:auto; display:flex; flex-direction:column-reverse; gap:12px; padding:14px;
  background:#120D26; border:4px solid #F8F8F8; box-shadow:6px 6px 0 #000; }
.msg { display:flex; gap:12px; align-items:flex-start; }
.avatar { background:#2A2155; border:3px solid #F8F8F8; padding:4px; line-height:0; }
.dlg { flex:1; background:#1D1640; border:3px solid #F8F8F8; box-shadow:inset 0 0 0 3px #1D1640, inset 0 0 0 5px #4A3F8C; padding:10px 14px; }
.dlg.flag { border-color:#FF4F5E; box-shadow:inset 0 0 0 3px #1D1640, inset 0 0 0 5px #7A2B3A; }
.who { font-size:17px; font-weight:700; margin-bottom:4px; }
.who .t { color:#8F86B8; font-family:'IBM Plex Mono', monospace; font-weight:400; font-size:13px; margin-left:8px; }
.text { font-size:17px; color:#F8F8F8; line-height:1.45; word-break:break-word; }
.hidden { font-size:15px; font-weight:700; color:#FF6B78; margin:2px 0 4px; }
.meta { margin-top:8px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:13px; color:#B8B0D8; }
.tag { font-family:'Pixelify Sans', monospace; font-weight:700; font-size:13px; padding:2px 8px; border:2px solid #F8F8F8; }
.tag.off { background:#FF4F5E; color:#0F0B1E; }
.tag.ok { background:#3BD16F; color:#0F0B1E; }
details.orig summary { cursor:pointer; color:#B8B0D8; font-size:13px; }
details.orig div { color:#E0DAF5; font-size:15px; }
.sys { font-size:15px; color:#FFD23F; padding:8px 12px; border:3px dashed #FFD23F; background:#1A1530; }

.console { background:#050805; border:4px solid #F8F8F8; box-shadow:6px 6px 0 #000; padding:14px 16px;
  font-family:'IBM Plex Mono', 'Courier New', monospace; color:#3BD16F; }
.console h4 { font-family:'Pixelify Sans', monospace; font-size:16px; color:#FFD23F; margin:0 0 10px 0; font-weight:700; }
.console .sec { margin-top:16px; }
.counters { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.ctr { border:2px solid #1F5F33; padding:6px 8px; }
.ctr .v { font-size:24px; font-weight:600; color:#9DFFB5; line-height:1.1; }
.ctr .l { font-size:12px; color:#3BD16F; }
.line { display:grid; grid-template-columns:72px 50px 40px 46px minmax(0,1fr); gap:8px; font-size:14px; padding:3px 0; white-space:nowrap; }
.line.hd { color:#7FBF8F; border-bottom:1px dashed #1F5F33; }
.line .bad, .cline .bad { color:#FF6B6B; }
.cline { display:grid; grid-template-columns:86px 40px 46px minmax(0,1fr); gap:8px; font-size:14px; padding:3px 0; white-space:nowrap; }
.cline.hd { color:#7FBF8F; border-bottom:1px dashed #1F5F33; }
.cline.active { color:#9DFFB5; font-weight:600; }
.bars { overflow:hidden; }
.scroll { max-height:220px; overflow-y:auto; overflow-x:hidden; }
.muted { color:#3E7A50; font-size:14px; }
"""


def esc(s):
    return html.escape(str(s))


def sprite(player, scale=4):
    rows = SPRITES.get(player, SPRITES["Kerem"])
    rects = "".join(
        f"<rect x='{x * scale}' y='{y * scale}' width='{scale}' height='{scale}' fill='{SPRITE_COLORS[c]}'/>"
        for y, row in enumerate(rows) for x, c in enumerate(row) if c != "."
    )
    n = 8 * scale
    return (f"<svg width='{n}' height='{n}' viewBox='0 0 {n} {n}' shape-rendering='crispEdges' "
            f"aria-label='{esc(player)}'>{rects}</svg>")


def _scene():
    """Pixel-art night scene: stars, moon, mountains, castle, ground (pure SVG shapes)."""
    stars = "".join(f"<rect x='{x}' y='{y}' width='4' height='4' fill='#F8F8F8'/>"
                    for x, y in [(40, 20), (130, 60), (220, 30), (330, 14), (420, 70), (520, 26), (610, 52),
                                 (700, 18), (790, 64), (880, 30), (960, 12), (1050, 58), (1130, 24), (1180, 80)])
    moon = ("<rect x='840' y='20' width='40' height='40' fill='#FFE9A8'/>"
            "<rect x='832' y='28' width='8' height='24' fill='#FFE9A8'/>"
            "<rect x='880' y='28' width='8' height='24' fill='#FFE9A8'/>"
            "<rect x='860' y='28' width='12' height='12' fill='#F2D27A'/>")
    mountains = ("<polygon points='0,190 120,90 230,190' fill='#2A2155'/>"
                 "<polygon points='160,190 300,70 450,190' fill='#33296A'/>"
                 "<polygon points='760,190 900,80 1040,190' fill='#2A2155'/>"
                 "<polygon points='980,190 1100,100 1200,190' fill='#33296A'/>")
    castle = ("<rect x='560' y='100' width='120' height='90' fill='#1D1640'/>"
              "<rect x='540' y='80' width='36' height='110' fill='#241B4D'/>"
              "<rect x='664' y='80' width='36' height='110' fill='#241B4D'/>"
              + "".join(f"<rect x='{x}' y='72' width='10' height='10' fill='#241B4D'/>"
                        for x in (540, 556, 566, 664, 680, 690))
              + "".join(f"<rect x='{x}' y='92' width='12' height='10' fill='#1D1640'/>" for x in (576, 600, 624, 648))
              + "<rect x='608' y='150' width='24' height='40' fill='#0F0B1E'/>"
              "<rect x='552' y='110' width='10' height='14' fill='#FFD23F'/>"
              "<rect x='678' y='110' width='10' height='14' fill='#FFD23F'/>")
    ground = ("<rect x='0' y='176' width='1200' height='14' fill='#1F5F33'/>"
              "<rect x='0' y='172' width='1200' height='4' fill='#3BD16F'/>")
    return ("<svg class='scene' viewBox='0 0 1200 190' preserveAspectRatio='xMidYMid slice' "
            "shape-rendering='crispEdges' aria-hidden='true'>"
            "<rect width='1200' height='190' fill='#151036'/><rect width='1200' height='90' fill='#1B1446'/>"
            f"{stars}{moon}{mountains}{castle}{ground}</svg>")


def header(model_name, budget_ms):
    """Top of the screen: the PIXEL KEEP title scene, HUD and party bar."""
    members = "".join(
        f"<div class='member'>{sprite(n, 3)}<div><div class='px nm' style='color:{PLAYERS[n]['color']}'>"
        f"{esc(n.upper())}</div><div class='vt rl'>{esc(PLAYERS[n]['role'])} · in chat</div></div></div>"
        for n in GAME["party"]
    )
    return (
        "<div class='stage'>"
        f"{_scene()}<div class='crt'></div>"
        "<div class='over'>"
        f"<div><div class='px title'>{GAME['name']}</div>"
        f"<div class='px subtitle'>CO-OP DUNGEON RUN · {GAME['level']} · {GAME['stage']}</div></div>"
        "<div class='px hud'>"
        "<div class='hearts'>♥♥♥</div>"
        f"<div>HI-SCORE {GAME['hiscore']}</div>"
        "<div class='modon'>MOD: ON</div>"
        f"<div class='vt' style='font-size:13px'>{esc(model_name)} · ≤ {budget_ms:.0f} ms · CPU</div>"
        "</div></div></div>"
        f"<div class='party'>{members}</div>"
    )


def _tag(off):
    return "<span class='tag off'>OFFENSIVE</span>" if off else "<span class='tag ok'>CLEAN</span>"


def bubble(m):
    p = PLAYERS.get(m["player"], {"role": "", "color": "#F8F8F8"})
    who = (f"<div class='px who' style='color:{p['color']}'>{esc(m['player'].upper())}"
           f"<span class='t'>{esc(p['role'])} · {esc(m['time'])}</span></div>")
    if m["off"]:
        body = "<div class='px hidden'>⚠ HIDDEN BY MODERATION</div>"
        if m["shown"]:
            body += f"<div class='vt text'>{esc(m['shown'])}</div>"
        body += (f"<details class='orig vt'><summary>show original</summary>"
                 f"<div>{esc(m['original'])}</div></details>")
    else:
        body = f"<div class='vt text'>{esc(m['original'])}</div>"
    meta = (f"<div class='vt meta'>{_tag(m['off'])}"
            f"<span>P(offensive) = {m['p']:.2f} · threshold {m['th']:.2f}</span>"
            f"<span>· {m['ms']:.1f} ms · {esc(SHORT.get(m['model'], m['model']))}</span></div>")
    return (f"<div class='msg'><div class='avatar'>{sprite(m['player'])}</div>"
            f"<div class='dlg{' flag' if m['off'] else ''}'>{who}{body}{meta}</div></div>")


def chat_html(chat):
    system = ("<div class='vt sys'>★ SYSTEM: Party entered the Crystal Caves. "
              "Chat moderation is ON: offensive messages are hidden.</div>")
    # column-reverse keeps the newest message at the bottom and in view
    inner = "".join(bubble(m) for m in reversed(chat)) + system
    tabs = "<div class='tabs px'><span class='tab on'>PARTY CHAT</span><span class='tab'>WORLD</span></div>"
    return f"{tabs}<div class='chatbox'>{inner}</div>"


def _bar(ms, budget, width=8):
    """Text latency bar: one block per 1/width of the budget, capped at the budget."""
    n = max(0, min(width, round(ms / budget * width)))
    return "▮" * n + "▯" * (width - n)


def _cls(bad):
    return " class='bad'" if bad else ""


def moderator_html(log, cmp_rows, model_name, budget):
    n = len(log)
    flagged = sum(1 for r in log if r["off"])
    mine = [r["ms"] for r in log if r["model"] == model_name]
    mean = sum(mine) / len(mine) if mine else None
    within = sum(1 for r in log if r["ms"] <= budget)
    counters = (
        "<div class='counters'>"
        f"<div class='ctr'><div class='v'>{n:03d}</div><div class='l'>MESSAGES</div></div>"
        f"<div class='ctr'><div class='v'>{flagged:03d}</div><div class='l'>FLAGGED</div></div>"
        f"<div class='ctr'><div class='v'>{'---' if mean is None else f'{mean:.1f}'}</div><div class='l'>AVG MS</div></div>"
        f"<div class='ctr'><div class='v'>{'---' if not n else f'{within}/{n}'}</div><div class='l'>≤ {budget:.0f} MS</div></div>"
        "</div>"
    )
    rows = "".join(
        f"<div class='line'><span>{esc(r['time'])}</span><span>{esc(r['player'][:5].upper())}</span>"
        f"<span{_cls(r['off'])}>{'OFF' if r['off'] else 'OK'}</span><span>{r['p']:.2f}</span>"
        f"<span class='bars'><span{_cls(r['ms'] > budget)}>{_bar(r['ms'], budget)}</span> {r['ms']:.1f}ms</span></div>"
        for r in reversed(log)
    ) or "<div class='muted'>&gt; waiting for messages_</div>"
    log_block = (
        f"<div class='sec'><h4>&gt; LOG · EVERY MESSAGE · FULL BAR = {budget:.0f} MS</h4>"
        "<div class='line hd'><span>TIME</span><span>WHO</span><span>FLAG</span><span>P(OFF)</span>"
        "<span>LATENCY</span></div>"
        f"<div class='scroll'>{rows}</div></div>"
    )
    if cmp_rows:
        crow = "".join(
            f"<div class='cline{' active' if c['model'] == model_name else ''}'>"
            f"<span>{esc(SHORT.get(c['model'], c['model']))}</span>"
            f"<span{_cls(c['off'])}>{'OFF' if c['off'] else 'OK'}</span><span>{c['p']:.2f}</span>"
            f"<span class='bars'><span{_cls(c['ms'] > budget)}>{_bar(c['ms'], budget)}</span> {c['ms']:.1f}ms</span></div>"
            for c in cmp_rows
        )
        cmp_block = ("<div class='sec'><h4>&gt; ALL MODELS · LAST MESSAGE</h4>"
                     "<div class='cline hd'><span>MODEL</span><span>FLAG</span><span>P(OFF)</span><span>LATENCY</span></div>"
                     f"{crow}</div>")
    else:
        cmp_block = ""
    return (f"<div class='console'><h4>&gt; MOD CONSOLE · {esc(SHORT.get(model_name, model_name))}</h4>"
            f"{counters}{log_block}{cmp_block}</div>")
