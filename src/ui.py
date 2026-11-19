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
.gradio-container { background:#0F0B1E !important; max-width:100% !important; width:100% !important;
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
.stage { position:relative; overflow:hidden; border:4px solid #F8F8F8; box-shadow:6px 6px 0 #000; height:250px; background:#0F0B1E; }
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
.scene .bob { animation:bob .9s steps(2,end) infinite; }
.scene .bob2 { animation:bob .9s steps(2,end) .45s infinite; }
.scene .flick { animation:flick .7s steps(2,end) infinite; }
.scene .squish { animation:squish 1.1s steps(2,end) infinite; transform-box:fill-box; transform-origin:50% 100%; }
@keyframes bob { 50% { transform:translateY(-6px); } }
@keyframes flick { 50% { opacity:.35; } }
@keyframes squish { 50% { transform:scale(1.1, .85); } }
.hudbar { display:grid; grid-template-columns:1fr 1fr 1.4fr .8fr auto; gap:12px; margin-top:12px; }
.hcell { background:#1D1640; border:3px solid #F8F8F8; box-shadow:4px 4px 0 #000; padding:8px 12px; display:flex; flex-direction:column; justify-content:center; gap:4px; }
.hrow { display:flex; align-items:center; gap:12px; }
.hname { font-size:17px; font-weight:700; }
.hrole { font-size:12px; color:#B8B0D8; font-family:'IBM Plex Mono', monospace; }
.stat { display:grid; grid-template-columns:28px minmax(0,1fr) 64px; gap:8px; align-items:center; font-size:12px; color:#E0DAF5; font-family:'IBM Plex Mono', monospace; }
.gauge { height:10px; background:#0F0B1E; border:2px solid #F8F8F8; }
.gauge i { display:block; height:100%; }
.hlabel { font-size:13px; color:#FFD23F; font-weight:700; letter-spacing:1px; }
.hbig { font-size:17px; color:#F8F8F8; }
.hsmall { font-size:12px; color:#B8B0D8; font-family:'IBM Plex Mono', monospace; }
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


def _sprite_g(player, x, y, scale, cls):
    rows = SPRITES[player]
    rects = "".join(
        f"<rect x='{x + cx * scale}' y='{y + cy * scale}' width='{scale}' height='{scale}' fill='{SPRITE_COLORS[c]}'/>"
        for cy, row in enumerate(rows) for cx, c in enumerate(row) if c != "."
    )
    return f"<g class='{cls}'>{rects}</g>"


def _scene():
    """Pixel-art game scene: night sky, mountains, castle with torches, the two
    party members walking towards a slime, coins on the ground (pure SVG)."""
    W, H, G = 1600, 250, 214   # width, height, ground line
    stars = "".join(f"<rect x='{x}' y='{y}' width='4' height='4' fill='#F8F8F8'/>"
                    for x, y in [(40, 20), (130, 60), (220, 30), (330, 14), (420, 70), (520, 26), (610, 52),
                                 (700, 18), (790, 64), (880, 30), (960, 12), (1050, 58), (1130, 24), (1180, 80),
                                 (1260, 40), (1340, 16), (1420, 66), (1500, 28), (1570, 54)])
    moon = ("<rect x='1180' y='54' width='40' height='40' fill='#FFE9A8'/>"
            "<rect x='1172' y='62' width='8' height='24' fill='#FFE9A8'/>"
            "<rect x='1220' y='62' width='8' height='24' fill='#FFE9A8'/>"
            "<rect x='1200' y='62' width='12' height='12' fill='#F2D27A'/>")
    mountains = (f"<polygon points='0,{G} 140,100 300,{G}' fill='#2A2155'/>"
                 f"<polygon points='200,{G} 380,80 560,{G}' fill='#33296A'/>"
                 f"<polygon points='1040,{G} 1200,96 1360,{G}' fill='#2A2155'/>"
                 f"<polygon points='1280,{G} 1450,110 1600,{G}' fill='#33296A'/>")
    cx = 760   # castle
    castle = (f"<rect x='{cx}' y='120' width='140' height='{G - 120}' fill='#1D1640'/>"
              f"<rect x='{cx - 24}' y='96' width='40' height='{G - 96}' fill='#241B4D'/>"
              f"<rect x='{cx + 124}' y='96' width='40' height='{G - 96}' fill='#241B4D'/>"
              + "".join(f"<rect x='{x}' y='86' width='10' height='10' fill='#241B4D'/>"
                        for x in (cx - 24, cx - 7, cx + 6, cx + 124, cx + 141, cx + 154))
              + "".join(f"<rect x='{x}' y='110' width='14' height='10' fill='#1D1640'/>"
                        for x in (cx + 18, cx + 46, cx + 74, cx + 102))
              + f"<rect x='{cx + 54}' y='{G - 46}' width='32' height='46' fill='#0F0B1E'/>"
              f"<g class='flick'><rect x='{cx - 12}' y='130' width='12' height='16' fill='#FFD23F'/>"
              f"<rect x='{cx + 136}' y='130' width='12' height='16' fill='#FFD23F'/>"
              f"<rect x='{cx + 40}' y='{G - 60}' width='8' height='12' fill='#FF9F1C'/>"
              f"<rect x='{cx + 92}' y='{G - 60}' width='8' height='12' fill='#FF9F1C'/></g>")
    coins = "".join(f"<rect x='{x}' y='{G - 14}' width='10' height='10' fill='#FFD23F'/>"
                    f"<rect x='{x + 3}' y='{G - 11}' width='4' height='4' fill='#B4410F'/>"
                    for x in (560, 600, 640, 1020, 1060))
    slime = (f"<g class='squish'><rect x='1130' y='{G - 30}' width='44' height='30' fill='#3BD16F'/>"
             f"<rect x='1124' y='{G - 18}' width='56' height='18' fill='#3BD16F'/>"
             f"<rect x='1140' y='{G - 22}' width='6' height='8' fill='#0F0B1E'/>"
             f"<rect x='1158' y='{G - 22}' width='6' height='8' fill='#0F0B1E'/>"
             f"<rect x='1136' y='{G - 28}' width='8' height='4' fill='#9DFFB5'/></g>")
    heroes = _sprite_g("Kerem", 420, G - 48, 6, "bob") + _sprite_g("Elif", 490, G - 48, 6, "bob2")
    ground = (f"<rect x='0' y='{G + 4}' width='{W}' height='{H - G - 4}' fill='#1F5F33'/>"
              f"<rect x='0' y='{G}' width='{W}' height='4' fill='#3BD16F'/>"
              + "".join(f"<rect x='{x}' y='{G + 14}' width='16' height='6' fill='#174A28'/>" for x in range(20, W, 90)))
    return (f"<svg class='scene' viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMax slice' "
            "shape-rendering='crispEdges' aria-hidden='true'>"
            f"<rect width='{W}' height='{H}' fill='#151036'/><rect width='{W}' height='110' fill='#1B1446'/>"
            f"{stars}{moon}{mountains}{castle}{coins}{slime}{heroes}{ground}</svg>")


HUD = {"Kerem": {"hp": (82, 100), "mp": (40, 60)}, "Elif": {"hp": (64, 80), "mp": (95, 120)}}


def _gauge(label, cur, mx, color):
    return (f"<div class='stat'><span>{label}</span><div class='gauge'>"
            f"<i style='width:{cur / mx * 100:.0f}%;background:{color}'></i></div><span>{cur}/{mx}</span></div>")


def _minimap():
    cells = ["....#....", ".###.###.", ".#.....#.", ".#..K..#.", ".###E###.", "....#...."]
    colors = {"#": "#4A3F8C", "K": PLAYERS["Kerem"]["color"], "E": PLAYERS["Elif"]["color"]}
    rects = "".join(f"<rect x='{x * 8}' y='{y * 8}' width='8' height='8' fill='{colors[c]}'/>"
                    for y, row in enumerate(cells) for x, c in enumerate(row) if c != ".")
    return (f"<svg width='72' height='48' viewBox='0 0 72 48' shape-rendering='crispEdges' aria-label='minimap'>"
            f"<rect width='72' height='48' fill='#0F0B1E'/>{rects}</svg>")


def header(model_name, budget_ms):
    """Top of the screen: the PIXEL KEEP game scene and a full-width game HUD strip."""
    cards = "".join(
        f"<div class='hcell'><div class='hrow'>{sprite(n, 4)}<div style='flex:1'>"
        f"<div class='px hname' style='color:{PLAYERS[n]['color']}'>{esc(n.upper())} "
        f"<span class='hrole'>{esc(PLAYERS[n]['role'])} · Lv 12 · in chat</span></div>"
        f"{_gauge('HP', *HUD[n]['hp'], '#FF4F5E')}{_gauge('MP', *HUD[n]['mp'], '#4CC9F0')}"
        "</div></div></div>"
        for n in GAME["party"]
    )
    quest = ("<div class='hcell'><div class='px hlabel'>QUEST</div>"
             "<div class='px hbig'>Defeat the Crystal Golem</div>"
             f"{_gauge('', 2, 3, '#FFD23F').replace('<span></span>', '<span>◆</span>')}"
             "<div class='hsmall'>2 of 3 crystals found</div></div>")
    gold = ("<div class='hcell'><div class='px hlabel'>GOLD</div><div class='px hbig'>◆ 1,240</div>"
            "<div class='hsmall'>run time 12:47</div></div>")
    mini = f"<div class='hcell' style='align-items:center'><div class='px hlabel'>MAP</div>{_minimap()}</div>"
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
        f"<div class='hudbar'>{cards}{quest}{gold}{mini}</div>"
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
