"""HTML rendering for the retro game chat and the moderator console (used by app.py).

The demo is framed as PIXEL KEEP, a fictional 8-bit co-op dungeon game: two
party members (Kerem the knight, Elif the mage) chat while every message is
checked by the moderation model. Pure functions: they take plain Python data
and return HTML strings, so they can be tested without Gradio or any model.
"""
import html
import os
from string import Template

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


# ---------------------------------------------------------------- colour themes
# Every colour of the UI by its job. "night" is the original purple night look;
# "sakura" is the pink sunset look. Pick one with:  APP_THEME=night python app.py
THEMES = {
    "night": {
        "bg": "#0F0B1E", "panel": "#1D1640", "panel2": "#120D26", "line": "#F8F8F8", "text": "#F8F8F8",
        "muted": "#B8B0D8", "soft": "#E0DAF5", "dim": "#8F86B8", "accent": "#FFD23F", "accent_ink": "#1D1640",
        "title": "#FFD23F", "title_shadow": "#B4410F", "sys_bg": "#1A1530", "inner": "#4A3F8C", "avatar_bg": "#2A2155",
        "ok": "#3BD16F", "bad": "#FF4F5E", "bad_text": "#FF6B78", "bad_inner": "#7A2B3A",
        "con_bg": "#050805", "con_text": "#3BD16F", "con_hi": "#9DFFB5", "con_line": "#1F5F33", "con_hd": "#7FBF8F",
        "con_muted": "#3E7A50", "con_bad": "#FF6B6B", "con_title": "#FFD23F", "hp": "#FF4F5E", "mp": "#4CC9F0",
        "sky": "#151036", "sky2": "#1B1446", "sky3": "#20184F", "star": "#F8F8F8", "moon": "#FFE9A8", "moon2": "#F2D27A",
        "mtn1": "#2A2155", "mtn2": "#33296A", "castle": "#1D1640", "tower": "#241B4D", "door": "#0F0B1E",
        "window": "#FFD23F", "torch": "#FF9F1C", "ground": "#1F5F33", "ground_top": "#3BD16F", "ground_dash": "#174A28",
        "coin": "#FFD23F", "coin_dot": "#B4410F", "slime": "#3BD16F", "slime_hi": "#9DFFB5", "eye": "#0F0B1E",
        "map_wall": "#4A3F8C", "petal": None,
    },
    "sakura": {
        "bg": "#1F0E1C", "panel": "#3A1835", "panel2": "#2A1027", "line": "#FFE6F2", "text": "#FFF4FA",
        "muted": "#E6B3CF", "soft": "#FFD9EC", "dim": "#C08AA8", "accent": "#FF8FC7", "accent_ink": "#2A1027",
        "title": "#FFB3D9", "title_shadow": "#8E2A64", "sys_bg": "#34142F", "inner": "#8E3F78", "avatar_bg": "#4A1F44",
        "ok": "#3BD16F", "bad": "#FF4F5E", "bad_text": "#FF7A86", "bad_inner": "#7A2B3A",
        "con_bg": "#12060F", "con_text": "#FF9ED0", "con_hi": "#FFD1E8", "con_line": "#6E2A5C", "con_hd": "#D98AB6",
        "con_muted": "#8A4A74", "con_bad": "#FF6B6B", "con_title": "#FFC2E0", "hp": "#FF4F5E", "mp": "#4CC9F0",
        "sky": "#B04A7E", "sky2": "#5C1E5A", "sky3": "#2A0E33", "star": "#FFE6F2", "moon": "#FFD6E8", "moon2": "#FFB3D1",
        "mtn1": "#7A2E66", "mtn2": "#93407A", "castle": "#4A1F44", "tower": "#5A2652", "door": "#1F0E1C",
        "window": "#FFD23F", "torch": "#FF9F1C", "ground": "#6B2A5C", "ground_top": "#FF8FC7", "ground_dash": "#55204A",
        "coin": "#FFD23F", "coin_dot": "#B4410F", "slime": "#3BD16F", "slime_hi": "#9DFFB5", "eye": "#1F0E1C",
        "map_wall": "#8E3F78", "petal": "#FFB7D5",
    },
}
THEME_NAME = os.environ.get("APP_THEME", "sakura")
T = THEMES.get(THEME_NAME, THEMES["sakura"])

_CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Pixelify+Sans:wght@400..700&family=IBM+Plex+Mono:wght@400;600&display=swap');
.gradio-container { background:$bg !important; max-width:100% !important; width:100% !important;
  font-family:'IBM Plex Mono', 'Courier New', monospace !important; }
.gradio-container label, .gradio-container .label-wrap span {
  font-family:'Pixelify Sans', 'IBM Plex Mono', monospace !important; font-size:17px !important; }
.gradio-container button { font-family:'IBM Plex Mono', 'Courier New', monospace !important; font-size:15px !important; }
.gradio-container textarea, .gradio-container input[type=text] {
  font-family:'IBM Plex Mono', 'Courier New', monospace !important; font-size:16px !important;
  background:$panel2 !important; color:$text !important; border:3px solid $line !important; border-radius:0 !important; }
.gradio-container .retro-btn { font-family:'Pixelify Sans', 'IBM Plex Mono', monospace !important; font-size:18px !important;
  font-weight:700 !important; background:$accent !important; color:$accent_ink !important; border-radius:0 !important;
  border:3px solid $line !important; box-shadow:4px 4px 0 #000 !important; }
.retro-btn:active { transform:translate(2px,2px); box-shadow:2px 2px 0 #000 !important; }

.px { font-family:'Pixelify Sans', 'IBM Plex Mono', monospace; }
.vt { font-family:'IBM Plex Mono', 'Courier New', monospace; }
.stage { position:relative; overflow:hidden; border:4px solid $line; box-shadow:6px 6px 0 #000; height:380px; background:$bg; }
.stage svg.scene { position:absolute; left:0; top:0; width:100%; height:100%; }
.stage .crt { position:absolute; inset:0; pointer-events:none;
  background:repeating-linear-gradient(to bottom, rgba(0,0,0,0) 0px, rgba(0,0,0,0) 2px, rgba(0,0,0,.22) 3px); }
.stage .over { position:absolute; inset:0; display:flex; justify-content:space-between; align-items:flex-start; padding:18px 22px; }
.title { font-family:'Press Start 2P', 'Pixelify Sans', monospace; font-size:44px; color:$title;
  text-shadow:4px 4px 0 $title_shadow, 8px 8px 0 #000; line-height:1.1; }
.subtitle { font-size:18px; color:$text; margin-top:14px; letter-spacing:1px; text-shadow:2px 2px 0 #000; }
.hud { display:flex; flex-direction:column; align-items:flex-end; gap:8px; font-size:16px; color:$text; text-shadow:2px 2px 0 #000; }
.hearts { color:$bad; font-size:20px; letter-spacing:4px; }
.modon { background:$ok; color:$bg; text-shadow:none; padding:4px 10px; border:3px solid $line; font-weight:700;
  animation:blink 1.2s steps(2,start) infinite; }
@keyframes blink { 50% { opacity:.55; } }
.scene .bob { animation:bob .9s steps(2,end) infinite; }
.scene .bob2 { animation:bob .9s steps(2,end) .45s infinite; }
.scene .flick { animation:flick .7s steps(2,end) infinite; }
.scene .squish { animation:squish 1.1s steps(2,end) infinite; transform-box:fill-box; transform-origin:50% 100%; }
@keyframes bob { 50% { transform:translateY(-6px); } }
@keyframes flick { 50% { opacity:.35; } }
.scene .petal { animation:fall 6s linear infinite; }
@keyframes fall { from { transform:translate(0,-40px); } to { transform:translate(-60px,260px); } }
@keyframes squish { 50% { transform:scale(1.1, .85); } }
.hudbar { display:grid; grid-template-columns:1.35fr 1.35fr 1.2fr .7fr auto; gap:12px; margin-top:12px; }
.hcell { background:$panel; border:3px solid $line; box-shadow:4px 4px 0 #000; padding:14px 16px; display:flex; flex-direction:column; justify-content:center; gap:4px; }
.hrow { display:flex; align-items:center; gap:12px; }
.hname { font-size:22px; font-weight:700; margin-bottom:4px; }
.hrole { font-size:14px; color:$muted; font-family:'IBM Plex Mono', monospace; }
.stat { display:grid; grid-template-columns:30px minmax(0,1fr) 76px; gap:10px; align-items:center; font-size:14px; margin-top:2px; color:$soft; font-family:'IBM Plex Mono', monospace; }
.gauge { height:14px; background:$bg; border:2px solid $line; }
.gauge i { display:block; height:100%; }
.hlabel { font-size:15px; color:$accent; font-weight:700; letter-spacing:1px; }
.hbig { font-size:22px; color:$text; }
.hsmall { font-size:14px; color:$muted; font-family:'IBM Plex Mono', monospace; }
.tabs { display:flex; gap:8px; margin:14px 0 8px; }
.tab { font-size:15px; padding:6px 12px; border:3px solid $line; color:$muted; background:$panel; }
.tab.on { background:$accent; color:$accent_ink; font-weight:700; }
.chatbox { height:470px; overflow-y:auto; display:flex; flex-direction:column-reverse; gap:12px; padding:14px;
  background:$panel2; border:4px solid $line; box-shadow:6px 6px 0 #000; }
.msg { display:flex; gap:12px; align-items:flex-start; }
.avatar { background:$avatar_bg; border:3px solid $line; padding:4px; line-height:0; }
.dlg { flex:1; background:$panel; border:3px solid $line; box-shadow:inset 0 0 0 3px $panel, inset 0 0 0 5px $inner; padding:10px 14px; }
.dlg.flag { border-color:$bad; box-shadow:inset 0 0 0 3px $panel, inset 0 0 0 5px $bad_inner; }
.who { font-size:17px; font-weight:700; margin-bottom:4px; }
.who .t { color:$dim; font-family:'IBM Plex Mono', monospace; font-weight:400; font-size:13px; margin-left:8px; }
.text { font-size:17px; color:$text; line-height:1.45; word-break:break-word; }
.hidden { font-size:15px; font-weight:700; color:$bad_text; margin:2px 0 4px; }
.meta { margin-top:8px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; font-size:13px; color:$muted; }
.tag { font-family:'Pixelify Sans', monospace; font-weight:700; font-size:13px; padding:2px 8px; border:2px solid $line; }
.tag.off { background:$bad; color:#14080F; }
.tag.ok { background:$ok; color:#14080F; }
details.orig summary { cursor:pointer; color:$muted; font-size:13px; }
details.orig div { color:$soft; font-size:15px; }
.sys { font-size:15px; color:$accent; padding:8px 12px; border:3px dashed $accent; background:$sys_bg; }

.console { background:$con_bg; border:4px solid $line; box-shadow:6px 6px 0 #000; padding:14px 16px;
  font-family:'IBM Plex Mono', 'Courier New', monospace; color:$con_text; }
.console h4 { font-family:'Pixelify Sans', monospace; font-size:16px; color:$con_title; margin:0 0 10px 0; font-weight:700; }
.console .sec { margin-top:16px; }
.counters { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
.ctr { border:2px solid $con_line; padding:6px 8px; }
.ctr .v { font-size:24px; font-weight:600; color:$con_hi; line-height:1.1; }
.ctr .l { font-size:12px; color:$con_text; }
.line { display:grid; grid-template-columns:72px 50px 40px 46px minmax(0,1fr); gap:8px; font-size:14px; padding:3px 0; white-space:nowrap; }
.line.hd { color:$con_hd; border-bottom:1px dashed $con_line; }
.line .bad, .cline .bad { color:$con_bad; font-weight:700; }
.cline { display:grid; grid-template-columns:86px 40px 46px minmax(0,1fr); gap:8px; font-size:14px; padding:3px 0; white-space:nowrap; }
.cline.hd { color:$con_hd; border-bottom:1px dashed $con_line; }
.cline.active { color:$con_hi; font-weight:600; }
.bars { overflow:hidden; }
.scroll { max-height:220px; overflow-y:auto; overflow-x:hidden; }
.muted { color:$con_muted; font-size:14px; }
"""
CSS = Template(_CSS_TEMPLATE).substitute(T)



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
    W, H, G = 1600, 380, 330   # width, height, ground line
    star_xy = [(40, 20), (130, 60), (220, 30), (330, 14), (420, 70), (520, 26), (610, 52), (700, 18),
               (790, 64), (880, 30), (960, 12), (1050, 58), (1130, 24), (1180, 80), (1260, 40), (1340, 16),
               (1420, 66), (1500, 28), (1570, 54), (80, 120), (300, 140), (470, 110), (650, 150), (980, 130),
               (1110, 160), (1380, 120), (1540, 150), (180, 190), (860, 180), (1290, 200)]
    stars = "".join(f"<rect x='{x}' y='{y + 30}' width='4' height='4' fill='{T['star']}'/>" for x, y in star_xy)
    my = 110   # moon
    moon = (f"<rect x='1180' y='{my}' width='48' height='48' fill='{T['moon']}'/>"
            f"<rect x='1170' y='{my + 10}' width='10' height='28' fill='{T['moon']}'/>"
            f"<rect x='1228' y='{my + 10}' width='10' height='28' fill='{T['moon']}'/>"
            f"<rect x='1204' y='{my + 10}' width='14' height='14' fill='{T['moon2']}'/>")
    mountains = (f"<polygon points='0,{G} 160,170 340,{G}' fill='{T['mtn1']}'/>"
                 f"<polygon points='220,{G} 420,130 620,{G}' fill='{T['mtn2']}'/>"
                 f"<polygon points='1020,{G} 1200,160 1380,{G}' fill='{T['mtn1']}'/>"
                 f"<polygon points='1260,{G} 1450,180 1600,{G}' fill='{T['mtn2']}'/>")
    cx, top = 740, G - 140   # castle
    castle = (f"<rect x='{cx}' y='{top + 30}' width='180' height='{G - top - 30}' fill='{T['castle']}'/>"
              f"<rect x='{cx - 30}' y='{top}' width='50' height='{G - top}' fill='{T['tower']}'/>"
              f"<rect x='{cx + 160}' y='{top}' width='50' height='{G - top}' fill='{T['tower']}'/>"
              + "".join(f"<rect x='{x}' y='{top - 12}' width='12' height='12' fill='{T['tower']}'/>"
                        for x in (cx - 30, cx - 11, cx + 8, cx + 160, cx + 179, cx + 198))
              + "".join(f"<rect x='{x}' y='{top + 18}' width='16' height='12' fill='{T['castle']}'/>"
                        for x in (cx + 22, cx + 56, cx + 90, cx + 124))
              + f"<rect x='{cx + 70}' y='{G - 60}' width='40' height='60' fill='{T['door']}'/>"
              f"<g class='flick'><rect x='{cx - 14}' y='{top + 40}' width='16' height='22' fill='{T['window']}'/>"
              f"<rect x='{cx + 176}' y='{top + 40}' width='16' height='22' fill='{T['window']}'/>"
              f"<rect x='{cx + 50}' y='{G - 80}' width='10' height='16' fill='{T['torch']}'/>"
              f"<rect x='{cx + 120}' y='{G - 80}' width='10' height='16' fill='{T['torch']}'/></g>")
    coins = "".join(f"<rect x='{x}' y='{G - 16}' width='12' height='12' fill='{T['coin']}'/>"
                    f"<rect x='{x + 4}' y='{G - 12}' width='4' height='4' fill='{T['coin_dot']}'/>"
                    for x in (560, 610, 660, 1020, 1070))
    slime = (f"<g class='squish'><rect x='1140' y='{G - 42}' width='60' height='42' fill='{T['slime']}'/>"
             f"<rect x='1132' y='{G - 26}' width='76' height='26' fill='{T['slime']}'/>"
             f"<rect x='1154' y='{G - 32}' width='8' height='12' fill='{T['eye']}'/>"
             f"<rect x='1180' y='{G - 32}' width='8' height='12' fill='{T['eye']}'/>"
             f"<rect x='1148' y='{G - 38}' width='12' height='6' fill='{T['slime_hi']}'/></g>")
    petals = "" if not T["petal"] else "".join(
        f"<rect class='petal' style='animation-delay:-{d}s' x='{x}' y='{y}' width='6' height='6' fill='{T['petal']}'/>"
        for x, y, d in [(120, 20, 0), (380, 60, 2), (640, 10, 4), (900, 40, 1), (1160, 70, 3), (1420, 30, 5),
                        (260, 120, 1.5), (780, 140, 3.5), (1300, 110, 0.5)])
    heroes = _sprite_g("Kerem", 400, G - 72, 9, "bob") + _sprite_g("Elif", 490, G - 72, 9, "bob2")
    ground = (f"<rect x='0' y='{G + 4}' width='{W}' height='{H - G - 4}' fill='{T['ground']}'/>"
              f"<rect x='0' y='{G}' width='{W}' height='4' fill='{T['ground_top']}'/>"
              + "".join(f"<rect x='{x}' y='{G + 16}' width='18' height='6' fill='{T['ground_dash']}'/>" for x in range(20, W, 90))
              + "".join(f"<rect x='{x}' y='{G + 32}' width='18' height='6' fill='{T['ground_dash']}'/>" for x in range(65, W, 90)))
    return (f"<svg class='scene' viewBox='0 0 {W} {H}' preserveAspectRatio='xMidYMax slice' "
            "shape-rendering='crispEdges' aria-hidden='true'>"
            f"<rect width='{W}' height='{H}' fill='{T['sky']}'/><rect width='{W}' height='220' fill='{T['sky2']}'/>"
            f"<rect width='{W}' height='120' fill='{T['sky3']}'/>"
            f"{stars}{moon}{mountains}{castle}{coins}{slime}{heroes}{ground}{petals}</svg>")


HUD = {"Kerem": {"hp": (82, 100), "mp": (40, 60)}, "Elif": {"hp": (64, 80), "mp": (95, 120)}}


def _gauge(label, cur, mx, color):
    return (f"<div class='stat'><span>{label}</span><div class='gauge'>"
            f"<i style='width:{cur / mx * 100:.0f}%;background:{color}'></i></div><span>{cur}/{mx}</span></div>")


def _minimap():
    cells = ["....#....", ".###.###.", ".#.....#.", ".#..K..#.", ".###E###.", "....#...."]
    colors = {"#": T["map_wall"], "K": PLAYERS["Kerem"]["color"], "E": PLAYERS["Elif"]["color"]}
    rects = "".join(f"<rect x='{x * 8}' y='{y * 8}' width='8' height='8' fill='{colors[c]}'/>"
                    for y, row in enumerate(cells) for x, c in enumerate(row) if c != ".")
    return (f"<svg width='108' height='72' viewBox='0 0 72 48' shape-rendering='crispEdges' aria-label='minimap'>"
            f"<rect width='72' height='48' fill='{T['door']}'/>{rects}</svg>")


def header(model_name, budget_ms):
    """Top of the screen: the PIXEL KEEP game scene and a full-width game HUD strip."""
    cards = "".join(
        f"<div class='hcell'><div class='hrow'>{sprite(n, 7)}<div style='flex:1'>"
        f"<div class='px hname' style='color:{PLAYERS[n]['color']}'>{esc(n.upper())} "
        f"<span class='hrole'>{esc(PLAYERS[n]['role'])} · Lv 12 · in chat</span></div>"
        f"{_gauge('HP', *HUD[n]['hp'], T['hp'])}{_gauge('MP', *HUD[n]['mp'], T['mp'])}"
        "</div></div></div>"
        for n in GAME["party"]
    )
    quest = ("<div class='hcell'><div class='px hlabel'>QUEST</div>"
             "<div class='px hbig'>Defeat the Crystal Golem</div>"
             f"{_gauge('', 2, 3, T['accent']).replace('<span></span>', '<span>◆</span>')}"
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
