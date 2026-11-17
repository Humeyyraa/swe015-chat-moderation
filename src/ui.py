"""HTML rendering for the game-lobby chat and the moderator panel (used by app.py).

Pure functions: they take plain Python data and return HTML strings, so they
can be tested without Gradio or any model.
"""
import html

PLAYERS = {
    "Kerem": {"team": "Team Aurora", "color": "#2DD4BF"},
    "Elif": {"team": "Team Nova", "color": "#B392F0"},
}
SHORT = {"Lexicon + edit distance": "Lexicon", "tf-idf + LogReg": "tf-idf + LogReg",
         "word2vec + FFN": "word2vec + FFN", "BERTurk (PyTorch)": "BERTurk FP32",
         "BERTurk INT8 (ONNX)": "BERTurk INT8"}

CSS = """
.gradio-container { background: #0E1320 !important; max-width: 1400px !important; }
.lobby-head { display:none; align-items:center; gap:16px; padding:14px 18px; border-radius:14px;
  background: linear-gradient(90deg, #1A2238, #141B2D); border:1px solid #2A3550; }
.lobby-head .title { font-size:22px; font-weight:700; color:#EEF2FA; }
.lobby-head .sub { font-size:14px; color:#9AA7C2; }
.pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:12px; font-weight:700; letter-spacing:.3px; }
.pill.on { background:#123B2A; color:#4ADE80; border:1px solid #1F6B47; }
.chatbox { height:470px; overflow-y:auto; display:flex; flex-direction:column-reverse; gap:10px;
  padding:14px; border-radius:14px; background:#121A2B; border:1px solid #26314A; }
.msg { display:flex; gap:12px; align-items:flex-start; }
.avatar { width:38px; height:38px; min-width:38px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-weight:800; color:#0E1320; font-size:16px; }
.mbody { flex:1; padding:8px 12px; border-radius:12px; background:#18223A; border:1px solid #233050; }
.mbody.flag { background:#2A1620; border-color:#7A2B3A; }
.mhead { font-size:13px; color:#8C99B5; margin-bottom:3px; }
.mhead b { font-size:14px; }
.mtext { font-size:15px; color:#E6EBF5; line-height:1.4; word-break:break-word; }
.hidden-note { color:#F59AA5; font-weight:700; font-size:14px; }
.mmeta { margin-top:6px; font-size:12px; color:#8C99B5; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.tag { padding:2px 8px; border-radius:6px; font-weight:800; font-size:11px; letter-spacing:.4px; }
.tag.off { background:#E5484D; color:#fff; }
.tag.ok { background:#1F8F55; color:#fff; }
details.orig summary { cursor:pointer; color:#B7C2DA; font-size:12px; }
details.orig div { color:#C9D2E6; font-size:13px; margin-top:3px; }
.empty { color:#6F7C99; text-align:center; margin:auto; font-size:15px; }
.game { border-radius:16px; overflow:hidden; border:1px solid #2A3550; background:#0B1020; }
.gbar { display:flex; align-items:center; gap:22px; padding:14px 20px;
  background: linear-gradient(90deg, #2A1408 0%, #1A1F35 45%, #101629 100%); border-bottom:1px solid #2A3550; }
.logo { display:flex; align-items:center; gap:20px; padding-left:6px; }
.logo .emb { width:36px; height:36px; transform:rotate(45deg); border-radius:8px;
  background: linear-gradient(135deg, #F7B046, #C2410C); box-shadow:0 0 18px rgba(247,176,70,.45); }
.logo .name { font-size:26px; font-weight:900; letter-spacing:4px; color:#F7E7C6; line-height:1; }
.logo .name small { display:block; font-size:11px; letter-spacing:6px; color:#F7B046; font-weight:700; margin-top:4px; }
.minfo { flex:1; display:flex; flex-direction:column; gap:3px; }
.minfo .l1 { font-size:13px; letter-spacing:1.5px; color:#AEB9D3;  font-weight:700; }
.minfo .l2 { font-size:13px; color:#7F8BA6; }
.score { display:flex; align-items:center; gap:12px; font-weight:900; font-size:26px; }
.score .b { color:#2DD4BF; } .score .r { color:#B392F0; } .score .sep { color:#56627E; font-size:20px; }
.score .lbl { font-size:11px; letter-spacing:2px; color:#7F8BA6; font-weight:700; }
.modstat { text-align:right; display:flex; flex-direction:column; gap:4px; align-items:flex-end; }
.modstat .m { font-size:12px; color:#AEB9D3; }
.roster { display:flex; justify-content:space-between; gap:16px; padding:10px 20px; background:#0F1528; }
.team { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.team .tl { font-size:11px; letter-spacing:2px; font-weight:800; margin-right:4px; }
.pl { font-size:12px; padding:4px 9px; border-radius:8px; background:#18223A; color:#AEB9D3; border:1px solid #233050; }
.pl.me { color:#FFFFFF; font-weight:700; }
.tabs { display:flex; gap:6px; margin-bottom:8px; }
.tab { font-size:12px; font-weight:800; letter-spacing:1px; padding:5px 12px; border-radius:8px; color:#7F8BA6; background:#121A2B; border:1px solid #26314A; }
.tab.on { color:#0E1320; background:#F7B046; border-color:#F7B046; }
.sys { font-size:13px; color:#F7B046; padding:6px 10px; border-radius:8px; background:#1E1A12; border:1px solid #4A3A1A; }
.mod { display:flex; flex-direction:column; gap:12px; }
.modcard { padding:14px; border-radius:14px; background:#121A2B; border:1px solid #26314A; }
.modcard h4 { margin:0 0 10px 0; font-size:13px; letter-spacing:.8px;  color:#9AA7C2; }
.kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; }
.kpi { background:#18223A; border-radius:10px; padding:10px; }
.kpi .v { font-size:22px; font-weight:800; color:#EEF2FA; }
.kpi .l { font-size:11px; color:#8C99B5;  letter-spacing:.5px; }
.lrow { display:grid; grid-template-columns: 64px 50px 88px 40px minmax(0,1fr); gap:8px; align-items:center;
  padding:6px 0; border-bottom:1px solid #1F2940; font-size:13px; color:#C9D2E6; }
.lrow.hd { color:#7F8BA6; font-size:11px;  letter-spacing:.5px; }
.lat { display:flex; align-items:center; gap:6px; }
.lat .track { flex:1; min-width:30px; height:8px; border-radius:4px; background:#232E47; position:relative; overflow:hidden; }
.lat .fill { height:8px; border-radius:4px; }
.lat .ms { width:58px; white-space:nowrap; text-align:right; font-variant-numeric: tabular-nums; }
.scroll { max-height:230px; overflow-y:auto; overflow-x:hidden; }
.lrow > *, .crow > * { min-width:0; }
.crow { display:grid; grid-template-columns: 112px 88px 40px minmax(0,1fr); gap:8px; align-items:center;
  padding:6px 0; border-bottom:1px solid #1F2940; font-size:13px; color:#C9D2E6; }
.crow.active { color:#FFFFFF; font-weight:700; }
.muted { color:#6F7C99; font-size:12px; }
"""


def esc(s):
    return html.escape(str(s))


GAME = {"name": "VALKORA", "mode": "Ranked · 5v5", "map": "Frostpeak Ruins", "server": "EU-TR",
        "score": (12, 9),
        "blue": ["Kerem", "Zeynep", "Mert", "Ayşe", "Can"],
        "red": ["Elif", "Burak", "Selin", "Emre", "Deniz"]}


def header(model_name, budget_ms):
    """Top bar of the (fictional) game VALKORA Arena: logo, match info, score, rosters, moderation status."""
    def team(names, color, label):
        pills = []
        for n in names:
            if n in PLAYERS:   # the two players who chat in this demo
                pills.append(f"<span class='pl me' style='border-color:{color}'>💬 {esc(n)}</span>")
            else:
                pills.append(f"<span class='pl'>{esc(n)}</span>")
        return f"<div class='team'><span class='tl' style='color:{color}'>{label}</span>{''.join(pills)}</div>"
    b, r = GAME["score"]
    return (
        "<div class='game'>"
        "<div class='gbar'>"
        "<div class='logo'><div class='emb'></div>"
        f"<div class='name'>{GAME['name']}<small>ARENA</small></div></div>"
        "<div class='minfo'>"
        f"<div class='l1'>{esc(GAME['mode'].upper())} · {esc(GAME['map'].upper())} · SERVER {esc(GAME['server'])}</div>"
        "<div class='l2'>Match in progress · all-chat is moderated before messages are shown</div></div>"
        f"<div class='score'><span class='lbl'>AURORA</span><span class='b'>{b}</span><span class='sep'>:</span>"
        f"<span class='r'>{r}</span><span class='lbl'>NOVA</span></div>"
        "<div class='modstat'><span class='pill on'>● MODERATION ON</span>"
        f"<span class='m'>{esc(model_name)} · ≤ {budget_ms:.0f} ms / message · CPU</span></div>"
        "</div>"
        "<div class='roster'>"
        f"{team(GAME['blue'], PLAYERS['Kerem']['color'], 'AURORA')}"
        f"{team(GAME['red'], PLAYERS['Elif']['color'], 'NOVA')}"
        "</div></div>"
    )


def _verdict_tag(off):
    return "<span class='tag off'>OFFENSIVE</span>" if off else "<span class='tag ok'>CLEAN</span>"


def bubble(m):
    p = PLAYERS.get(m["player"], {"team": "", "color": "#AAB4C8"})
    head = (f"<div class='mhead'>[ALL] <b style='color:{p['color']}'>{esc(m['player'])}</b> "
            f"· {esc(p['team'])} · {esc(m['time'])}</div>")
    if m["off"]:
        shown = m["shown"]
        text = "<div class='hidden-note'>⚠ Hidden by moderation</div>"
        if shown:
            text += f"<div class='mtext'>{esc(shown)}</div>"
        text += (f"<details class='orig'><summary>show original</summary>"
                 f"<div>{esc(m['original'])}</div></details>")
    else:
        text = f"<div class='mtext'>{esc(m['original'])}</div>"
    meta = (f"<div class='mmeta'>{_verdict_tag(m['off'])}"
            f"<span>P(offensive) = {m['p']:.2f} · threshold {m['th']:.2f}</span>"
            f"<span>· {m['ms']:.1f} ms · {esc(SHORT.get(m['model'], m['model']))}</span></div>")
    return (f"<div class='msg'><div class='avatar' style='background:{p['color']}'>{esc(m['player'][:1])}</div>"
            f"<div class='mbody{' flag' if m['off'] else ''}'>{head}{text}{meta}</div></div>")


def chat_html(chat):
    system = (f"<div class='sys'>[System] Match started on {esc(GAME['map'])}. "
              "Chat moderation is active: offensive messages are hidden.</div>")
    # column-reverse keeps the newest message at the bottom and in view
    inner = "".join(bubble(m) for m in reversed(chat)) + system
    tabs = "<div class='tabs'><span class='tab on'>ALL CHAT</span><span class='tab'>TEAM</span></div>"
    return f"{tabs}<div class='chatbox'>{inner}</div>"


def _lat_bar(ms, budget):
    frac = min(ms / budget, 1.0)
    color = "#4ADE80" if ms <= budget else "#E5484D"
    return (f"<div class='lat'><div class='track'><div class='fill' style='width:{frac * 100:.0f}%;"
            f"background:{color}'></div></div><span class='ms'>{ms:.1f} ms</span></div>")


def moderator_html(log, cmp_rows, model_name, budget):
    n = len(log)
    flagged = sum(1 for r in log if r["off"])
    mine = [r["ms"] for r in log if r["model"] == model_name]
    mean = sum(mine) / len(mine) if mine else None
    within = sum(1 for r in log if r["ms"] <= budget)
    kpis = (
        "<div class='kpis'>"
        f"<div class='kpi'><div class='v'>{n}</div><div class='l'>MESSAGES</div></div>"
        f"<div class='kpi'><div class='v'>{flagged}</div><div class='l'>FLAGGED</div></div>"
        f"<div class='kpi'><div class='v'>{'–' if mean is None else f'{mean:.1f}'}</div><div class='l'>MEAN MS, ACTIVE</div></div>"
        f"<div class='kpi'><div class='v'>{'–' if not n else f'{within}/{n}'}</div><div class='l'>≤ {budget:.0f} MS</div></div>"
        "</div>"
    )
    rows = "".join(
        f"<div class='lrow'><span>{esc(r['time'])}</span><span>{esc(r['player'])}</span>"
        f"<span>{_verdict_tag(r['off'])}</span><span>{r['p']:.2f}</span>{_lat_bar(r['ms'], budget)}</div>"
        for r in reversed(log)
    ) or "<div class='muted'>No messages yet.</div>"
    log_block = (
        "<div class='modcard'><h4>MODERATION LOG · EVERY MESSAGE</h4>"
        "<div class='lrow hd'><span>TIME</span><span>PLAYER</span><span>VERDICT</span><span>P(OFF)</span>"
        f"<span>LATENCY · BAR = {budget:.0f} MS</span></div>"
        f"<div class='scroll'>{rows}</div></div>"
    )
    if cmp_rows:
        crow = "".join(
            f"<div class='crow{' active' if c['model'] == model_name else ''}'>"
            f"<span>{esc(SHORT.get(c['model'], c['model']))}</span><span>{_verdict_tag(c['off'])}</span>"
            f"<span>{c['p']:.2f}</span>{_lat_bar(c['ms'], budget)}</div>"
            for c in cmp_rows
        )
        cmp_block = ("<div class='modcard'><h4>ALL MODELS ON THE LAST MESSAGE</h4>"
                     "<div class='crow' style='color:#7F8BA6;font-size:11px;'>"
                     "<span>MODEL</span><span>VERDICT</span><span>P(OFF)</span><span>LATENCY</span></div>"
                     f"{crow}</div>")
    else:
        cmp_block = ""
    return f"<div class='mod'><div class='modcard'><h4>SESSION</h4>{kpis}</div>{log_block}{cmp_block}</div>"
