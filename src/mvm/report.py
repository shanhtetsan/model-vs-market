"""Charts + post drafts for the weekly X threads."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output"
HANDLE = "@havertzShan"
REPO = "github.com/shanhtetsan/model-vs-market"

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet font fallbacks

# ---- Theme: built on Kalshi's public brand kit (kalshi.com/brandkit) ----------------
# Palette: their six core colors. Type: Archivo (open source) standing in for the proprietary
# Kalshi Sans, using the same idea of a standard width for text and a condensed width for data.
WHITE, LIGHT, GREEN, DARK_GREEN, BLACK = "#FFFFFF", "#F5F7F4", "#00B67A", "#014737", "#050D0A"
MUTED, RULE, SOFT = "#6B726E", "#E3E7E2", "#C9D3CD"
# One color family + black: Kalshi green for the market, black for my model and my mark.
ACCENT = DARK_GREEN   # hero number, highlighted gap
MINE = BLACK          # my model's ring, logo ring
_A = Path(__file__).parent / "assets"
BOLD, SEMI, REG = (FontProperties(fname=str(_A / f"Archivo-{w}.ttf")) for w in ("Bold", "SemiBold", "Regular"))
COND, COND_B = (FontProperties(fname=str(_A / f"Archivo-{w}.ttf")) for w in ("CondReg", "CondBold"))
AUTHOR = "havertzShan"
SLOGAN = "Probabilities, not opinions."


def _t(fig, x, y, s, fp, size, color=BLACK, **kw):
    return fig.text(x, y, s, fontproperties=fp, fontsize=size, color=color, **kw)


def _frame(fig, kicker: str, title: str, subtitle: str | None = None):
    _t(fig, .07, .935, kicker.upper(), COND_B, 12, GREEN)
    _t(fig, .07, .875, title, BOLD, 26)
    if subtitle:
        _t(fig, .07, .842, subtitle, REG, 12.5, MUTED)


def _unused_ring(ax, cx: float, stem_top: float, r: float, color: str, bg: str):
    """The 'my model' marker as the dot of the i: a solid ring with a clear gap above the stem."""
    from matplotlib.patches import Circle
    cy = stem_top + r * .45 + r
    ax.add_patch(Circle((cx, cy), r, color=color, lw=0))
    ax.add_patch(Circle((cx, cy), r * .58, color=bg, lw=0))


def wordmark(ax, x: float, y: float, height: float, color: str = BLACK, bg: str = WHITE,
             text: str | None = None, ring: str | None = None) -> float:
    """Ring mark + name. The ring is the 'my model' marker from every chart.

    ax needs equal aspect. (x, y) = baseline-left, height = x-height (data units). Returns width.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D
    text = text or AUTHOR
    k = height / TextPath((0, 0), "x", size=1, prop=BOLD).get_extents().height
    r = height * .5
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x + r, y + height / 2), r, color=ring or MINE, lw=0))
    ax.add_patch(Circle((x + r, y + height / 2), r * .55, color=bg, lw=0))
    tx = x + 2 * r + height * .32
    tp = TextPath((0, 0), text, size=1, prop=BOLD)
    ax.add_patch(PathPatch(Affine2D().scale(k).translate(tx, y).transform_path(tp), color=color, lw=0))
    return tx - x + tp.get_extents().x1 * k


def icon(ax, cx: float, cy: float, size: float, color: str = BLACK, bg: str = LIGHT, letters: str = "hS",
         ring: str | None = None):
    """Profile picture: the ring, with initials inside."""
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch, Circle
    from matplotlib.transforms import Affine2D
    r = size / 2
    ax.add_patch(Circle((cx, cy), r, color=ring or MINE, lw=0))
    ax.add_patch(Circle((cx, cy), r * .80, color=bg, lw=0))
    tp = TextPath((0, 0), letters, size=1, prop=BOLD); e = tp.get_extents()
    k = (r * .95) / e.width
    ax.add_patch(PathPatch(Affine2D().translate(-(e.x0 + e.width / 2), -(e.y0 + e.height / 2)).scale(k)
                           .translate(cx, cy).transform_path(tp), color=color, lw=0))


def _footer(fig):
    fig.add_artist(plt.Line2D([.07, .93], [.072, .072], color=RULE, lw=1.2))
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    wm = wordmark(ax, .07 * W, .028 * H, .12)
    _t(fig, (.07 * W + wm + .16) / W, .031, SLOGAN.upper(), COND_B, 10.5, MUTED)
    _t(fig, .93, .031, f"{HANDLE}  ·  INDEPENDENT  ·  DATA: KALSHI", COND, 10.5, MUTED, ha="right")


def _key(fig, y, items):
    """items: [(label, kind, color)] where kind is 'dot' or 'ring'."""
    ax = fig.add_axes([0, 0, 1, 1], facecolor="none"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x = .07
    for label, kind, color in items:
        if kind == "dot":
            ax.scatter([x + .006], [y], s=60, color=color)
        else:
            ax.scatter([x + .006], [y], s=60, facecolor=WHITE, edgecolor=color, lw=2.2)
        _t(fig, x + .02, y, label.upper(), COND_B, 10.5, MUTED, va="center")
        x += .03 + .0105 * len(label)


def _axis_style(ax):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(COND); lab.set_fontsize(11); lab.set_color(MUTED)


def _signed(v: float) -> str:
    g = round(v * 100)
    return "0" if g == 0 else f"{g:+d}".replace("-", "\u2212")


def preview_chart(df: pd.DataFrame, matchweek: str, path: Path, score: str | None = None) -> Path:
    """Model vs Kalshi on every match. df needs home, away, p_H (model), k_H (Kalshi)."""
    d = df.assign(gap=df.p_H - df.k_H)
    d = d.reindex(d.gap.abs().sort_values().index).reset_index(drop=True)
    top = d.index[-1]
    t = d.iloc[-1]
    fig = plt.figure(figsize=(8, 10), dpi=200, facecolor=WHITE)
    _frame(fig, f"{matchweek}  ·  Model vs Market", f"{t.home} v {t.away}")

    # hero: the biggest gap, Kalshi's price next to mine
    _t(fig, .07, .745, _signed(t.gap), BOLD, 76, ACCENT)
    _t(fig, .31, .80, "points between my model and Kalshi", REG, 13)
    _t(fig, .31, .772, "on a home win, the biggest gap this week", REG, 13, MUTED)
    _t(fig, .31, .737, f"KALSHI {t.k_H:.0%}   ·   MODEL {t.p_H:.0%}", COND_B, 13)

    _key(fig, .672, [("Kalshi", "dot", GREEN), ("My model", "ring", MINE)])
    y0, h = 0.12, 0.53
    ax = fig.add_axes([0.33, y0, 0.513, h], facecolor="none")  # gap column lands on the right margin
    row = h / (len(d) - .3 + .6)
    yc = y0 + (top + .6) * row
    fig.patches.append(plt.Rectangle((.07, yc - row * .45), .86, row * .9, transform=fig.transFigure,
                                     facecolor=LIGHT, edgecolor="none", zorder=-1))
    for i, r in d.iterrows():
        hi = i == top
        ax.plot([r.k_H, r.p_H], [i, i], color=ACCENT if hi else SOFT, lw=5 if hi else 3.5,
                solid_capstyle="butt", zorder=2)
        ax.scatter(r.k_H, i, s=95 if hi else 60, color=GREEN, edgecolor=WHITE, lw=1.2, zorder=3)
        ax.scatter(r.p_H, i, s=95 if hi else 60, facecolor=WHITE, edgecolor=MINE, lw=2.2 if hi else 1.8, zorder=4)
        ax.text(1.17, i, _signed(r.gap), va="center", ha="right", fontproperties=COND_B,
                fontsize=14 if hi else 12.5, color=ACCENT if hi else MUTED)
        ax.text(-.04, i, f"{r.home} v {r.away}", va="center", ha="right",
                fontproperties=COND_B if hi else COND, fontsize=13.5 if hi else 12.5, color=BLACK if hi else MUTED)
    ax.text(1.17, len(d) - .45, "GAP", ha="right", fontproperties=COND_B, fontsize=10, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(-.6, len(d) - .3); ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=RULE, lw=1); ax.set_axisbelow(True)
    _axis_style(ax)
    _t(fig, .93, .672, "HOME WIN PROBABILITY", COND, 10.5, MUTED, ha="right", va="center")
    if score:
        _t(fig, .93, .935, f"SEASON  ·  {score.upper()}", COND_B, 12, BLACK, ha="right")
    _footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)
    return path


def calibration_chart(cal: pd.DataFrame, title: str, path: Path) -> Path:
    cal = cal[cal.n >= 20]  # tiny bins are noise
    fig = plt.figure(figsize=(8, 10), dpi=200, facecolor=WHITE)
    _frame(fig, "Backtest  ·  Calibration", title,
           "When my model says X%, does it happen X% of the time?")
    ax = fig.add_axes([0.13, 0.15, 0.78, 0.62], facecolor=WHITE)
    ax.plot([0, 1], [0, 1], color=SOFT, lw=1.5, ls=(0, (4, 3)))
    ax.text(.70, .74, "PERFECT", fontproperties=COND_B, fontsize=10, color=MUTED, rotation=38)
    ax.plot(cal.predicted, cal.observed, color=ACCENT, lw=2.5, zorder=2)
    ax.scatter(cal.predicted, cal.observed, s=40 + cal.n / cal.n.max() * 260, color=GREEN,
               edgecolor=WHITE, lw=1.5, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{x:.0%}" for x in ticks]); ax.set_yticklabels([f"{x:.0%}" for x in ticks])
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(COND); lab.set_fontsize(11); lab.set_color(MUTED)
    ax.grid(color=RULE, lw=1); ax.set_axisbelow(True)
    _axis_style(ax)
    ax.set_xlabel("MODEL SAID", fontproperties=COND_B, fontsize=10.5, color=MUTED)
    ax.set_ylabel("HAPPENED", fontproperties=COND_B, fontsize=10.5, color=MUTED)
    _t(fig, .07, .095, "Walk-forward, no look-ahead. Bigger dot = more predictions.", REG, 10.5, MUTED)
    _footer(fig)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)
    return path


def _pct(x: float) -> str:
    return f"{x:.0%}"


def preview_thread(df: pd.DataFrame, matchweek: str, top_n: int = 3) -> str:
    big = df.sort_values("edge_abs", ascending=False).head(top_n)
    lines = [
        f"1/ ⚽ Model vs Market: {matchweek}\n\n"
        f"My Dixon-Coles model vs @Kalshi prices for all {len(df)} Premier League games. "
        f"Biggest disagreements below 🧵",
    ]
    for i, r in enumerate(big.itertuples(), start=2):
        side = r.home if r.edge > 0 else r.away
        lines.append(
            f"{i}/ {r.home} v {r.away}\n"
            f"Model: {r.home} {_pct(r.p_H)} · Draw {_pct(r.p_D)} · {r.away} {_pct(r.p_A)}\n"
            f"Kalshi: {r.home} {_pct(r.k_H)} · Draw {_pct(r.k_D)} · {r.away} {_pct(r.k_A)}\n"
            f"Model is higher on {side} by {abs(r.edge):.0%}. [YOUR TAKE: why might the model be wrong here?]"
        )
    lines.append(f"{len(lines) + 1}/ Scored Monday with Brier score, model vs market. Every miss stays up.\n"
                 f"Code: {REPO}")
    return "\n\n---\n\n".join(lines)


def results_thread(week: pd.DataFrame, season: pd.DataFrame, matchweek: str) -> str:
    wm, wk = week["brier_model"].mean(), week["brier_kalshi"].mean()
    sm, sk = season["brier_model"].mean(), season["brier_kalshi"].mean()
    winner = "Model" if wm < wk else "Kalshi"
    best = week.sort_values("brier_model").iloc[0]
    worst = week.sort_values("brier_model").iloc[-1]
    return "\n\n---\n\n".join([
        f"1/ 📊 {matchweek} results: Model vs Market\n\n"
        f"Brier score (lower = better)\nModel: {wm:.3f}\nKalshi: {wk:.3f}\n\n{winner} wins the week.",
        f"2/ Best call: {best.home} v {best.away} ({best.hg}-{best.ag})\n"
        f"Worst miss: {worst.home} v {worst.away} ({worst.hg}-{worst.ag}). [WHAT DID THE MODEL MISS?]",
        f"3/ Season so far ({len(season)} games)\nModel: {sm:.3f} · Kalshi: {sk:.3f}\n\nCode: {REPO}",
    ])


def _fmt_pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%" if p > 0 else "\u2013"
    return f"{p:.0%}"


def index_chart(table: pd.DataFrame, title: str, subtitle: str, path: Path) -> Path:
    """Premier League Market Index. table: team, index, delta, p_title, p_top4, p_relegation."""
    t = table.sort_values("index", ascending=False).reset_index(drop=True)
    n = len(t)
    fig = plt.figure(figsize=(8, 10), dpi=200, facecolor=WHITE)
    _frame(fig, "Premier League Market Index", title, subtitle)
    top, bottom = .775, .11
    rh = (top - bottom) / n
    lo, hi = max(0, t["index"].min() - .15), t["index"].max() + .05
    bx, bw = .29, .25
    cols = {"TITLE": .70, "TOP 4": .805, "DOWN": .905}
    hy = top + rh * .5
    _t(fig, .07, hy, "#", COND_B, 10, MUTED, va="center")
    _t(fig, .105, hy, "TEAM", COND_B, 10, MUTED, va="center")
    _t(fig, bx, hy, "INDEX \u00b7 EXP. PTS / GAME", COND_B, 10, MUTED, va="center")
    for k, x in cols.items():
        _t(fig, x, hy, k, COND_B, 10, MUTED, va="center", ha="center")
    fig.add_artist(plt.Line2D([.07, .93], [top, top], color=RULE, lw=1.2))
    for i, r in t.iterrows():
        y = top - (i + .5) * rh
        if i == 0:
            fig.patches.append(plt.Rectangle((.07, y - rh / 2), .86, rh, transform=fig.transFigure,
                                             facecolor=LIGHT, edgecolor="none", zorder=0))
        if i in (3, n - 4):  # cut lines: top 4 / relegation zone
            fig.add_artist(plt.Line2D([.07, .93], [y - rh / 2, y - rh / 2], color=SOFT, lw=1, ls=(0, (3, 3))))
        _t(fig, .07, y, f"{i + 1}", COND, 12, MUTED, va="center")
        _t(fig, .105, y, r.team, SEMI if i == 0 else REG, 12, va="center")
        frac = (r["index"] - lo) / (hi - lo)
        fig.patches.append(plt.Rectangle((bx, y - rh * .26), bw * frac, rh * .52, transform=fig.transFigure,
                                         facecolor=DARK_GREEN if i == 0 else GREEN, edgecolor="none"))
        _t(fig, bx + bw * frac + .008, y, f"{r['index']:.2f}", COND_B, 11.5, va="center")
        d = r.get("delta", np.nan)
        if pd.notna(d) and abs(d) >= .005:
            sign = "+" if d > 0 else "\u2212"
            _t(fig, .625, y, f"{sign}{abs(d):.2f}", COND, 10,
               GREEN if d > 0 else MUTED, va="center", ha="center")
        for k, col in zip(cols, ["p_title", "p_top4", "p_relegation"]):
            p = r[col]
            strong = p >= (.25 if col == "p_relegation" else .5)
            _t(fig, cols[k], y, _fmt_pct(p), COND_B if strong else COND, 12.5,
               (DARK_GREEN if col != "p_relegation" else BLACK) if strong else (BLACK if p >= .005 else SOFT),
               va="center", ha="center")
    _t(fig, .07, .085, "Ratings implied by Kalshi match prices  \u00b7  season odds from 20,000 simulations",
       REG, 9.5, MUTED)
    _footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)
    return path


def title_gap_chart(rows: pd.DataFrame, path: Path, title: str, subtitle: str, note: str | None = None) -> Path:
    """Kalshi title market vs title odds implied by Kalshi match prices.

    rows: team, market (Kalshi title price), implied (index simulation), both in [0, 1].
    """
    d = rows.assign(gap=rows.implied - rows.market).reset_index(drop=True)
    d = d.iloc[::-1].reset_index(drop=True)          # biggest favourite on top
    top = d.gap.abs().idxmax()
    fig = plt.figure(figsize=(8, 10), dpi=200, facecolor=WHITE)
    _frame(fig, "Premier League title  ·  Two Kalshi prices", title, subtitle)
    _key(fig, .775, [("Title market", "dot", GREEN), ("Implied by match prices", "ring", MINE)])
    y0, h = 0.25, 0.47
    ax = fig.add_axes([0.25, y0, 0.40, h], facecolor="none")
    cols = {"MARKET": .72, "MATCHES": .82, "GAP": .925}
    row = h / (len(d) - .2 + .6)   # matches ylim span below
    yc = y0 + (top + .6) * row
    fig.patches.append(plt.Rectangle((.07, yc - row * .42), .86, row * .84, transform=fig.transFigure,
                                     facecolor=LIGHT, edgecolor="none", zorder=-1))
    for i, r in d.iterrows():
        hi = i == top
        ax.plot([r.market, r.implied], [i, i], color=ACCENT if hi else SOFT, lw=6 if hi else 4,
                solid_capstyle="butt", zorder=2)
        ax.scatter(r.market, i, s=170 if hi else 110, color=GREEN, edgecolor=WHITE, lw=1.5, zorder=3)
        ax.scatter(r.implied, i, s=170 if hi else 110, facecolor=WHITE, edgecolor=MINE, lw=2.6, zorder=4)
        ax.text(-.03, i, r.team, va="center", ha="right", fontproperties=COND_B if hi else COND,
                fontsize=16 if hi else 15, color=BLACK if hi else MUTED, transform=ax.transData)
        yf = y0 + (i + .6) * row
        _t(fig, cols["MARKET"], yf, f"{r.market:.0%}", COND_B, 15, DARK_GREEN, ha="center", va="center")
        _t(fig, cols["MATCHES"], yf, f"{r.implied:.0%}", COND_B, 15, BLACK, ha="center", va="center")
        _t(fig, cols["GAP"], yf, _signed(-r.gap), COND_B, 17 if hi else 15, ACCENT if hi else MUTED,
           ha="right", va="center")
    hy = y0 + (len(d) - .2 + .6) * row - row * .35
    for k, x in cols.items():
        _t(fig, x, hy, k, COND_B, 10, MUTED, ha="right" if k == "GAP" else "center", va="center")
    ax.set_xlim(0, .8); ax.set_ylim(-.6, len(d) - .2); ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75]); ax.set_xticklabels(["0%", "25%", "50%", "75%"])
    ax.grid(axis="x", color=RULE, lw=1); ax.set_axisbelow(True)
    _axis_style(ax)
    if note:
        _t(fig, .07, .135, note, REG, 11, MUTED, wrap=True)
    _t(fig, .07, .095, "Implied = 20,000 simulated seasons using team ratings fitted to Kalshi pre-match prices.",
       REG, 9.5, MUTED)
    _footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)
    return path


def ladder_chart(b: pd.DataFrame, path: Path, kicker: str, title: str, subtitle: str,
                 hero: str, hero_label: str, issues: list[str] | None = None, note: str | None = None) -> Path:
    """Market's full forecast from a ladder of 'above X' contracts. b: label, p (bucket probability)."""
    fig = plt.figure(figsize=(8, 10), dpi=200, facecolor=WHITE)
    _frame(fig, kicker, title, subtitle)
    _t(fig, .07, .735, hero, BOLD, 64, ACCENT)
    _t(fig, .07, .71, hero_label.upper(), COND_B, 11, MUTED)
    n = len(b)
    top, bottom = .66, .2
    rh = (top - bottom) / n
    best = int(b.p.idxmax())
    pmax = b.p.max()
    _t(fig, .07, top + .012, "OUTCOME", COND_B, 10, MUTED)
    _t(fig, .93, top + .012, "MARKET PROBABILITY", COND_B, 10, MUTED, ha="right")
    for i, r in b.iloc[::-1].reset_index().iterrows():
        y = top - (i + .5) * rh
        hi = r["index"] == best
        _t(fig, .07, y, r.label, COND_B if hi else COND, 15 if hi else 14, BLACK if hi else MUTED, va="center")
        w = .52 * (r.p / pmax if pmax else 0)
        fig.patches.append(plt.Rectangle((.28, y - rh * .3), max(w, .002), rh * .6, transform=fig.transFigure,
                                         facecolor=DARK_GREEN if hi else GREEN, edgecolor="none"))
        _t(fig, .28 + w + .012, y, f"{r.p:.0%}", COND_B, 14 if hi else 13, BLACK, va="center")
    y = .155
    if issues:
        _t(fig, .07, y, "CONSISTENCY CHECK: " + str(len(issues)) + " ISSUE" + ("S" if len(issues) > 1 else ""),
           COND_B, 10.5, BLACK); y -= .022
        for s in issues[:2]:
            _t(fig, .07, y, s, REG, 10, MUTED); y -= .02
    else:
        _t(fig, .07, y, "CONSISTENCY CHECK: PASSED", COND_B, 10.5, ACCENT); y -= .022
        if note:
            _t(fig, .07, y, note, REG, 10, MUTED)
    _t(fig, .07, .088, "Bucket probabilities = differences between adjacent Kalshi 'above' contracts.", REG, 9.5, MUTED)
    _footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=WHITE)
    plt.close(fig)
    return path
