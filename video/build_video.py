#!/usr/bin/env python3
"""Build the submission video: slides + AI narration + captions -> MP4.

Reproducible: renders slides from code (charts use only verified, locally
recomputed figures), synthesises narration with edge-tts, assembles with the
bundled imageio-ffmpeg binary, and burns sentence captions timed from TTS word
boundaries. Run: python3 video/build_video.py  (writes video/pricing_across_markets.mp4)

The storyboard/narration text lives here and in video/script.md.
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "build"
W, H = 1920, 1080
DPI = 100

BG = "#101828"
INK = "#F5F7FA"
MUTED = "#9AA7B8"
ACCENT = "#F2AA4C"
BLUE = "#4EA8DE"
GRID = "#2A3852"

VOICE = "en-US-GuyNeural"
SEG_PAD = 0.6  # seconds of silence after each slide's narration


# ---------------------------------------------------------------- slides ---

SLIDES = [
    dict(
        kind="title",
        title="Pricing Across Markets",
        subtitle="Predicting 2025 claim cost in two insurance markets",
        meta="2026 IAA AI-Assisted Hackathon · October 2026",
        narration=(
            "This is our submission to the 2026 IAA AI-Assisted Hackathon. "
            "In the next five minutes: how we found what moves claim cost "
            "across two insurance markets, and turned it into prices for "
            "every 2025 policy."
        ),
    ),
    dict(
        kind="bullets",
        title="The Business Problem",
        bullets=[
            "Set 2025 prices for every policy in both markets",
            "Established market: 2.4M policy periods, 2021–2024",
            "New market: two thin years, ~9,100 claims",
            "76% of the 2025 book is business never seen before",
            "   — a broker portfolio takeover",
            "Prices must be defensible: renewals, the incoming",
            "   portfolio, and corrections to current rates",
        ],
        narration=(
            "The business problem. You asked us to find what actually moves "
            "claim cost on your policies, and turn that into prices for every "
            "2025 policy in two markets. Your established market gives four "
            "years of history on two point four million policies. The new "
            "market gives only two thin years — and seventy-six percent of its "
            "2025 book is business you have never seen, arriving from a broker "
            "portfolio takeover. These prices will drive renewals, cover the "
            "incoming portfolio, and correct misalignments in your current "
            "rates."
        ),
    ),
    dict(
        kind="bullets",
        title="Data Overview",
        bullets=[
            "Claims at 31 Dec 2024 are unfinished — every training",
            "   label developed to its final value",
            "Corrupt exposure column → recomputed from policy dates",
            "Annualised premiums → treated per-period; 604 duplicate",
            "   claims removed",
            "80 occupation spellings mapped to 20 categories",
            "Thin book: ~1,000 claims per property line, against a",
            "   test book ~4× a normal year",
        ],
        narration=(
            "The data, and what we did to it. Claim costs at the end of 2024 "
            "are unfinished — a claim reported in December is still growing — "
            "so every training label was developed to its final value. The "
            "supplied exposure column mixed unit conventions, so it is now "
            "recomputed from policy dates. Premiums were annualized and "
            "are treated per-period. Six hundred and four duplicate claims "
            "were removed. The new market's eighty occupation spellings mapped "
            "cleanly onto twenty categories. The honest headline: your new "
            "market holds about one thousand claims per property line, against "
            "a test book nearly four times a normal year."
        ),
    ),
    dict(
        kind="bullets",
        title="Limitations",
        bullets=[
            "1.  Development factors are estimates; a handful of",
            "    large claims dominate them",
            "2.  The heavy tail is real — verified, deliberately not",
            "    trimmed",
            "3.  The 2025 cost level was tuned on a live evaluation",
            "    board — it may sit too high",
            "4.  Weather is used as a pricing variable",
            "5.  Our inexperience — and AI errors caught only by",
            "    re-deriving every number — may hide impacts",
        ],
        footnote="These insights may be nullified by these limitations.",
        narration=(
            "Five limitations should shape how you use these results. One: "
            "development factors are estimates, and a handful of large claims "
            "dominate them. Two: the heavy tail is real — verified, and "
            "deliberately not trimmed. Three: the level of the 2025 cost "
            "adjustment was tuned on a live evaluation board, and may sit too "
            "high. Four: weather is used as a pricing variable. Five: our "
            "inexperience — and an AI assistant that made real errors, caught "
            "only by re-deriving every number — may have left important "
            "impacts overlooked. These insights may be nullified by these "
            "limitations."
        ),
    ),
    dict(
        kind="flow",
        title="Adaptation for the New Market",
        boxes=[
            "Mature-market models\n(2.4M policies)",
            "Prior recalibrated\nto the new market",
            "Tiny correction layer\n(20–35 shallow trees)",
            "2025 cost-drift\nadjustment",
        ],
        bullets=[
            "The pivot: stop learning from scratch",
            "Small, careful models beat big ones",
            "With ~9,000 claims: measure noise first, pre-register every test",
        ],
        narration=(
            "When the new market's data arrived, almost every assumption "
            "changed. The pivot: stop learning from scratch. Models trained on "
            "the two point four million established-market policies become a "
            "prior, recalibrated to the new market's cost level. A deliberately "
            "tiny correction layer — twenty to thirty-five shallow trees — "
            "learns only what the prior cannot know: region, gender, channel, "
            "flood zone, new-business status. A measured adjustment covers 2025 "
            "cost drift. The lesson of this data: small, careful models beat "
            "big ones; deeper trees memorize noise. And with so few claims, we "
            "changed how we decide — measure the noise first, pre-register "
            "every test."
        ),
    ),
    dict(
        kind="chart",
        chart="ladder",
        title="Model Validation",
        takeaway="Nearly all the skill is transferred, not learned from the thin book.",
        narration=(
            "Model validation. Each model was tested honestly: train on part of "
            "the book, predict held-out policies. A flat rate scores about zero "
            "point zero eight. The thin new book alone: zero point one one. The "
            "transfer prior alone: nearly zero point one four. The prior with "
            "its small correction layer — our final model — reaches zero point "
            "one four one. Nearly all the skill is transferred from the mature "
            "market, not learned from the thin book. The final model reproduces "
            "its predictions byte for byte."
        ),
    ),
    dict(
        kind="chart",
        chart="flood",
        title="What Moves Claim Cost",
        takeaway=(
            "We can conclude prices are misaligned with experience;\n"
            "we cannot conclude the disparities will persist."
        ),
        narration=(
            "Now, what actually moves claim cost. The starkest finding: in your "
            "new market, contents prices are flat across flood zones — around "
            "one hundred and twenty-five currency units — while observed claim "
            "cost rises six-fold, from sixty-three units in zone one to over "
            "four hundred in zone five. Flood risk is entirely unpriced. Young "
            "drivers show a loss ratio of zero point nine one against zero "
            "point six for middle ages, and male drivers carry a third more "
            "claim cost — both priced about the same today. New business is "
            "priced too cheap relative to renewals. We can conclude your prices "
            "are misaligned with observed experience; we cannot conclude the "
            "disparities will persist."
        ),
    ),
    dict(
        kind="bullets",
        title="AI in the Workflow — and Its Guardrails",
        bullets=[
            "AI did the hands-on analysis: exploring data, writing",
            "   pipelines, running experiments",
            "The human held scope, judgement and every authorisation",
            "Adversarial AI review found 2 critical bugs →",
            "   the largest single improvement",
            "All work ran in an isolated container — no access to",
            "   our machines",
            "103 automated tests green before every commit; version",
            "   control as audit evidence; every number recomputed",
        ],
        narration=(
            "Because this was an AI-assisted engagement, how AI entered the "
            "work matters. The AI agent did the hands-on analysis — exploring "
            "data, writing pipelines, running experiments — while the human "
            "held scope, judgment, and every authorization. A fresh AI model, "
            "instructed to attack the existing work adversarially, found two "
            "critical bugs and produced the largest single improvement in the "
            "project. All of it ran inside an isolated container with no access "
            "to our machines. And trust came from guardrails, not promises: one "
            "hundred and three automated tests, nothing committed until they "
            "pass, version control as audit evidence, and every number "
            "recomputed from source before publication."
        ),
    ),
    dict(
        kind="bullets",
        title="Next Steps",
        bullets=[
            "Discuss and pressure-test the flood, gender and",
            "   new-business findings before any filing",
            "Seek richer data as the new market earns",
            "Monitor 2025 experience, refit, and extend the",
            "   approach to further new markets",
        ],
        footnote="Thank you.",
        narration=(
            "Next steps. Discuss these results with your pricing team, and "
            "pressure-test the flood, gender, and new-business findings before "
            "any filing. Seek richer data as the new market earns. And monitor "
            "2025 experience, refit, and extend the approach to further "
            "markets. We look forward to that conversation. Thank you."
        ),
    ),
]


# ------------------------------------------------------------- rendering ---

def _base_fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def _title(fig, text, y=0.925, size=48):
    fig.text(0.075, y, text, color=INK, fontsize=size, weight="bold",
             va="center", ha="left")


def render_slide(spec: dict, path: Path) -> None:
    fig = _base_fig()
    kind = spec["kind"]

    if kind == "title":
        fig.text(0.5, 0.62, spec["title"], color=INK, fontsize=76, weight="bold",
                 va="center", ha="center")
        fig.text(0.5, 0.50, spec["subtitle"], color=ACCENT, fontsize=38,
                 va="center", ha="center")
        fig.text(0.5, 0.20, spec["meta"], color=MUTED, fontsize=28,
                 va="center", ha="center")

    elif kind == "bullets":
        _title(fig, spec["title"])
        tight = len(spec["bullets"]) >= 9
        y = 0.82 if tight else 0.80
        for line in spec["bullets"]:
            stripped = line.lstrip()
            # A genuine sub-bullet is marked with a leading em-dash. Any other
            # indented line is just the wrapped continuation of the line above
            # and must render as the same colour/size as its parent, not as a
            # muted sub-bullet (D-095).
            sub = stripped.startswith("— ")
            cont = line.startswith("   ") and not sub
            if sub:
                x, colour, size = 0.105, MUTED, 27
            elif cont:
                x, colour, size = 0.105, INK, 33
            else:
                x, colour, size = 0.075, INK, 33
            fig.text(x, y, line, color=colour, fontsize=size,
                     va="center", ha="left")
            y -= (0.075 if not (sub or cont) else 0.052) if tight else (
                0.088 if not (sub or cont) else 0.062)
        if spec.get("footnote"):
            # kept clear of the burned-in caption band (top edge ≈ 0.13)
            fig.text(0.075, 0.225, spec["footnote"], color=ACCENT, fontsize=30,
                     weight="bold", va="center", ha="left")

    elif kind == "flow":
        _title(fig, spec["title"])
        n = len(spec["boxes"])
        bw, bh, gap, x0, yc = 0.185, 0.16, 0.055, 0.075, 0.635
        for i, label in enumerate(spec["boxes"]):
            x = x0 + i * (bw + gap)
            box = FancyBboxPatch((x, yc - bh / 2), bw, bh,
                                 boxstyle="round,pad=0.008,rounding_size=0.012",
                                 facecolor="#1B2A44", edgecolor=BLUE, linewidth=2)
            fig.add_artist(box)
            fig.text(x + bw / 2, yc, label, color=INK, fontsize=22,
                     va="center", ha="center", linespacing=1.5)
            if i < n - 1:
                arr = FancyArrowPatch((x + bw + 0.006, yc),
                                      (x + bw + gap - 0.006, yc),
                                      arrowstyle="-|>", mutation_scale=26,
                                      color=ACCENT, linewidth=2.5)
                fig.add_artist(arr)
        y = 0.40
        for line in spec["bullets"]:
            fig.text(0.075, y, "•  " + line, color=INK, fontsize=31,
                     va="center", ha="left")
            y -= 0.105

    elif kind == "chart":
        _title(fig, spec["title"])
        ax = fig.add_axes([0.075, 0.355, 0.62, 0.40])
        ax.set_facecolor(BG)
        if spec["chart"] == "ladder":
            draw_ladder(ax)
        else:
            draw_flood(ax)
        tk = spec["takeaway"]
        y = 0.25
        for i, line in enumerate(tk.split("\n")):
            fig.text(0.075, y - i * 0.062, line, color=ACCENT, fontsize=29,
                     weight="bold", va="center", ha="left")

    fig.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(fig)


def draw_ladder(ax):
    labels = ["Flat rate", "Premium × LR", "New market\nalone",
              "Transfer\nprior", "Prior +\ncorrection"]
    vals = [0.0804, 0.1097, 0.1073, 0.1386, 0.1407]
    colors = [MUTED, MUTED, MUTED, BLUE, ACCENT]
    bars = ax.bar(range(5), vals, color=colors, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", color=INK, fontsize=17)
    ax.set_xticks(range(5), labels, color=INK, fontsize=15)
    ax.set_ylim(0, 0.162)
    ax.set_ylabel("Out-of-fold D² (higher is better)", color=MUTED, fontsize=16)
    ax.tick_params(colors=MUTED, labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def draw_flood(ax):
    zones = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]
    cost = [63.2, 92.4, 145.6, 237.7, 417.7]   # claim cost / exposure (CU)
    bars = ax.bar(range(5), cost, color=[BLUE, BLUE, BLUE, BLUE, ACCENT], width=0.62)
    for b, v in zip(bars, cost):
        ax.text(b.get_x() + b.get_width() / 2, v + 7, f"{v:.0f}",
                ha="center", va="bottom", color=INK, fontsize=17)
    ax.axhline(127, color="#E45756", linewidth=2.4, linestyle="--")
    ax.text(4.42, 127, "Price per exposure\n≈ 127 CU (flat)", color="#E45756",
            fontsize=15, va="center", ha="right",
            bbox=dict(facecolor=BG, edgecolor="none", pad=2))
    ax.annotate("", xy=(4, 417.7), xytext=(0, 63.2),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(2.05, 300, "6.6× rise in claim cost\nacross flood zones",
            color=ACCENT, fontsize=17, weight="bold", ha="center", va="center")
    ax.set_xticks(range(5), zones, color=INK, fontsize=15)
    ax.set_ylabel("Observed claim cost per exposure (CU)", color=MUTED, fontsize=16)
    ax.set_ylim(0, 470)
    ax.tick_params(colors=MUTED, labelsize=13)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------- narration ---

async def synth(text: str, voice: str, mp3: Path, srt_words: Path):
    """Synthesise narration; capture word boundaries for caption timing.

    edge-tts >= 7 defaults to SentenceBoundary events — word-level timing must
    be requested explicitly via boundary="WordBoundary" (D-095).
    """
    import edge_tts
    comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    words = []
    with open(mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                words.append((chunk["offset"] / 1e7, chunk["duration"] / 1e7,
                              chunk["text"]))
    srt_words.write_text(json.dumps(words))


def audio_duration(mp3: Path) -> float:
    out = subprocess.run([FFMPEG, "-i", str(mp3)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out.stderr)
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def fmt_ass(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000:d}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,DejaVu Sans,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,2,0,2,60,60,42,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _sig(w: str) -> str:
    """Significant content of a word for alignment (punctuation-agnostic)."""
    return re.sub(r"[^A-Za-z0-9]", "", w).lower()


def _chunk_text(text: str, limit: int = 44) -> list[str]:
    """Split narration into caption-sized chunks at word boundaries,
    preferring sentence ends. Punctuation is preserved."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        cur = []
        for w in sent.split():
            if cur and len(" ".join(cur + [w])) > limit:
                out.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            out.append(" ".join(cur))
    return out


def _align_words(narration: str, words) -> list | None:
    """Map each whitespace token of `narration` to a (start, end) speech span.

    TTS word boundaries drop punctuation, so align on punctuation-stripped
    content. Returns None on any mismatch (caller falls back to proportional
    timing).
    """
    n_words = narration.split()
    sig_n = [i for i, w in enumerate(n_words) if _sig(w)]
    sig_t = [i for i, t in enumerate(words) if _sig(t[2])]
    if len(sig_n) != len(sig_t):
        return None
    spans: list = [None] * len(n_words)
    for ni, ti in zip(sig_n, sig_t):
        if _sig(n_words[ni]) != _sig(words[ti][2]):
            return None
        spans[ni] = (words[ti][0], words[ti][0] + words[ti][1])
    # pure-punctuation tokens inherit the neighbouring span
    last = None
    for i, sp in enumerate(spans):
        if sp is None:
            spans[i] = last
        else:
            last = sp
    for i in range(len(spans) - 1, -1, -1):
        if spans[i] is None:
            spans[i] = last
        else:
            last = spans[i]
    return spans


def build_captions(slides, timings, words_by_slide, srt: Path, ass: Path):
    """Group TTS word boundaries into caption lines, offset per slide.

    Writes SRT (reference/accessible copy) and ASS (burned in). The ASS carries
    explicit PlayRes 1920x1080 — libass renders bare SRT at its default 288p
    PlayRes, which inflates force_style font sizes ~3.75x at 1080p.
    """
    entries = []
    for spec, (start, dur), words in zip(slides, timings, words_by_slide):
        chunks = _chunk_text(spec["narration"])
        spans = _align_words(spec["narration"], words) if words else None
        idx, t = 0, start
        for ch in chunks:
            nw = len(ch.split())
            if spans is not None:
                a, b = spans[idx][0], spans[idx + nw - 1][1]
                t = start + b + 0.25
            else:  # fallback: spread each chunk proportionally to its length
                per = dur * len(ch) / max(len(spec["narration"]), 1)
                a, b = t - start, t - start + per
                t += per
            entries.append((start + a, t if spans is not None else start + b, ch))
            idx += nw

    # no two captions on screen at once: clamp end to the next entry's start
    for i in range(len(entries) - 1):
        a, b, text = entries[i]
        nxt = entries[i + 1][0]
        if b > nxt:
            entries[i] = (a, max(a + 0.05, nxt), text)
    with open(srt, "w") as f:
        for i, (a, b, text) in enumerate(entries, 1):
            f.write(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{text}\n\n")
    with open(ass, "w") as f:
        f.write(ASS_HEADER)
        for a, b, text in entries:
            f.write(f"Dialogue: 0,{fmt_ass(a)},{fmt_ass(b)},Caption,,0,0,0,,{text}\n")


# ----------------------------------------------------------------- build ---

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def main():
    OUT.mkdir(exist_ok=True)
    timings, words_by_slide = [], []

    for i, spec in enumerate(SLIDES, 1):
        png = OUT / f"slide_{i:02d}.png"
        mp3 = OUT / f"narr_{i:02d}.mp3"
        wj = OUT / f"words_{i:02d}.json"
        render_slide(spec, png)
        if not mp3.exists():
            asyncio.run(synth(spec["narration"], VOICE, mp3, wj))
        dur = audio_duration(mp3)
        words = json.loads(wj.read_text()) if wj.exists() else []
        timings.append((sum(t[1] for t in timings), dur + SEG_PAD))
        words_by_slide.append(words)
        print(f"slide {i:02d}: {dur:6.2f}s narration -> segment {dur + SEG_PAD:6.2f}s")

    # segments: still image + narration + trailing silence
    segs = []
    for i, spec in enumerate(SLIDES, 1):
        seg = OUT / f"seg_{i:02d}.mp4"
        png, mp3 = OUT / f"slide_{i:02d}.png", OUT / f"narr_{i:02d}.mp3"
        total = timings[i - 1][1]
        run([FFMPEG, "-y", "-loop", "1", "-framerate", "24", "-i", str(png),
             "-i", str(mp3), "-c:v", "libx264", "-tune", "stillimage",
             "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
             "-af", "apad", "-t", f"{total:.3f}", "-pix_fmt", "yuv420p",
             str(seg)])
        segs.append(seg)

    concat = OUT / "concat.txt"
    concat.write_text("".join(f"file '{s.name}'\n" for s in segs))
    merged = OUT / "merged.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", str(merged)])

    srt = OUT / "captions.srt"
    ass = OUT / "captions.ass"
    build_captions(SLIDES, timings, words_by_slide, srt, ass)

    final = ROOT / "pricing_across_markets.mp4"
    run([FFMPEG, "-y", "-i", str(merged), "-vf", f"subtitles={ass}",
         "-c:v", "libx264", "-crf", "20", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-c:a", "copy",
         "-metadata", "title=Pricing Across Markets — 2026 IAA AI-Assisted Hackathon",
         str(final)])

    total = sum(t[1] for t in timings)
    print(f"\ntotal: {total / 60:.1f} min ({total:.0f}s)")
    print(f"wrote {final} ({final.stat().st_size / 1e6:.1f} MB)")
    (OUT / "timings.json").write_text(json.dumps(
        {"slide_starts": [t[0] for t in timings], "durations": [t[1] for t in timings]},
        indent=1))


if __name__ == "__main__":
    main()
