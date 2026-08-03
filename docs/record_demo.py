"""Record docs/demo.gif: a real install-and-run session, captured then drawn.

The demo makes the same claim the README makes, so it is generated the same way
the results are: by running the thing and recording what came back. Nothing in
the frames is typed by hand. The script builds a fresh virtual environment,
installs the published package from PyPI, runs the sparse benchmark, and keeps
every line the two processes printed along with the second it appeared at.

Only the waiting is edited. Idle gaps are capped at 0.9 s, which turns a session
of about a minute into about seventeen seconds, and that cap is the single
reason the recording is shorter than the run. Line order, line content and the
numbers are untouched.

    pip install "rag-ablations[demo]"
    python docs/record_demo.py --workdir C:\\demo

The committed GIF was recorded with that exact command. --workdir decides the
path the benchmark prints in its last line, so a short one keeps the frame tidy;
it defaults to .demo-run under the current directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DOCS = Path(__file__).resolve().parent

# Terminal geometry, in characters and pixels.
COLS = 96
ROWS = 21
PAD = 18
BAR = 30
LINE_H = 21
FONT_SIZE = 15

BG = (13, 15, 20)
BAR_BG = (22, 25, 32)
DOT = (58, 64, 76)
TITLE = (124, 133, 148)
DIM = (148, 158, 172)
TEXT = (226, 232, 240)
PROMPT = (126, 213, 145)
HILITE = (125, 205, 233)

MAX_GAP = 0.9
MIN_GAP = 0.06
TYPE_STEP = 0.045
TYPE_CHUNK = 3
HOLD_END = 3.0

FONT_CANDIDATES = [
    (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\consolab.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    ("/Library/Fonts/Menlo.ttc", "/Library/Fonts/Menlo.ttc"),
]


def load_fonts():
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists() and Path(bold).exists():
            return (ImageFont.truetype(regular, FONT_SIZE),
                    ImageFont.truetype(bold, FONT_SIZE))
    raise SystemExit(
        "No monospace TrueType font found. Add one to FONT_CANDIDATES; a "
        "proportional fallback would misalign the results table."
    )


def capture(workdir: Path) -> list[dict]:
    """Run the two commands for real and keep every line with its timestamp."""
    venv = workdir / ".venv"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)

    commands = [
        ("pip install rag-ablations",
         [str(python), "-m", "pip", "install", "--no-cache-dir",
          "--disable-pip-version-check", "rag-ablations"]),
        ("python -m rag_ablations.benchmark --dataset scifact --systems sparse",
         [str(python), "-m", "rag_ablations.benchmark",
          "--dataset", "scifact", "--systems", "sparse"]),
    ]

    transcript = []
    for display, argv in commands:
        block = {"cmd": display, "lines": []}
        start = time.monotonic()
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, cwd=str(workdir), env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in proc.stdout:
            elapsed = round(time.monotonic() - start, 3)
            block["lines"].append({"t": elapsed, "line": line.rstrip("\r\n")})
            print(f"[{elapsed:7.3f}] {line.rstrip()}", flush=True)
        if proc.wait() != 0:
            raise SystemExit(f"{display} exited {proc.returncode}; not recording a failed run")
        transcript.append(block)
    return transcript


def style(kind: str, text: str, font, font_bold):
    """Colour is presentation only. It never changes what a line says."""
    if kind == "prompt":
        return TEXT, font_bold
    if text.startswith("  nDCG@10") or text.startswith("  5183 "):
        return HILITE, font_bold
    if text.startswith("|") or text.startswith("Successfully installed"):
        return TEXT, font
    return DIM, font


def draw(lines, cursor_on, font, font_bold, char_w, size):
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0], BAR], fill=BAR_BG)
    for x in (16, 32, 48):
        d.ellipse([x - 4, BAR // 2 - 4, x + 4, BAR // 2 + 4], fill=DOT)
    d.text((72, BAR // 2), "rag-ablations", font=font, fill=TITLE, anchor="lm")

    visible = lines[-ROWS:]
    y = BAR + PAD
    for kind, text in visible:
        x = PAD
        if kind == "prompt":
            d.text((x, y), "$", font=font_bold, fill=PROMPT)
            x += char_w * 2
        colour, chosen = style(kind, text, font, font_bold)
        d.text((x, y), text[:COLS], font=chosen, fill=colour)
        if cursor_on and (kind, text) is visible[-1]:
            cx = x + char_w * len(text[:COLS])
            d.rectangle([cx + 1, y + 2, cx + char_w - 1, y + LINE_H - 4], fill=TEXT)
        y += LINE_H
    return img


def render(transcript, out: Path):
    font, font_bold = load_fonts()
    char_w = font.getlength("M")
    size = (int(PAD * 2 + char_w * COLS), PAD * 2 + BAR + LINE_H * ROWS)

    frames, delays, lines = [], [], []

    def push(seconds, cursor=True):
        frames.append(draw(lines, cursor, font, font_bold, char_w, size))
        delays.append(max(0.02, seconds))

    push(0.7)
    for index, block in enumerate(transcript):
        cmd = block["cmd"]
        lines.append(("prompt", ""))
        for i in range(0, len(cmd) + 1, TYPE_CHUNK):
            lines[-1] = ("prompt", cmd[:i])
            push(TYPE_STEP)
        lines[-1] = ("prompt", cmd)
        push(0.45)

        previous = 0.0
        for record in block["lines"]:
            gap = min(MAX_GAP, max(MIN_GAP, record["t"] - previous))
            previous = record["t"]
            lines.append(("out", record["line"]))
            push(gap, cursor=False)

        if index < len(transcript) - 1:
            lines.append(("out", ""))
            push(0.5, cursor=False)
    push(HOLD_END, cursor=False)

    palette = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=32) for f in frames]
    palette[0].save(
        out, save_all=True, append_images=palette[1:],
        duration=[int(round(s * 1000)) for s in delays],
        loop=0, optimize=True, disposal=1,
    )
    print(f"{out}  {size[0]}x{size[1]}  {len(frames)} frames  "
          f"{sum(delays):.1f}s  {out.stat().st_size / 1024:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=".demo-run", type=Path,
                        help="where the recorded session runs; its path shows in the last frame")
    parser.add_argument("--out", default=DOCS / "demo.gif", type=Path)
    parser.add_argument("--transcript", type=Path,
                        help="render a saved transcript instead of running the commands again")
    args = parser.parse_args()

    if args.transcript:
        transcript = json.loads(args.transcript.read_text(encoding="utf-8"))
    else:
        transcript = capture(args.workdir.resolve())
        # Committed next to the GIF so the frames can be checked against the
        # lines the processes actually printed.
        args.out.with_name(args.out.stem + "-transcript.json").write_text(
            json.dumps(transcript, indent=2), encoding="utf-8")
    render(transcript, args.out)


if __name__ == "__main__":
    main()
