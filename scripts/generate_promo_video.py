#!/usr/bin/env python3
"""Generate ToucanDB's repository promo video, poster, and original soundtrack.

Requires Pillow and ffmpeg. The soundtrack is synthesized locally by this script;
it does not include samples or third-party music.
"""

from __future__ import annotations

import argparse
import functools
import math
import subprocess
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH = 1920
HEIGHT = 1080
FPS = 24
DURATION = 18.0
FRAME_COUNT = int(FPS * DURATION)

ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = ROOT / "assets" / "toucandb-logo.png"

BG = "#071018"
PANEL = "#0E1B26"
PANEL_2 = "#132433"
CREAM = "#FFF8E9"
WHITE = "#F6FAFC"
MUTED = "#9CB0BD"
ORANGE = "#FF8A19"
YELLOW = "#FFD15C"
CYAN = "#54D6D1"
GREEN = "#65D68B"
RED = "#FF6B67"

FONT_REGULAR = Path("/System/Library/Fonts/Avenir Next.ttc")
FONT_BOLD = Path("/System/Library/Fonts/Avenir Next.ttc")
FONT_MONO = Path("/System/Library/Fonts/Menlo.ttc")


@functools.cache
def font(
    size: int, *, bold: bool = False, mono: bool = False
) -> ImageFont.FreeTypeFont:
    path = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REGULAR)
    index = 0 if bold or mono else 7
    try:
        return ImageFont.truetype(str(path), size=size, index=index)
    except OSError:
        return ImageFont.load_default(size=size)


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def ease(value: float) -> float:
    value = clamp(value)
    return 1 - (1 - value) ** 3


def phase(t: float, start: float, end: float) -> float:
    return clamp((t - start) / (end - start))


def opacity_window(t: float, start: float, end: float, fade: float = 0.45) -> float:
    return min(ease(phase(t, start, start + fade)), ease(phase(end - t, 0, fade)))


def with_alpha(color: str, alpha: int) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def draw_text(
    layer: Image.Image,
    xy: tuple[float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str,
    alpha: float = 1.0,
    anchor: str = "la",
) -> None:
    ImageDraw.Draw(layer).text(
        xy,
        text,
        font=text_font,
        fill=with_alpha(fill, int(255 * clamp(alpha))),
        anchor=anchor,
    )


def draw_chip(
    layer: Image.Image,
    xy: tuple[int, int],
    label: str,
    accent: str,
    alpha: float = 1.0,
) -> int:
    draw = ImageDraw.Draw(layer)
    chip_font = font(28, bold=True)
    box = draw.textbbox((0, 0), label, font=chip_font)
    width = box[2] - box[0] + 76
    x, y = xy
    a = int(255 * clamp(alpha))
    draw.rounded_rectangle(
        (x, y, x + width, y + 58), radius=29, fill=with_alpha(PANEL_2, a)
    )
    draw.ellipse((x + 22, y + 21, x + 38, y + 37), fill=with_alpha(accent, a))
    draw_text(layer, (x + 52, y + 29), label, chip_font, WHITE, alpha, anchor="lm")
    return width


def make_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    pixels = image.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            orange_glow = max(0.0, 1.0 - math.hypot(x - 1640, y - 90) / 980)
            teal_glow = max(0.0, 1.0 - math.hypot(x - 120, y - 1000) / 1050)
            vignette = max(0.0, math.hypot(x - WIDTH / 2, y - HEIGHT / 2) / 1220)
            r = int(7 + orange_glow * 24 + teal_glow * 2 - vignette * 3)
            g = int(16 + orange_glow * 8 + teal_glow * 20 - vignette * 5)
            b = int(24 + orange_glow * 1 + teal_glow * 24 - vignette * 5)
            pixels[x, y] = (max(0, r), max(0, g), max(0, b))
    return image


def place_logo(
    layer: Image.Image, x: int, y: int, size: int, alpha: float = 1.0
) -> None:
    logo = (
        Image.open(LOGO_PATH)
        .convert("RGBA")
        .resize((size, size), Image.Resampling.LANCZOS)
    )
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size, size), radius=int(size * 0.18), fill=int(255 * alpha)
    )
    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (x + 10, y + 18, x + size + 10, y + size + 18),
        radius=int(size * 0.18),
        fill=(0, 0, 0, int(90 * alpha)),
    )
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
    logo.putalpha(mask)
    layer.alpha_composite(logo, (x, y))


def draw_window(
    layer: Image.Image, box: tuple[int, int, int, int], alpha: float = 1.0
) -> None:
    x1, y1, x2, y2 = box
    draw = ImageDraw.Draw(layer)
    a = int(255 * alpha)
    draw.rounded_rectangle(
        box,
        radius=34,
        fill=with_alpha(PANEL, a),
        outline=with_alpha("#294150", a),
        width=2,
    )
    draw.rounded_rectangle(
        (x1, y1, x2, y1 + 78), radius=34, fill=with_alpha("#142431", a)
    )
    draw.rectangle((x1, y1 + 43, x2, y1 + 78), fill=with_alpha("#142431", a))
    for index, color in enumerate((RED, YELLOW, GREEN)):
        cx = x1 + 38 + index * 31
        draw.ellipse((cx, y1 + 29, cx + 16, y1 + 45), fill=with_alpha(color, a))
    draw_text(
        layer,
        (x1 + 152, y1 + 39),
        "ToucanDB Studio",
        font(25, bold=True),
        MUTED,
        alpha,
        anchor="lm",
    )


def scene_hero(t: float, layer: Image.Image) -> None:
    alpha = opacity_window(t, 0.0, 4.2, 0.55)
    if alpha <= 0:
        return
    progress = ease(phase(t, 0.0, 1.0))
    offset = int((1 - progress) * 42)
    place_logo(layer, 226 - offset, 280, 330, alpha)
    draw_text(
        layer, (640 + offset, 330), "ToucanDB", font(112, bold=True), WHITE, alpha
    )
    draw_text(
        layer,
        (646 + offset, 462),
        "Vector memory that lives with your app.",
        font(48),
        CREAM,
        alpha,
    )
    draw_text(
        layer,
        (646 + offset, 538),
        "Embedded  •  Local-first  •  RAG-ready",
        font(33, bold=True),
        CYAN,
        alpha,
    )
    draw_chip(layer, (648 + offset, 630), "No database server", ORANGE, alpha)
    draw_chip(layer, (1020 + offset, 630), "Python 3.10+", CYAN, alpha)


def scene_flow(t: float, layer: Image.Image) -> None:
    alpha = opacity_window(t, 3.5, 8.6)
    if alpha <= 0:
        return
    draw_text(
        layer,
        (960, 142),
        "Built into your application",
        font(62, bold=True),
        WHITE,
        alpha,
        anchor="ma",
    )
    draw_text(
        layer,
        (960, 215),
        "Durable source of truth. Fast vector retrieval.",
        font(32),
        MUTED,
        alpha,
        anchor="ma",
    )
    draw_window(layer, (174, 284, 1746, 900), alpha)
    draw = ImageDraw.Draw(layer)

    cards = [
        (270, 425, 585, 712, "YOUR APP", "Documents\nSignals\nEmbeddings", CYAN),
        (
            802,
            387,
            1118,
            750,
            "TOUCANDB",
            "Atomic writes\nMetadata filters\nNamespace isolation",
            ORANGE,
        ),
        (
            1335,
            425,
            1650,
            712,
            "RESULTS",
            "Relevant context\nSource attribution\nGrounded answers",
            GREEN,
        ),
    ]
    reveal = [phase(t, 3.8, 4.5), phase(t, 4.45, 5.2), phase(t, 5.8, 6.55)]
    for idx, (x1, y1, x2, y2, title, body, accent) in enumerate(cards):
        card_alpha = alpha * ease(reveal[idx])
        y_offset = int((1 - ease(reveal[idx])) * 30)
        y1 += y_offset
        y2 += y_offset
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=26,
            fill=with_alpha(PANEL_2, int(255 * card_alpha)),
            outline=with_alpha(accent, int(150 * card_alpha)),
            width=2,
        )
        draw.rounded_rectangle(
            (x1 + 25, y1 + 25, x1 + 91, y1 + 91),
            radius=18,
            fill=with_alpha(accent, int(255 * card_alpha)),
        )
        icon = ("{}", "DB", "OK")[idx]
        draw_text(
            layer,
            (x1 + 58, y1 + 58),
            icon,
            font(21, bold=True, mono=idx < 2),
            BG,
            card_alpha,
            anchor="mm",
        )
        draw_text(
            layer, (x1 + 28, y1 + 126), title, font(27, bold=True), accent, card_alpha
        )
        for line_index, line in enumerate(body.splitlines()):
            draw_text(
                layer,
                (x1 + 28, y1 + 172 + line_index * 40),
                line,
                font(24),
                WHITE,
                card_alpha,
            )

    flow_progress = ease(phase(t, 4.7, 6.2))
    for x1, x2 in ((594, 792), (1128, 1325)):
        end = int(x1 + (x2 - x1) * flow_progress)
        draw.line(
            (x1, 570, end, 570), fill=with_alpha(ORANGE, int(255 * alpha)), width=5
        )
        if flow_progress > 0.95:
            draw.polygon(
                ((x2, 570), (x2 - 20, 557), (x2 - 20, 583)),
                fill=with_alpha(ORANGE, int(255 * alpha)),
            )
    if t > 6.7:
        label_alpha = alpha * ease(phase(t, 6.7, 7.25))
        draw.rounded_rectangle(
            (698, 792, 1222, 856),
            radius=32,
            fill=with_alpha("#15392F", int(255 * label_alpha)),
        )
        draw_text(
            layer,
            (960, 824),
            "SQLite WAL  +  FAISS acceleration",
            font(27, bold=True),
            GREEN,
            label_alpha,
            anchor="mm",
        )


def scene_code(t: float, layer: Image.Image) -> None:
    alpha = opacity_window(t, 8.0, 13.1)
    if alpha <= 0:
        return
    draw_text(
        layer,
        (960, 138),
        "From vectors to grounded context",
        font(61, bold=True),
        WHITE,
        alpha,
        anchor="ma",
    )
    draw_text(
        layer,
        (960, 210),
        "A production-ready retrieval pipeline in a few lines.",
        font(31),
        MUTED,
        alpha,
        anchor="ma",
    )
    draw_window(layer, (214, 278, 1706, 910), alpha)
    draw = ImageDraw.Draw(layer)
    draw.line(
        (1040, 357, 1040, 850), fill=with_alpha("#294150", int(255 * alpha)), width=2
    )
    lines = [
        ("from toucandb.integrations import RAGStore", CYAN),
        ("", WHITE),
        ("rag = await RAGStore.create(", WHITE),
        ('    "./knowledge.tdb", embeddings,', YELLOW),
        ('    namespace="product-docs",', YELLOW),
        (")", WHITE),
        ("", WHITE),
        ('hits = await rag.retrieve("How does it scale?")', GREEN),
    ]
    line_delay = 0.18
    for idx, (text, color) in enumerate(lines):
        line_alpha = alpha * ease(
            phase(t, 8.3 + idx * line_delay, 8.65 + idx * line_delay)
        )
        draw_text(
            layer, (280, 385 + idx * 53), text, font(25, mono=True), color, line_alpha
        )

    result_alpha = alpha * ease(phase(t, 10.0, 10.65))
    draw_text(
        layer, (1100, 392), "TOP MATCHES", font(24, bold=True), ORANGE, result_alpha
    )
    results = [
        ("0.94", "SQLite persists every record atomically.", "architecture.md"),
        ("0.89", "FAISS accelerates in-memory similarity search.", "performance.md"),
        ("0.86", "One process owns each database directory.", "deployment.md"),
    ]
    for idx, (score, body, source) in enumerate(results):
        item_alpha = result_alpha * ease(
            phase(t, 10.2 + idx * 0.25, 10.65 + idx * 0.25)
        )
        top = 438 + idx * 126
        draw.rounded_rectangle(
            (1090, top, 1630, top + 104),
            radius=20,
            fill=with_alpha(PANEL_2, int(255 * item_alpha)),
        )
        draw_text(
            layer, (1120, top + 30), score, font(26, bold=True), GREEN, item_alpha
        )
        draw_text(layer, (1194, top + 29), body, font(22), WHITE, item_alpha)
        draw_text(
            layer, (1194, top + 70), source, font(19, mono=True), MUTED, item_alpha
        )


def scene_features(t: float, layer: Image.Image) -> None:
    alpha = opacity_window(t, 12.5, 15.8)
    if alpha <= 0:
        return
    draw_text(
        layer,
        (960, 193),
        "Small footprint. Serious foundations.",
        font(67, bold=True),
        WHITE,
        alpha,
        anchor="ma",
    )
    draw_text(
        layer,
        (960, 272),
        "Choose exact, HNSW, or IVF search as your corpus grows.",
        font(32),
        MUTED,
        alpha,
        anchor="ma",
    )
    labels = [
        ("Atomic batches", "SQLite-backed durability", ORANGE),
        ("Bounded memory", "LRU budgets and compaction", CYAN),
        ("Private by design", "Optional authenticated encryption", GREEN),
        ("Framework-neutral", "Bring any embedding or LLM", YELLOW),
    ]
    positions = ((260, 390), (990, 390), (260, 650), (990, 650))
    draw = ImageDraw.Draw(layer)
    for idx, ((title, subtitle, accent), (x, y)) in enumerate(
        zip(labels, positions, strict=True)
    ):
        item_alpha = alpha * ease(phase(t, 12.75 + idx * 0.18, 13.35 + idx * 0.18))
        draw.rounded_rectangle(
            (x, y, x + 670, y + 206),
            radius=28,
            fill=with_alpha(PANEL, int(255 * item_alpha)),
            outline=with_alpha("#294150", int(255 * item_alpha)),
            width=2,
        )
        draw.rounded_rectangle(
            (x + 34, y + 48, x + 106, y + 120),
            radius=20,
            fill=with_alpha(accent, int(255 * item_alpha)),
        )
        draw_text(
            layer,
            (x + 70, y + 84),
            ("TX", "LRU", "AES", "ANY")[idx],
            font(18, bold=True, mono=True),
            BG,
            item_alpha,
            anchor="mm",
        )
        draw_text(
            layer, (x + 140, y + 63), title, font(33, bold=True), WHITE, item_alpha
        )
        draw_text(layer, (x + 140, y + 116), subtitle, font(25), MUTED, item_alpha)


def scene_outro(t: float, layer: Image.Image) -> None:
    alpha = ease(phase(t, 15.0, 15.7))
    if alpha <= 0:
        return
    place_logo(layer, 820, 130, 280, alpha)
    draw_text(
        layer,
        (960, 492),
        "Own your vector memory.",
        font(79, bold=True),
        WHITE,
        alpha,
        anchor="ma",
    )
    draw_text(
        layer,
        (960, 585),
        "One process. Zero database servers.",
        font(37),
        CREAM,
        alpha,
        anchor="ma",
    )
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        (610, 675, 1310, 765),
        radius=28,
        fill=with_alpha("#0A0E12", int(255 * alpha)),
        outline=with_alpha(ORANGE, int(255 * alpha)),
        width=2,
    )
    draw_text(
        layer,
        (960, 720),
        "$  pip install toucandb",
        font(30, mono=True),
        GREEN,
        alpha,
        anchor="mm",
    )
    draw_text(
        layer,
        (960, 850),
        "github.com/ToucanDB/ToucanDB",
        font(27, bold=True),
        CYAN,
        alpha,
        anchor="ma",
    )


def draw_progress(layer: Image.Image, t: float) -> None:
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle(
        (142, 1012, 1778, 1018), radius=3, fill=with_alpha("#28404D", 120)
    )
    draw.rounded_rectangle(
        (142, 1012, 142 + int(1636 * t / DURATION), 1018),
        radius=3,
        fill=with_alpha(ORANGE, 230),
    )


def render_frame(t: float, background: Image.Image) -> Image.Image:
    frame = background.copy().convert("RGBA")
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    scene_hero(t, layer)
    scene_flow(t, layer)
    scene_code(t, layer)
    scene_features(t, layer)
    scene_outro(t, layer)
    draw_progress(layer, t)
    frame.alpha_composite(layer)
    return frame.convert("RGB")


def make_poster(background: Image.Image) -> Image.Image:
    poster = background.copy().convert("RGBA")
    layer = Image.new("RGBA", poster.size, (0, 0, 0, 0))
    draw_text(layer, (960, 102), "ToucanDB", font(82, bold=True), WHITE, anchor="ma")
    draw_text(
        layer,
        (960, 188),
        "Embedded vector search for RAG and application memory",
        font(34),
        CREAM,
        anchor="ma",
    )
    draw_window(layer, (185, 260, 1735, 910))
    draw = ImageDraw.Draw(layer)
    place_logo(layer, 250, 374, 350)
    draw_text(layer, (690, 414), "Your vectors.", font(62, bold=True), WHITE)
    draw_text(layer, (690, 505), "Your process.", font(62, bold=True), WHITE)
    draw_text(layer, (690, 596), "No database server.", font(62, bold=True), ORANGE)
    x = 692
    for label, accent in (
        ("SQLite durability", GREEN),
        ("FAISS speed", CYAN),
        ("RAG ready", YELLOW),
    ):
        x += draw_chip(layer, (x, 704), label, accent) + 18
    # Familiar play affordance used by GitHub video thumbnails.
    shadow = Image.new("RGBA", poster.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((1358, 438, 1538, 618), fill=(0, 0, 0, 100))
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
    draw.ellipse((1350, 426, 1530, 606), fill=with_alpha(ORANGE, 245))
    draw.polygon(((1422, 468), (1422, 566), (1496, 517)), fill=with_alpha(WHITE, 255))
    draw_text(
        layer, (1440, 651), "WATCH  •  18 SEC", font(22, bold=True), MUTED, anchor="ma"
    )
    poster.alpha_composite(layer)
    return poster.convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)


def synthesize_music(path: Path) -> None:
    sample_rate = 48_000
    total_samples = int(DURATION * sample_rate)
    chords = [
        (130.81, 164.81, 196.00),
        (110.00, 130.81, 164.81),
        (87.31, 110.00, 130.81),
        (98.00, 123.47, 146.83),
    ]
    melody = [261.63, 329.63, 392.00, 493.88, 392.00, 329.63, 293.66, 261.63]
    fade_samples = int(1.0 * sample_rate)
    frames = bytearray()
    for index in range(total_samples):
        current = index / sample_rate
        chord = chords[int(current / 2.25) % len(chords)]
        pad = sum(
            math.sin(2 * math.pi * frequency * current + voice * 0.7)
            for voice, frequency in enumerate(chord)
        ) / len(chord)
        pad += (
            0.28
            * sum(
                math.sin(2 * math.pi * frequency * 2 * current) for frequency in chord
            )
            / len(chord)
        )

        beat = current % 0.5625
        kick = math.sin(
            2 * math.pi * (54 + 75 * math.exp(-beat * 20)) * beat
        ) * math.exp(-beat * 13)
        snare_beat = (current + 0.28125) % 1.125
        noise = math.sin(index * 12.9898) * math.sin(index * 78.233)
        snare = noise * math.exp(-snare_beat * 20) if snare_beat < 0.18 else 0.0

        note_position = (current % 4.5) / 0.5625
        note_index = int(note_position) % len(melody)
        note_age = (note_position - int(note_position)) * 0.5625
        pluck = math.sin(2 * math.pi * melody[note_index] * current) * math.exp(
            -note_age * 7.0
        )
        pluck += (
            0.22
            * math.sin(2 * math.pi * melody[note_index] * 2 * current)
            * math.exp(-note_age * 9.0)
        )

        envelope = min(
            1.0, index / fade_samples, (total_samples - index) / fade_samples
        )
        value = envelope * (0.18 * pad + 0.13 * kick + 0.035 * snare + 0.075 * pluck)
        left = int(max(-1.0, min(1.0, value * 0.97)) * 32767)
        right = int(max(-1.0, min(1.0, value * 1.03)) * 32767)
        frames.extend(left.to_bytes(2, "little", signed=True))
        frames.extend(right.to_bytes(2, "little", signed=True))

    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(frames)


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "toucandb-promo.mp4"
    poster_path = output_dir / "toucandb-promo-poster.png"
    background = make_background()
    make_poster(background).save(poster_path, optimize=True)

    with tempfile.TemporaryDirectory(prefix="toucandb-promo-") as temporary:
        audio_path = Path(temporary) / "original-soundtrack.wav"
        synthesize_music(audio_path)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            str(FPS),
            "-i",
            "-",
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-af",
            "volume=2.5dB",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        try:
            for frame_index in range(FRAME_COUNT):
                frame = render_frame(frame_index / FPS, background)
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed to encode the promo video")

    print(f"Generated {output_path}")
    print(f"Generated {poster_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "assets" / "promo",
        help="Directory for the MP4 and poster",
    )
    args = parser.parse_args()
    generate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
