#!/usr/bin/env python3
"""Regenerate the fixture images for the ci-visual-regression example.

Writes tiny synthetic "screenshots" (flat UI mockups, no fonts needed) into
fixtures/baseline/ and fixtures/screenshots/:

  home.png      nearly identical pair (a 2 px nudge on one element stands in
  detail.png    for font/AA jitter between runs)
  settings.png  clearly different, screenshots-only: the gate reports it as
                NEW instead of failing, so the proof run stays green

Run: python3 examples/ci-visual-regression/make_fixtures.py  (needs Pillow)
"""

from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
W, H = 400, 300

BG = (245, 246, 248)
CARD = (255, 255, 255)
BORDER = (210, 214, 220)
TEXT_DARK = (90, 98, 110)
TEXT_MID = (160, 168, 180)
ACCENT = (66, 133, 244)


def nav(d, accent=ACCENT):
    d.rectangle([0, 0, W, 40], fill=accent)
    for i in range(3):
        x = 16 + i * 80
        d.rectangle([x, 14, x + 56, 26], fill=CARD)


def card(d, box):
    d.rectangle(box, fill=CARD, outline=BORDER)


def home(jitter=0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    nav(d)
    card(d, [24, 64, 376, 140])
    d.rectangle([40, 84 + jitter, 240, 96 + jitter], fill=TEXT_DARK)
    d.rectangle([40, 108, 320, 118], fill=TEXT_MID)
    card(d, [24, 160, 190, 276])
    card(d, [210, 160, 376, 276])
    d.rectangle([40, 184, 100, 196], fill=ACCENT)
    d.rectangle([226, 184, 286, 196], fill=(234, 67, 53))
    d.rectangle([40, 208 + jitter, 160, 218 + jitter], fill=TEXT_MID)
    d.rectangle([226, 208, 350, 218], fill=TEXT_MID)
    d.rectangle([40, 232, 150, 242], fill=TEXT_MID)
    d.rectangle([226, 232, 340, 242], fill=TEXT_MID)
    return img


def detail(jitter=0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    nav(d, accent=(52, 168, 83))
    card(d, [24, 64, 240, 276])
    d.rectangle([40, 88, 224, 200], fill=(52, 168, 83))
    d.rectangle([40, 216 + jitter, 180, 228 + jitter], fill=TEXT_DARK)
    d.rectangle([40, 240, 200, 250], fill=TEXT_MID)
    card(d, [256, 64, 376, 276])
    for i in range(4):
        y = 84 + i * 44
        d.ellipse([272, y, 296, y + 24], fill=ACCENT)
        d.rectangle([308, y + 4, 364, y + 12], fill=TEXT_DARK)
        d.rectangle([308, y + 16, 352, y + 22], fill=TEXT_MID)
    return img


def settings_new():
    """A clearly different page: dark theme, sidebar layout, no baseline pair."""
    img = Image.new("RGB", (W, H), (24, 26, 32))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 96, H], fill=(32, 35, 44))
    for i in range(5):
        d.rectangle([16, 24 + i * 52, 80, 44 + i * 52], fill=(70, 76, 94))
    d.rectangle([112, 24, 376, 56], fill=(200, 205, 216))
    for i in range(3):
        y = 80 + i * 68
        d.rectangle([112, y, 376, y + 52], fill=(38, 42, 52))
        d.ellipse([128, y + 14, 152, y + 38], fill=(251, 188, 5))
        d.rectangle([168, y + 12, 300, y + 22], fill=(150, 156, 170))
        d.rectangle([168, y + 30, 260, y + 38], fill=(96, 102, 116))
    return img


def main():
    pairs = {
        "baseline": {"home.png": home(), "detail.png": detail()},
        "screenshots": {
            "home.png": home(jitter=2),
            "detail.png": detail(jitter=-2),
            "settings.png": settings_new(),
        },
    }
    for sub, images in pairs.items():
        out = HERE / "fixtures" / sub
        out.mkdir(parents=True, exist_ok=True)
        for name, img in images.items():
            img.save(out / name)
            print(f"wrote {out.relative_to(HERE) / name}")


if __name__ == "__main__":
    main()
