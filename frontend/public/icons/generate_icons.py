"""Run once to generate placeholder PNG icons for PWA.
Requires Pillow: pip install Pillow
Usage: python generate_icons.py
"""

from PIL import Image, ImageDraw

def make_icon(size: int, path: str) -> None:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    margin = size // 6
    draw.ellipse([margin, margin, size - margin, size - margin], fill=(59, 130, 246, 255))
    inner = size // 3
    draw.ellipse([inner, inner, size - inner, size - inner], fill=(0, 0, 0, 200))
    img.save(path)
    print(f"Saved {path}")

if __name__ == "__main__":
    make_icon(192, "icon-192.png")
    make_icon(512, "icon-512.png")
    make_icon(180, "apple-touch-icon.png")
