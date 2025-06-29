# r/vision - Visual Effects Tool

<div align="center">

![r/vision Logo](https://img.shields.io/badge/r%2Fvision-v2.0.0-orange?style=for-the-badge&logo=python)

**🎬 Add Tracking Effects to Your Videos**

</div>

## What It Does

r/vision adds colorful tracking boxes around objects in your videos. Perfect for:

- **Social Media** - Eye-catching effects for TikTok, Instagram, YouTube

## Installation

```bash
# Clone or download this repo
git clone <repository-url>
cd r-vision

# Install requirements
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python r_vision.py your_video.mp4

# Save with custom name
python r_vision.py input.mp4 --output enhanced.mp4

# More sensitive tracking (detects more objects)
python r_vision.py video.mp4 --yolo-confidence 0.1

# Less sensitive (only main objects)
python r_vision.py video.mp4 --yolo-confidence 0.7
```

## Features

- **Color-coded boxes** - Different colors for people, cars, electronics, etc.
- **Thin, modern design** - Clean HUD-style overlays
- **Live stats** - Shows confidence and FPS on each box
- **Customizable** - Adjust sensitivity and colors

## Options

| Option | Description | Example |
|--------|-------------|---------|
| `--output` | Output filename | `--output cool_video.mp4` |
| `--yolo-confidence` | Sensitivity (0.1-1.0) | `--yolo-confidence 0.3` |
| `--no-yolo` | Only track people | `--no-yolo` |
| `--no-pose` | Only track objects | `--no-pose` |

## Requirements

- Python 3.8+
- 4GB+ RAM

## Colors

- 🟢 **Green**: People
- 🔴 **Red**: Vehicles (cars, bikes, planes)
- 🟣 **Purple**: Animals
- 🔵 **Blue**: Electronics
- 🟠 **Orange**: Food & drinks
- 🟡 **Yellow**: Furniture & objects

## For Developers

Want to customize? Edit the colors in `r_vision.py`:

```python
self.specific_colors = {
    'person': (0, 255, 255),    # Cyan for people
    'car': (255, 0, 255),       # Pink for cars
    'laptop': (0, 255, 0),      # Green for laptops
}
```

## License

MIT License - use it however you want!

---

<div align="center">
Made with ❤️ for content creators
</div>
